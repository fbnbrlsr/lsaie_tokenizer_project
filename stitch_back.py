import argparse
import os
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from vision_tokenization.pipelines.indexed_dataset_megatron import IndexedDatasetBuilder
from vision_tokenization.pipelines.dataset import IndexedDataset
from tqdm import tqdm

class LateBindingEncapsulator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Cache IDs for the NEW tokenizer
        self.img_start = tokenizer.convert_tokens_to_ids("<|img_start|>")
        self.img_end = tokenizer.convert_tokens_to_ids("<|img_end|>")
        self.img_tok_start = tokenizer.convert_tokens_to_ids("<|img_token_start|>")
        self.eol = tokenizer.convert_tokens_to_ids("<|img_end_of_row|>")
        self.eof = tokenizer.convert_tokens_to_ids("<|img_end_of_frame|>")
        self.offset = tokenizer.convert_tokens_to_ids("<|visual token 000000|>")
        
        if self.offset is None:
            raise ValueError("New tokenizer missing visual tokens!")

    def encapsulate(self, raw_indices):
        # 1. Calc Dimensions (Assume Square)
        n = len(raw_indices)
        if n == 0: return []
        side = int(np.sqrt(n))
        
        # 2. Build Structure
        dim_tokens = self.tokenizer.encode(f"{side}*{side}", add_special_tokens=False)
        res = [self.img_start] + dim_tokens + [self.img_tok_start]
        
        # Body: Vision + Offset + EOLs
        shifted = raw_indices + self.offset
        for r in range(side):
            row = shifted[r*side : (r+1)*side].tolist()
            res.extend(row)
            res.append(self.eol)
            
        # Footer
        res.extend([self.eof, self.img_end])
        return res

def stitch_shard(raw_img_path, raw_text_path, output_path, tokenizer, encapsulator):
    # Load Raw Images
    img_ds = IndexedDataset(str(raw_img_path).replace('.bin', ''))
    
    # Prepare Output
    builder = IndexedDatasetBuilder(str(output_path))
    
    # Read Raw Text Line by Line
    with open(raw_text_path, 'r') as f:
        # Use tqdm to see progress
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc=f"Stitching {Path(raw_img_path).stem}")):
            data = json.loads(line)
            
            # 1. Re-Tokenize Text
            t_before = tokenizer.encode(data['text_before'], add_special_tokens=False)
            t_after = tokenizer.encode(data['text_after'], add_special_tokens=False)
            
            # 2. Encapsulate Image
            if i < len(img_ds):
                raw_img = img_ds[i]
                if len(raw_img) > 0:
                    img_block = encapsulator.encapsulate(raw_img)
                else:
                    img_block = []
            else:
                img_block = []
            
            # 3. Combine [BOS] + Text + Image + Text + [EOS]
            combined = [tokenizer.bos_token_id] + t_before + img_block + t_after + [tokenizer.eos_token_id]
            
            # [FIX IS HERE] Use add_document instead of add_item
            builder.add_document(np.array(combined, dtype=np.int32), [len(combined)])
            
    builder.finalize(str(output_path).replace('.bin', '.idx'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Folder with raw_images and raw_texts")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--new_tokenizer_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.new_tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer_path, trust_remote_code=True)
    encapsulator = LateBindingEncapsulator(tokenizer)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each recycled shard
    raw_img_dir = Path(args.raw_data_dir) / "raw_images"
    raw_text_dir = Path(args.raw_data_dir) / "raw_texts"
    
    # Find bin files
    bin_files = sorted(list(raw_img_dir.glob("*.bin")))
    if not bin_files:
        print(f"No bin files found in {raw_img_dir}")
        return

    for bin_file in bin_files:
        shard_name = bin_file.stem
        text_file = raw_text_dir / f"{shard_name}.jsonl"
        output_file = Path(args.output_dir) / f"{shard_name}.bin"
        
        if text_file.exists():
            stitch_shard(bin_file, text_file, output_file, tokenizer, encapsulator)
        else:
            print(f"Warning: No matching text file for {shard_name}")

if __name__ == "__main__":
    main()