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
        
        # Standard Special Tokens
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        
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

def stitch_shard_sft(raw_img_path, raw_text_path, output_path, tokenizer, encapsulator):
    # Load Raw Images
    img_ds = IndexedDataset(str(raw_img_path).replace('.bin', ''))
    
    # Prepare Output
    builder = IndexedDatasetBuilder(str(output_path))
    
    # Read Raw Text Line by Line
    with open(raw_text_path, 'r') as f:
        # Use tqdm to see progress
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc=f"Stitching SFT {Path(raw_img_path).stem}")):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Error decoding JSON at line {i}")
                continue
            
            # 1. Re-Tokenize Text
            t_before = tokenizer.encode(data.get('text_before', ''), add_special_tokens=False)
            t_after = tokenizer.encode(data.get('text_after', ''), add_special_tokens=False)
            
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
            
            builder.add_document(np.array(combined, dtype=np.int32), [len(combined)])
            
    builder.finalize(str(output_path).replace('.bin', '.idx'))

def stitch_shard_image2text(text_path, img_path, output_path, tokenizer, encapsulator):
    text_ds = IndexedDataset(str(text_path).replace('.bin', ''))
    img_ds = IndexedDataset(str(img_path).replace('.bin', ''))
    
    builder = IndexedDatasetBuilder(str(output_path))
    
    for i in tqdm(range(len(img_ds)), desc=f"Stitching I2T {Path(img_path).stem}"):
        text_seq = text_ds[i]
        img_seq = img_ds[i]
        
        # Special token indices within the pre-structured sequence
        try:
            img_start_idx = np.where(img_seq == encapsulator.img_start)[0][0]
            img_tokens_start_idx = np.where(img_seq == encapsulator.img_tok_start)[0][0]
        except IndexError:
            # Skip malformed sequences
            continue
            
        # Selective Shifting: Add offset everywhere except at special tokens
        final_seq = np.where(
            (img_seq == 1) | (img_seq == 2) |
            (img_seq == encapsulator.img_start) |
            (img_seq == encapsulator.img_end) |
            (img_seq == encapsulator.img_tok_start) |
            (img_seq == encapsulator.eol) |
            (img_seq == encapsulator.eof),
            img_seq, 
            img_seq + encapsulator.offset
        )
        
        # Remove offset at dimension encoding (digits between img_start and tok_start)
        final_seq[img_start_idx+1:img_tokens_start_idx] -= encapsulator.offset
        
        # Append text tokens
        final_seq = np.concatenate([final_seq, text_seq])
        
        builder.add_document(np.array(final_seq, dtype=np.int32), [len(final_seq)])
        
    builder.finalize(str(output_path).replace('.bin', '.idx'))

def stitch_shard_image_only(raw_img_path, output_path, tokenizer, encapsulator):
    """
    Handles Image-Only mode: Encapsulates raw vision tokens and wraps them in BOS/EOS.
    No text file is required.
    """
    img_ds = IndexedDataset(str(raw_img_path).replace('.bin', ''))
    builder = IndexedDatasetBuilder(str(output_path))
    
    for i in tqdm(range(len(img_ds)), desc=f"Stitching ImgOnly {Path(raw_img_path).stem}"):
        raw_img = img_ds[i]
        
        if len(raw_img) > 0:
            # Encapsulate: <img_start> + Dims + ... + <img_end>
            img_block = encapsulator.encapsulate(raw_img)
            # Wrap: [BOS] + Image + [EOS]
            combined = [tokenizer.bos_token_id] + img_block + [tokenizer.eos_token_id]
        else:
            # Empty image: Just BOS/EOS
            combined = [tokenizer.bos_token_id, tokenizer.eos_token_id]

        builder.add_document(np.array(combined, dtype=np.int32), [len(combined)])

    builder.finalize(str(output_path).replace('.bin', '.idx'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Folder containing raw data shards")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save stitched output")
    parser.add_argument("--new_tokenizer_path", type=str, required=True, help="Path to the target tokenizer")
    parser.add_argument("--mode", type=str, required=True, choices=["sft", "image2text", "image_only"])
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.new_tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer_path, trust_remote_code=True)
    encapsulator = LateBindingEncapsulator(tokenizer)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- SFT MODE ---
    if args.mode == "sft":
        raw_img_dir = Path(args.raw_data_dir) / "raw_images"
        raw_text_dir = Path(args.raw_data_dir) / "raw_texts"
        
        bin_files = sorted(list(raw_img_dir.glob("*.bin")))
        if not bin_files:
            print(f"No bin files found in {raw_img_dir}")
            return

        for bin_file in bin_files:
            shard_name = bin_file.stem
            text_file = raw_text_dir / f"{shard_name}.jsonl"
            output_file = Path(args.output_dir) / f"{shard_name}.bin"
            
            if text_file.exists():
                stitch_shard_sft(bin_file, text_file, output_file, tokenizer, encapsulator)
            else:
                print(f"Warning: No matching text file for {shard_name}")
                
    # --- IMAGE2TEXT MODE ---
    elif args.mode == "image2text":
        img_dir = Path(args.raw_data_dir) / "image"
        text_dir = Path(args.raw_data_dir) / "text"
        bin_files = sorted(list(img_dir.glob("*.bin")))
        
        for bin_file in bin_files:
            shard_name = bin_file.stem
            text_file = text_dir / f"{shard_name}.bin"
            output_file = Path(args.output_dir) / f"{shard_name}.bin"
            
            if text_file.exists():
                stitch_shard_image2text(text_file, bin_file, output_file, tokenizer, encapsulator)
            else:
                print(f"Warning: No matching text file for {shard_name}")

    # --- IMAGE ONLY MODE ---
    elif args.mode == "image_only":
        # Usually stored in 'raw_images' for consistency with SFT generation, 
        # or just 'image' if generated simply. We check 'raw_images' first.
        raw_img_dir = Path(args.raw_data_dir) / "raw_images"
        if not raw_img_dir.exists():
             raw_img_dir = Path(args.raw_data_dir) / "image" # Fallback
        
        bin_files = sorted(list(raw_img_dir.glob("*.bin")))
        
        if not bin_files:
            print(f"No bin files found in {raw_img_dir}")
            return
            
        for bin_file in bin_files:
            shard_name = bin_file.stem
            output_file = Path(args.output_dir) / f"{shard_name}.bin"
            stitch_shard_image_only(bin_file, output_file, tokenizer, encapsulator)

if __name__ == "__main__":
    main()