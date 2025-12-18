import argparse
import os
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
from vision_tokenization.pipelines.indexed_dataset_megatron import IndexedDatasetBuilder
from vision_tokenization.pipelines.dataset import IndexedDataset

def recycle_shard(input_prefix, output_dir, tokenizer, shard_name):
    """
    Reads a "Baked" SFT shard, extracts raw components, and saves them.
    - Images -> .bin/.idx (Raw Codebook Indices 0-32k)
    - Text -> .jsonl (Raw Strings)
    """
    # Load Input
    ds = IndexedDataset(input_prefix)
    
    # Prepare Outputs
    image_dir = os.path.join(output_dir, 'raw_images')
    text_dir = os.path.join(output_dir, 'raw_texts')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    
    # Image Builder (for raw indices)
    image_builder = IndexedDatasetBuilder(os.path.join(image_dir, f"{shard_name}.bin"))
    
    # Text File (JSONL for raw strings)
    text_file_path = os.path.join(text_dir, f"{shard_name}.jsonl")
    
    # Identify Special Token IDs for Parsing (from the OLD tokenizer)
    try:
        img_start_id = tokenizer.convert_tokens_to_ids("<|img_start|>")
        img_end_id = tokenizer.convert_tokens_to_ids("<|img_end|>")
        img_token_start_id = tokenizer.convert_tokens_to_ids("<|img_token_start|>")
        eol_id = tokenizer.convert_tokens_to_ids("<|img_end_of_row|>")
        eof_id = tokenizer.convert_tokens_to_ids("<|img_end_of_frame|>")
        vision_offset = tokenizer.convert_tokens_to_ids("<|visual token 000000|>")
    except:
        print("Error: The provided tokenizer is missing Emu3 special tokens.")
        return

    print(f"Processing {shard_name} | Vision Offset: {vision_offset}")

    with open(text_file_path, 'w') as text_f:
        for i in tqdm(range(len(ds)), desc=f"Recycling {shard_name}"):
            tokens = ds[i]
            
            # --- 1. Locate Image Block ---
            start_indices = np.where(tokens == img_start_id)[0]
            end_indices = np.where(tokens == img_end_id)[0]
            
            # Data Containers
            raw_vision = np.array([], dtype=np.int32)
            text_before_str = ""
            text_after_str = ""
            
            if len(start_indices) > 0 and len(end_indices) > 0:
                # Assuming 1 image per sample
                start_idx = start_indices[0]
                end_idx = end_indices[0]
                
                # --- 2. Extract & Decode Text ---
                # Decode separates the text from the tokens, making it tokenizer-agnostic
                text_before_tokens = tokens[:start_idx]
                text_after_tokens = tokens[end_idx + 1:]
                
                text_before_str = tokenizer.decode(text_before_tokens, skip_special_tokens=True)
                text_after_str = tokenizer.decode(text_after_tokens, skip_special_tokens=True)
                
                # --- 3. Extract & Clean Image ---
                full_block = tokens[start_idx:end_idx+1]
                
                # Find visual tokens (after <img_token_start>)
                token_start_indices = np.where(full_block == img_token_start_id)[0]
                
                if len(token_start_indices) > 0:
                    vis_start = token_start_indices[0] + 1
                    # Handle EOF if present
                    eof_indices = np.where(full_block == eof_id)[0]
                    vis_end = eof_indices[0] if len(eof_indices) > 0 else len(full_block) - 1
                    
                    vision_segment = full_block[vis_start:vis_end]
                    
                    # Remove EOLs and Un-shift
                    mask = (vision_segment != eol_id)
                    raw_vision = vision_segment[mask] - vision_offset
            else:
                # Text-only sample
                text_before_str = tokenizer.decode(tokens, skip_special_tokens=True)

            # --- 4. Save ---
            # Save Raw Image Indices
            # [FIX] Use add_document instead of add_item to create proper index entries
            image_builder.add_document(raw_vision, [len(raw_vision)])
            
            # Save Raw Text Strings
            meta_obj = {
                "text_before": text_before_str,
                "text_after": text_after_str,
                "has_image": len(raw_vision) > 0
            }
            text_f.write(json.dumps(meta_obj) + "\n")

    image_builder.finalize(os.path.join(image_dir, f"{shard_name}.idx"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with existing rank_*.bin")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save RAW data")
    parser.add_argument("--old_tokenizer_path", type=str, required=True, help="Path to the tokenizer used originally")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.old_tokenizer_path, trust_remote_code=True)
    input_path = Path(args.input_dir)
    
    # Find all .idx files
    files = sorted(list(input_path.glob("rank_*.idx")))
    if not files:
        print("No .idx files found.")
        return
        
    for idx_file in files:
        recycle_shard(str(input_path / idx_file.stem), args.output_dir, tokenizer, idx_file.stem)

if __name__ == "__main__":
    main()