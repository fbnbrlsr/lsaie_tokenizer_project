import argparse
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from vision_tokenization.vokenizers.emu.image_text_pair import EMUImageTextPairTokenizer
from vision_tokenization.pipelines.indexed_dataset_megatron import IndexedDatasetBuilder
from vision_tokenization.pipelines.dataset import IndexedDataset

# --- SYSTEM PATH SETUP ---
# We need to find the 'Tokenizer' folder to load the Vision Decoder
# This attempts to find it relative to the script or current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
possible_paths = [
    current_dir,
    os.path.join(current_dir, 'Tokenizer'),
    os.path.dirname(current_dir), # Parent
    os.getcwd()
]
sys.path.extend(possible_paths)

try:
    from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer
    print("Successfully imported Emu3VisionTokenizer")
except ImportError:
    print("\nCRITICAL ERROR: Could not import 'Emu3VisionTokenizer'.")
    print("Please run this script from the root of your project (where the 'Tokenizer' folder is).")
    print(f"Searched paths: {possible_paths}")
    sys.exit(1)

def reconstruct_image(tokens, tokenizer, vision_decoder, output_path):
    """
    Extracts image tokens, un-shifts them, and decodes them to a PNG.
    """
    # 1. Special Token IDs
    img_start = tokenizer.convert_tokens_to_ids("<|img_start|>")
    img_end = tokenizer.convert_tokens_to_ids("<|img_end|>")
    img_tok_start = tokenizer.convert_tokens_to_ids("<|img_token_start|>")
    eol = tokenizer.convert_tokens_to_ids("<|img_end_of_row|>")
    eof = tokenizer.convert_tokens_to_ids("<|img_end_of_frame|>")
    vision_offset = tokenizer.convert_tokens_to_ids("<|visual token 000000|>")

    # 2. Extract Block
    start_indices = np.where(tokens == img_start)[0]
    end_indices = np.where(tokens == img_end)[0]
    
    if len(start_indices) == 0:
        print(" -> No image found in this sample.")
        return

    s_idx = start_indices[0]
    e_idx = end_indices[0]
    block = tokens[s_idx : e_idx+1]
    
    # 3. Parse Dimensions
    tok_start_loc = np.where(block == img_tok_start)[0]
    if len(tok_start_loc) == 0:
        print(" -> Malformed block (no img_token_start)")
        return
    ts_idx = tok_start_loc[0]
    
    dim_tokens = block[1:ts_idx]
    dim_str = tokenizer.decode(dim_tokens).strip()
    try:
        h, w = map(int, dim_str.split('*'))
        print(f" -> Found Image Structure: {h}x{w} tokens")
    except:
        print(f" -> Could not parse dims: '{dim_str}'")
        return

    # 4. Extract Codebook Indices (Un-shift)
    # The tokens are between <img_token_start> and <img_end>
    vision_segment = block[ts_idx+1 : -1] 
    
    # Remove EOL and EOF tokens
    mask = (vision_segment != eol) & (vision_segment != eof)
    clean_tokens = vision_segment[mask]
    
    # Check count
    expected_count = h * w
    if len(clean_tokens) != expected_count:
        print(f" -> WARNING: Token count mismatch! Expected {expected_count}, got {len(clean_tokens)}.")
        print(f" -> This usually means EOL removal failed or the block is truncated.")
        # Attempt to proceed anyway if possible
        if len(clean_tokens) > expected_count:
            clean_tokens = clean_tokens[:expected_count]
        else:
             print(" -> Cannot decode: Not enough tokens.")
             return

    # Un-shift (Llama ID -> Raw Codebook ID)
    codebook_indices = clean_tokens - vision_offset
    
    # 5. Decode
    try:
        # Reshape to [1, H, W]
        indices_tensor = torch.tensor(codebook_indices, dtype=torch.long).reshape(1, h, w).cuda()
        
        with torch.no_grad():
            # Emu3 Decoder call
            image = vision_decoder.decode(indices_tensor)
            
        # Post-process ([-1, 1] -> [0, 255])
        image = (image.clamp(-1, 1) + 1.0) / 2.0 * 255.0
        image = image.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
        
        # Save
        pil_img = Image.fromarray(image)
        pil_img.save(output_path)
        print(f" -> SAVED IMAGE: {output_path}")
        
    except Exception as e:
        print(f" -> Decoding Failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_prefix", type=str, required=True, help="Path to stitched .bin file (no extension)")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to text tokenizer")
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=3, help="How many samples to verify")
    args = parser.parse_args()

    if args.mode == "sft":
        # 1. Load Text Tokenizer
        print(f"Loading Text Tokenizer: {args.tokenizer_path}")
        text_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        
        # 2. Load Vision Decoder
        print("Loading Vision Decoder (Emu3)...")
        # Using standard resolution params for decoding
        vision_decoder = Emu3VisionTokenizer(
            device="cuda",
            min_pixels=256*256, 
            max_pixels=1024*1024
        )
        
        # 3. Load Dataset
        print(f"Loading Dataset: {args.data_prefix}")
        ds = IndexedDataset(args.data_prefix)
        
        # 4. Process Samples
        for i in range(min(len(ds), args.num_samples)):
            print(f"\n{'='*20} Sample {i} {'='*20}")
            tokens = ds[i]
            
            # A. Print Text
            # Decode everything to find text parts
            full_text = text_tokenizer.decode(tokens, skip_special_tokens=False)
            # Split by image start/end for cleaner reading if possible, else just print full
            print("--- Text Content ---")
            # Truncate if massive
            if len(full_text) > 500:
                print(full_text[:250] + "\n...[IMAGE BLOCK]...\n" + full_text[-250:])
            else:
                print(full_text)
                
            # B. Decode Image
            output_file = f"/users/ohatipoglu/scratch/lsaie_tokenizer_project/verified_sample_{i}.png"
            reconstruct_image(tokens, text_tokenizer, vision_decoder, output_file)
    
    elif args.mode == "image2text":
        
        # Load dataset
        print(f"Loading Dataset: {args.data_prefix}")
        ds = IndexedDataset(args.data_prefix)

        # Load tokenizer
        print(f"Loading tokenizer: {args.tokenizer_path}")
        text_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        vision_decoder = Emu3VisionTokenizer(
            device="cuda",
            min_pixels=256*256, 
            max_pixels=1024*1024
        )
        
        for i in range(len(ds)):
            
            tokens = ds[i]
            decoded_tokens = text_tokenizer.decode(tokens)
            
            print(f"--- sample {i} ---")
            print("Number of tokens:", len(tokens))
            print("Decoded text part:", decoded_tokens.split("</s>")[1])
            
            output_file = f"/users/fbrulisauer/scratch/ApertusProject/lsaie_tokenizer_project/verified_sample_{i}.png"
            reconstruct_image(tokens, text_tokenizer, vision_decoder, output_file)
    

if __name__ == "__main__":
    main()
    
"""
python verify_stitching.py \
    --data_prefix /users/fbrulisauer/scratch/ApertusProject/lsaie_tokenizer_project/my_tokenized_data_output/explanation_image2text/multimodal/rank_0_shard_0_2 \
    --tokenizer_path /users/fbrulisauer/scratch/ApertusProject/lsaie_tokenizer_project/my_omni_tokenizer \
    --num_samples 3 \
    --mode image2text
"""