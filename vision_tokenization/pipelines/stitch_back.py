from pathlib import Path
import re
import logging
import time
from typing import Dict, Any
import os
import numpy as np
import json

from indexed_dataset_megatron import IndexedDatasetBuilder
from dataset import IndexedDataset



def stich_together_tokens(work_dir: str, mode: str, BATCH_SIZE: int = 1000):
    
    os.makedirs(work_dir + '/multimodal', exist_ok=True)
    
    ### Tokenizer information ###
    with open(work_dir + "/" 'dataset_info.json', 'r') as f:
        dataset_info = json.load(f)
    
    tokenizer_path = dataset_info["tokenizer"]["path"]
    with open(tokenizer_path + "/" 'tokenizer_config.json', 'r') as f:
        tokenizer_config = json.load(f)
        
    with open(tokenizer_path + "/" 'added_tokens.json', 'r') as f:
        added_tokens = json.load(f)
    VISION_TOKEN_OFFSET = added_tokens["<|visual token 00000|>"]
    
    print("====== TOKENIZER CONFIG ======")
    print(" - vocab_size:", tokenizer_config["vocab_size"])
    print(" - base_vocab_size:", tokenizer_config["base_vocab_size"])
    print(" - added_tokens_count:", tokenizer_config["added_tokens_count"])
    print(" - vision_tokenizer.codebook_size:", tokenizer_config["vision_tokenizer"]["codebook_size"])
    print(" - vision_token_offset:", VISION_TOKEN_OFFSET)
    print("==============================")
    
    vocab_ranges = {
        "text_vocab": (0, tokenizer_config["base_vocab_size"]),
        "vision_vocab": (tokenizer_config["vocab_size"]-tokenizer_config["vision_tokenizer"]["codebook_size"], tokenizer_config["vocab_size"]),
        "special_vocab": (tokenizer_config["base_vocab_size"]+1, tokenizer_config["vocab_size"]-tokenizer_config["vision_tokenizer"]["codebook_size"]-1)
    }
    
    print("\nVOCAB RANGES:")
    for k, v in vocab_ranges.items():
        print(f"{k} -> {v[0]} - {v[1]}, size: {v[1] - v[0]}")
    
    
    ### Read token files ###
    
    subdirs = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
    print(f"\nFound folders {subdirs} in directory {work_dir}")
    
    image_mode = "image" in subdirs
    text_mode = "text" in subdirs
    
    # All unique file names
    file_names = set([
        f.split(".")[0]
        for subdir in subdirs if subdir in ["text", "image"]
        for f in os.listdir(work_dir+"/"+subdir) 
    ])
    print("File names:", file_names)
    
    # New multimodal dataset for combined tokens
    mm_dataset_count = 0
    mm_dataset = IndexedDatasetBuilder(
        str(work_dir + f"/multimodal/multimodal_dataset_{mm_dataset_count}.bin"),
        dtype=np.int32
    )
    
    # Debug info
    debug_info = {
        "text_min": 100000,
        "text_max": -1,
        "image_min": 100000,
        "image_max": -1
    }
    mm_dataset_sizes = []
    
    # Iterate over all files
    for name in file_names:
        
        # Read data
        text_shard = IndexedDataset(work_dir + "/text/" + name)
        image_shard = IndexedDataset(work_dir + "/image/" + name) 
        
        # Iterate over all samples in the current shard
        for i in range(len(image_shard)):
            
            # Get tokens
            text_tokens = text_shard[i]
            image_tokens = image_shard[i]
            
            # Update debug info
            debug_info["text_min"] = min(debug_info["text_min"], text_tokens.min())
            debug_info["text_max"] = max(debug_info["text_max"], text_tokens.max())
            debug_info["image_min"] = min(debug_info["image_min"], image_tokens.min())
            debug_info["image_max"] = max(debug_info["image_max"], image_tokens.max())
            
            # print(f"\n--- TEXT {i} ({name}) ---")
            # print(f" - tokens [{len(text_tokens)}]:", text_tokens)
            # print(f" - type: {type(text_tokens)}, shape: {text_tokens.shape}")
            # print(f" - min: {text_tokens.min()}, max: {text_tokens.max()}")
            
            # print(f"--- IMAGE {i} ({name}) ---")
            # print(f" - tokens [{len(image_tokens)}]:", image_tokens)
            # print(f" - type: {type(image_tokens)}, shape: {image_tokens.shape}")
            # print(f" - min: {image_tokens.min()}, max: {image_tokens.max()}")
            
            # TODO: handle special tokens
            if mode == "image2text":
                # First image tokens, then text tokens
                
                image_tokens_shifted = image_tokens + VISION_TOKEN_OFFSET
                
                combined_tokens = np.concatenate((image_tokens_shifted, text_tokens))
                
                
            elif mode == "text2image":
                # First text tokens, then image tokens
                raise NotImplementedError
            
            elif mode == "sft":
                raise NotImplementedError

            else:
                raise ValueError(f"Mode '{mode}' is not a valid option.")
            
            mm_dataset.add_item(combined_tokens)
            
            # Finalize combined dataset if batch size is reached
            if len(mm_dataset) == BATCH_SIZE:
                
                mm_dataset_sizes.append(len(mm_dataset))
                # Finalize
                mm_dataset.finalize(work_dir + f"/multimodal/multimodal_dataset_{mm_dataset_count}.idx")
                
                # New dataset
                mm_dataset_count += 1
                mm_dataset = IndexedDatasetBuilder(
                    str(work_dir + f"/multimodal/multimodal_dataset_{mm_dataset_count}.bin"),
                    dtype=np.int32
                )
                
            
    mm_dataset_sizes.append(len(mm_dataset))
    mm_dataset.finalize(work_dir + f"/multimodal/multimodal_dataset_{mm_dataset_count}.idx")
    mm_dataset_count += 1
    
    print("\n====== FINAL RESULTS ======")
    print("Number of multimodal datasets:", mm_dataset_count)
    print("Multimodal dataset sizes:", mm_dataset_sizes)
    print("Token IDs:")
    for k, v in debug_info.items():
        print(f"    {k} -> {v}")
    print("===========================")
    
    

def stitch_shards(output_dir: str, mode: str) -> Dict[str, Any]:
    """
    Merges separate text and image shards into a single multimodal shard.

    Args:
        output_dir: The base directory (e.g., './output/CoSyn_400k_chart_sft')
                    containing 'text/' and 'image/' subdirectories.
        mode: The tokenization mode (e.g., 'sft', 'image2text').

    Returns:
        A dictionary with stitching results and statistics.
    """
    # Setup paths and logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger("StitchUtility")
    
    output_path = Path(output_dir)
    image_dir = output_path / "image"
    text_dir = output_path / "text"
    multimodal_dir = output_path / "multimodal"
    multimodal_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all shard pairs
    image_idx_files = list(image_dir.glob('*.idx'))
    if not image_idx_files:
        logger.error(f"No image index files found in {image_dir}. Exiting.")
        return {"status": "error", "message": "No image index files found."}

    shard_files = {} # Key: filename prefix (e.g., rank_0_shard_1_100)

    pattern = re.compile(r'(rank_\d+_shard_\d+_\d+)\.idx')

    for image_idx in image_idx_files:
        match = pattern.search(image_idx.name)
        if match:
            prefix = match.group(1)
            # Check for corresponding text file (if not image_only)
            if mode != "image_only":
                text_idx = text_dir / f"{prefix}.idx"
                if text_idx.exists():
                    # Store as (text_idx_path, image_idx_path)
                    shard_files[prefix] = (str(text_idx), str(image_idx))
                else:
                    logger.warning(f"Found image shard {prefix} but no matching text shard. Skipping.")
            else:
                # For image_only, only the image path is needed
                shard_files[prefix] = (None, str(image_idx))

    if not shard_files:
        logger.error("No complete shard pairs found for stitching. Exiting.")
        return {"status": "error", "message": "No complete shard pairs found."}

    logger.info(f"Found {len(shard_files)} shard pairs to stitch.")
    start_time = time.time()
    total_samples = 0
    
    # Stitch each pair
    for i, (prefix, (text_idx_path, image_idx_path)) in enumerate(shard_files.items()):
        logger.info(f"Stitching {i+1}/{len(shard_files)}: {prefix}")
        
        # Load input datasets
        try:
            print("image_idx_path:", image_idx_path)
            print("text_idx_path:", text_idx_path)
            image_dataset = IndexedDataset(image_idx_path.split(".")[0])
            # dtype = image_dataset.dtype
            text_dataset = IndexedDataset(text_idx_path.split(".")[0]) if text_idx_path else None
        except Exception as e:
            logger.error(f"Failed to load datasets for {prefix}: {e}. Skipping.")
            continue

        num_documents = len(image_dataset)
        total_samples += num_documents

        # Initialize multimodal builder
        multimodal_builder = IndexedDatasetBuilder(
            str(multimodal_dir / f"{prefix}.bin")
        )

        for doc_idx in range(num_documents):
            image_tokens = image_dataset.get(doc_idx)
            text_tokens = text_dataset.get(doc_idx) if text_dataset else None

            # Start with image tokens (which include BOS/EOS/boundary markers)
            combined_tokens = [t.item() for t in image_tokens]
            
            # Append text tokens if they exist (and are not an empty padding document)
            if text_tokens is not None and len(text_tokens) > 0:
                combined_tokens.extend([t.item() for t in text_tokens])
            
            # Write the combined sequence
            multimodal_builder.add_document(combined_tokens, [len(combined_tokens)])

        # Finalize the combined builder
        multimodal_builder.finalize(str(multimodal_dir / f"{prefix}.idx"))

        # Clean up file handles
        del image_dataset
        if text_dataset:
            del text_dataset

        logger.info(f"Finished {prefix}. Total tokens in new shard: {len(combined_tokens):,}")

    elapsed_time = time.time() - start_time
    logger.info(f"Stitching completed for {len(shard_files)} shards in {elapsed_time:.1f}s. Total samples: {total_samples:,}")
    logger.info(f"Multimodal shards saved to {multimodal_dir.resolve()}")

    return {
        "status": "success",
        "total_shards_stitched": len(shard_files),
        "total_samples": total_samples,
        "processing_time": elapsed_time
    }

if __name__ == '__main__':
    # Example directory structure:
    # ./output/sft_data/
    # ├── text/
    # │   └── rank_0_shard_0_100.idx
    # └── image/
    #     └── rank_0_shard_0_100.idx

    results = stitch_shards(
        output_dir='/users/fbrulisauer/scratch/ApertusProject/lsaie_tokenizer_project/my_tokenized_data_output/explanation_image2text', 
        mode='image_only'
    )
    
    for k, v in results.items():
        print(f"{k} -> {v}")
    
    # stich_together_tokens(
    #     work_dir='/users/fbrulisauer/scratch/ApertusProject/lsaie_tokenizer_project/my_tokenized_data_output/explanation_image2text',
    #     mode = "image2text"
    # )