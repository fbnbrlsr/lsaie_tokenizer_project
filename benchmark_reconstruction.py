import time
import argparse
import logging
import json
import numpy as np
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict
from vision_tokenization.pipelines.dataset import IndexedDataset
from vision_tokenization.pipelines.indexed_dataset_megatron import IndexedDatasetBuilder, DType


# Ensure local imports work
import sys
import os
sys.path.append(os.getcwd())


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("ReconstructionBench")

def benchmark_stitching(data_dir: str, output_dir: str, mode: str, max_shards: int = None):
    """
    Measures the reconstruction (stitching) complexity.
    """
    input_path = Path(data_dir)
    mm_dir = Path(output_dir) / "multimodal"
    mm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Auto-Detect Directory Structure
    if (input_path / "image").exists():
        # Separated structure
        image_dir = input_path / "image"
        text_dir = input_path / "text"
        meta_dir = input_path / "metadata"
        logger.info("📂 Detected SEPARATED structure (text/ & image/ folders)")
    else:
        # Flat structure (assume these are the 'image' shards for testing purposes)
        # Note: If these are already multimodal, we are essentially re-stitching them,
        # which is still a valid test of IO/Memory throughput.
        image_dir = input_path
        text_dir = input_path 
        meta_dir = input_path
        logger.info("📂 Detected FLAT structure (using root folder)")

    # 2. Identify Shards
    image_files = sorted(list(image_dir.glob("*.idx")))
    
    # --- BUG FIX START ---
    if not image_files:
        logger.error(f"No .idx files found in {image_dir}")
        return 0, 0, 0  # <--- Fixed: Returns 3 values now
    # --- BUG FIX END ---

    if max_shards:
        image_files = image_files[:max_shards]
    
    logger.info(f"🚀 Benchmarking Reconstruction on {len(image_files)} shards...")
    
    total_tokens = 0
    total_docs = 0
    start_time = time.time()
    
    # 3. Process Shards
    for img_idx_path in image_files:
        prefix = img_idx_path.stem 
        
        # Paths
        img_ds_path = str(img_idx_path.with_suffix(''))
        
        # Try to find matching text shard
        txt_idx_path = text_dir / f"{prefix}.idx"
        txt_ds_path = str(text_dir / prefix) if txt_idx_path.exists() else None
        
        meta_path = meta_dir / f"{prefix}_metadata.json"
        
        # Load Data
        try:
            img_ds = IndexedDataset(img_ds_path)
            txt_ds = IndexedDataset(txt_ds_path) if txt_ds_path else None
        except Exception as e:
            logger.warning(f"Skipping shard {prefix} due to load error: {e}")
            continue
        
        # Load Metadata
        metadata = []
        if mode == "sft" and meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            except: pass

        # Output Builder
        out_bin = mm_dir / f"{prefix}.bin"
        builder = IndexedDatasetBuilder(str(out_bin), dtype=np.int32)

        # 4. The Reconstruction Loop
        for i in range(len(img_ds)):
            image_tokens = list(img_ds.get(i))
            combined = []
            
            if mode == "sft" and txt_ds and i < len(metadata):
                # SFT Reconstruction
                text_tokens = list(txt_ds.get(i))
                meta = metadata[i]
                p = meta.get("image_position", 0)
                
                if not text_tokens:
                     combined = image_tokens
                else:
                    combined = text_tokens[:p] + image_tokens + text_tokens[p:]
                    
            elif mode == "image2text" and txt_ds:
                text_tokens = list(txt_ds.get(i))
                combined = image_tokens + text_tokens
                
            else:
                # Fallback / Image Only / Already Unified
                combined = image_tokens
            
            builder.add_document(combined, [len(combined)])
            total_tokens += len(combined)
            total_docs += 1

        builder.finalize(str(mm_dir / f"{prefix}.idx"))

    duration = time.time() - start_time
    return duration, total_tokens, total_docs

def main():
    parser = argparse.ArgumentParser(description="Measure Reconstruction Complexity")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to token files")
    parser.add_argument("--mode", type=str, default="sft", choices=["sft", "image_only", "image2text"])
    parser.add_argument("--num-shards", type=int, default=10, help="Number of shards to test")
    parser.add_argument("--cleanup", action="store_true", help="Delete output after test")
    
    args = parser.parse_args()
    
    work_dir = tempfile.mkdtemp(prefix="bench_reconstruct_")
    
    try:
        duration, tokens, docs = benchmark_stitching(
            args.data_dir, 
            work_dir, 
            args.mode, 
            args.num_shards
        )
        
        if duration > 0:
            tps = tokens / duration
            dps = docs / duration
            
            print("\n" + "="*60)
            print(f"RECONSTRUCTION COMPLEXITY REPORT")
            print("="*60)
            print(f"Mode:              {args.mode.upper()}")
            print(f"Processed:         {docs:,} documents")
            print(f"Total Tokens:      {tokens:,}")
            print(f"Time Elapsed:      {duration:.4f} s")
            print("-" * 60)
            print(f"Stitching Speed: {tps:,.0f} tokens/sec")
            print(f"Document Rate:   {dps:,.0f} docs/sec")
            print("-" * 60)
            
            trillion = 1_000_000_000_000
            hours_for_trillion = (trillion / tps) / 3600
            print(f"   Extrapolation for 1 Trillion Tokens:")
            print(f"   Time required:   ~{hours_for_trillion:.2f} hours (single process)")
            print(f"   With 64 CPUs:    ~{hours_for_trillion/64:.2f} hours")
            print("="*60)
        else:
            print("No data processed (check inputs).")

    finally:
        if args.cleanup:
            shutil.rmtree(work_dir)
            print(f"Cleaned up {work_dir}")
        else:
            print(f"Output files in {work_dir}")

if __name__ == "__main__":
    main()