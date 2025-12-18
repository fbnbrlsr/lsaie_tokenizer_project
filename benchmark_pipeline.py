import os
import sys
import time
import json
import logging
import argparse
import tempfile
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from vision_tokenization.pipelines.dataset import IndexedDataset
from vision_tokenization.pipelines.indexed_dataset_megatron import IndexedDatasetBuilder, DType

# Ensure local imports work
sys.path.append(os.getcwd())

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkMetrics:
    def __init__(self, name):
        self.name = name
        self.start_time = 0
        self.end_time = 0
        self.total_tokens = 0
        self.total_bytes = 0

    def start(self):
        self.start_time = time.time()

    def stop(self, tokens, bytes_processed=0):
        self.end_time = time.time()
        self.total_tokens = tokens
        self.total_bytes = bytes_processed

    def report(self):
        duration = self.end_time - self.start_time
        tps = self.total_tokens / duration if duration > 0 else 0
        mbps = (self.total_bytes / 1024 / 1024) / duration if duration > 0 else 0
        
        print(f"\n--- Results: {self.name} ---")
        print(f"Time Taken:      {duration:.2f} seconds")
        print(f"Total Tokens:    {self.total_tokens:,}")
        print(f"Throughput:      {tps:,.2f} tokens/sec")
        if self.total_bytes > 0:
            print(f"Data Throughput: {mbps:.2f} MB/s")
        print("----------------------------")
        return duration, tps

def get_tokenizer_info(data_dir: str):
    """
    Extracts vision offset from dataset_info.json -> tokenizer path.
    Needed to separate Vision vs Text tokens.
    """
    try:
        info_path = Path(data_dir) / "dataset_info.json"
        
        # Default fallback
        offset, vocab_size = 32000, 131072 

        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            tokenizer_path = info.get("tokenizer", {}).get("path")
            if tokenizer_path:
                # Try to find added_tokens.json or config
                added_tokens_path = Path(tokenizer_path) / "added_tokens.json"
                if added_tokens_path.exists():
                    with open(added_tokens_path, 'r') as f:
                        added = json.load(f)
                    # Heuristic: Find first token looking like <|visual token ...|>
                    for k, v in added.items():
                        if "visual token" in k and "00000" in k:
                            logger.info(f"Found visual token offset from tokenizer config: {v}")
                            return v, 131072
        
        logger.warning(f"Could not verify offset from config. Using default offset: {offset}")
        return offset, vocab_size
    except Exception as e:
        logger.warning(f"Failed to load tokenizer info: {e}. Using defaults.")
        return 32000, 131072

def benchmark_separation(input_dir: str, output_dir: str, vision_offset: int, max_shards: int = None, input_is_flat: bool = False):
    """
    Benchmark splitting Multimodal -> Text + Image
    """
    metrics = BenchmarkMetrics("Separation (Split)")
    
    out_text_dir = Path(output_dir) / "text"
    out_image_dir = Path(output_dir) / "image"
    out_text_dir.mkdir(parents=True, exist_ok=True)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    # Detect input files
    if input_is_flat:
        # User provided flat dir (rank_000.idx)
        mm_files = sorted(list(Path(input_dir).glob("*.idx")))
    else:
        # Standard pipeline structure (multimodal/rank_000.idx)
        mm_files = sorted(list((Path(input_dir) / "multimodal").glob("*.idx")))

    if not mm_files:
        logger.error(f"No .idx files found in {input_dir} to separate.")
        return 0, 0
    
    if max_shards:
        mm_files = mm_files[:max_shards]

    total_tokens_processed = 0
    total_bytes_processed = 0

    logger.info(f"Separating {len(mm_files)} shards using vision offset {vision_offset}...")
    metrics.start()

    for mm_idx_path in mm_files:
        prefix = mm_idx_path.stem # e.g., rank_000
        ds = IndexedDataset(str(mm_idx_path.with_suffix('')))

        text_builder = IndexedDatasetBuilder(str(out_text_dir / f"{prefix}.bin"), dtype=np.int32)
        image_builder = IndexedDatasetBuilder(str(out_image_dir / f"{prefix}.bin"), dtype=np.int32)

        for i in range(len(ds)):
            tokens = ds.get(i)
            
            # Separation Logic:
            # Assume tokens >= vision_offset are image tokens
            is_image = tokens >= vision_offset
            
            img_tokens = tokens[is_image]
            txt_tokens = tokens[~is_image]

            if len(img_tokens) > 0:
                image_builder.add_document(img_tokens, [len(img_tokens)])
            
            if len(txt_tokens) > 0:
                text_builder.add_document(txt_tokens, [len(txt_tokens)])
            
            total_tokens_processed += len(tokens)
            total_bytes_processed += len(tokens) * 4

        text_builder.finalize(str(out_text_dir / f"{prefix}.idx"))
        image_builder.finalize(str(out_image_dir / f"{prefix}.idx"))

    metrics.stop(total_tokens_processed, total_bytes_processed)
    return metrics.report()

def benchmark_stitching(data_dir: str, output_dir: str, mode: str, max_shards: int = None):
    """
    Benchmark merging Text + Image -> Multimodal
    """
    metrics = BenchmarkMetrics("Stitching (Merge)")
    
    text_dir = Path(data_dir) / "text"
    image_dir = Path(data_dir) / "image"
    mm_dir = Path(output_dir) / "multimodal"
    mm_dir.mkdir(parents=True, exist_ok=True)

    # Gather shards based on image files
    image_files = sorted(list(image_dir.glob("*.idx")))
    
    if not image_files:
        logger.error(f"No image files found in {image_dir} for stitching.")
        return 0, 0

    if max_shards:
        image_files = image_files[:max_shards]
    
    total_tokens_processed = 0
    total_bytes_processed = 0

    logger.info(f"Stitching {len(image_files)} shards...")
    metrics.start()
    
    for img_idx_path in image_files:
        prefix = img_idx_path.stem
        text_idx_path = text_dir / f"{prefix}.idx"
        
        # Load datasets
        img_ds = IndexedDataset(str(img_idx_path.with_suffix('')))
        txt_ds = None
        if text_idx_path.exists():
            txt_ds = IndexedDataset(str(text_idx_path.with_suffix('')))

        # Create Output Builder
        out_path = mm_dir / f"{prefix}.bin"
        builder = IndexedDatasetBuilder(str(out_path), dtype=np.int32)

        # Process Documents
        # Note: We rely on image_ds length. 
        # If text_ds has fewer docs (shouldn't happen in aligned shards), handle gracefully.
        for i in range(len(img_ds)):
            img_tokens = list(img_ds.get(i))
            combined = []

            # Simulate logic: Image + Text (or Text + Image based on mode)
            # For pure IO throughput, concatenation order doesn't affect speed, just logic.
            if txt_ds and i < len(txt_ds):
                txt_tokens = list(txt_ds.get(i))
                # Simple Stitching: Image then Text (standard Emu pattern)
                combined = img_tokens + txt_tokens
            else:
                combined = img_tokens
            
            # Write
            builder.add_document(combined, [len(combined)])
            total_tokens_processed += len(combined)
            total_bytes_processed += len(combined) * 4 # int32

        builder.finalize(str(mm_dir / f"{prefix}.idx"))

    metrics.stop(total_tokens_processed, total_bytes_processed)
    return metrics.report()

def main():
    parser = argparse.ArgumentParser(description="Benchmark Vision Tokenization Stitch/Separate")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing token files")
    parser.add_argument("--mode", type=str, choices=["image_only", "image2text", "text2image", "sft"], default="sft")
    parser.add_argument("--num-shards", type=int, default=5, help="Limit number of shards to process")
    parser.add_argument("--cleanup", action="store_true", help="Delete temporary output files after run")
    
    args = parser.parse_args()

    # Create temporary workspace
    work_dir = tempfile.mkdtemp(prefix="bench_viz_tok_")
    logger.info(f"Working directory: {work_dir}")
    
    data_path = Path(args.data_dir)
    vision_offset, _ = get_tokenizer_info(args.data_dir)

    # --- DETECT PIPELINE FLOW ---
    # Check if data_dir has 'image' folder (Separated) or flat files (Unified)
    has_image_subdir = (data_path / "image").exists()
    
    # Vars for summary
    stitch_time = 0
    stitch_tps = 0
    sep_time = 0
    sep_tps = 0

    try:
        if has_image_subdir:
            # Case A: Input is SEPARATED. Flow: Stitch ONLY.
            logger.info("Detected 'image' subdirectory. Assuming SEPARATED input.")
            logger.info("Running STITCHING benchmark only (as requested).")
            
            print("\n" + "="*50)
            print("BENCHMARK 1: STITCHING (Source -> Unified)")
            print("="*50)
            stitch_time, stitch_tps = benchmark_stitching(
                data_dir=args.data_dir, 
                output_dir=work_dir, 
                mode=args.mode,
                max_shards=args.num_shards
            )

        else:
            # Case B: Input is UNIFIED/FLAT. Flow: Separate -> Stitch
            logger.info("No 'image' subdirectory found. Assuming UNIFIED/FLAT input.")
            
            print("\n" + "="*50)
            print("BENCHMARK 1: SEPARATION (Source -> Separate)")
            print("="*50)
            
            sep_time, sep_tps = benchmark_separation(
                input_dir=args.data_dir,
                output_dir=work_dir,
                vision_offset=vision_offset,
                max_shards=args.num_shards,
                input_is_flat=True
            )

            print("\n" + "="*50)
            print("BENCHMARK 2: STITCHING (Separate -> Unified)")
            print("="*50)
            # Use the result of benchmark 1 (work_dir/text and work_dir/image)
            stitch_time, stitch_tps = benchmark_stitching(
                data_dir=work_dir,
                output_dir=Path(work_dir) / "stitched_output",
                mode=args.mode,
                max_shards=args.num_shards
            )

        # --- Combined Summary ---
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"{'Operation':<20} | {'Time (s)':<10} | {'Tokens/sec':<15}")
        print("-" * 50)
        
        if stitch_time > 0:
            print(f"{'Stitching':<20} | {stitch_time:<10.2f} | {stitch_tps:<15,.0f}")
        
        if sep_time > 0:
            print(f"{'Separation':<20} | {sep_time:<10.2f} | {sep_tps:<15,.0f}")
            
        print("-" * 50)
        print(f"Total Pipeline Time: {stitch_time + sep_time:.2f} s")

    finally:
        if args.cleanup:
            logger.info(f"Cleaning up {work_dir}...")
            shutil.rmtree(work_dir)
        else:
            logger.info(f"Results left in {work_dir}")

if __name__ == "__main__":
    main()