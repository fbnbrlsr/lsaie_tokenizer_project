from pathlib import Path
import re
import logging
import time
from typing import Dict, Any

from indexed_dataset_megatron import IndexedDataset, IndexedDatasetBuilder


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
    multimodal_dir.mkdir(exist_ok=True)
    
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
            image_dataset = IndexedDataset(image_idx_path)
            dtype = image_dataset.dtype
            text_dataset = IndexedDataset(text_idx_path) if text_idx_path else None
        except Exception as e:
            logger.error(f"Failed to load datasets for {prefix}: {e}. Skipping.")
            continue

        num_documents = len(image_dataset)
        total_samples += num_documents

        # Initialize multimodal builder
        multimodal_builder = IndexedDatasetBuilder(
            str(multimodal_dir / f"{prefix}.bin"),
            dtype=dtype
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

        logger.info(f"Finished {prefix}. Total tokens in new shard: {multimodal_builder.total_tokens:,}")

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

    stitch_shards(output_dir='./output/sft_data', mode='sft')