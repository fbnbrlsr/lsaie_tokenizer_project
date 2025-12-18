#!/usr/bin/env python3
"""
Unified Ray workers for distributed tokenization.
Handles all tokenization modes with a single flexible worker class.
"""

import time
from typing import Dict, Optional
import ray


@ray.remote
class ShardQueue:
    """Dynamic queue for distributing shards to workers."""

    def __init__(self, num_shards: int, initial_shards: Optional[list] = None):
        self.num_shards = num_shards
        if initial_shards is not None:
            # Resume mode: only process specified shards
            self.remaining_shards = list(initial_shards)
        else:
            # Normal mode: process all shards (0 to num_shards-1)
            self.remaining_shards = list(range(num_shards))
        self.current_index = 0  # Index to track position in remaining_shards
        self.in_progress = {}  # shard_id -> (worker_id, start_time)
        self.completed = []
        self.failed = []

    def get_next_shard(self, worker_id: int) -> Optional[int]:
        """Get next shard index for a worker (work-stealing)."""
        if self.current_index >= len(self.remaining_shards):
            return None

        shard_id = self.remaining_shards[self.current_index]
        self.current_index += 1
        self.in_progress[shard_id] = (worker_id, time.time())

        return shard_id

    def mark_completed(self, shard_id: int, stats: Dict):
        """Mark shard as completed."""
        if shard_id in self.in_progress:
            del self.in_progress[shard_id]
        self.completed.append((shard_id, stats))

    def mark_failed(self, shard_id: int, error: str):
        """Mark shard as failed."""
        if shard_id in self.in_progress:
            del self.in_progress[shard_id]
        self.failed.append((shard_id, error))

    def get_status(self) -> Dict:
        """Get current processing status."""
        return {
            'processed': self.next_shard,
            'total': self.num_shards,
            'in_progress': len(self.in_progress),
            'completed': len(self.completed),
            'failed': len(self.failed)
        }


from vision_tokenization.pipelines.base import BaseTokenizerWorker


@ray.remote(num_gpus=1)
class Worker(BaseTokenizerWorker):
    """
    HuggingFace dataset worker that extends BaseTokenizerWorker.
    Adds HF-specific data loading, work queue processing, and rank-based output.
    """

    def __init__(
        self,
        tokenizer_path: str,
        output_dir: str,
        worker_id: int,
        mode: str,
        min_pixels: int,
        max_pixels: int,
        image_field: str = "image",
        text_field: str = "text",
        min_image_pixels: Optional[int] = None,
        max_image_pixels: Optional[int] = None
    ):
        """
        Initialize HF worker with tokenizer and output configuration.

        Args:
            tokenizer_path: Path to tokenizer
            output_dir: Directory for output files
            worker_id: Unique worker identifier
            mode: Tokenization mode ('image_only', 'image_text_pair', or 'sft')
            min_pixels: Min pixels for tokenizer preprocessing
            max_pixels: Max pixels for tokenizer preprocessing
            image_field: Field name for images in dataset
            text_field: Field name for text in dataset
            min_image_pixels: Min pixels to filter images (optional)
            max_image_pixels: Max pixels to filter images (optional)
        """
        # Initialize base tokenizer with resolution filtering parameters
        super().__init__(
            tokenizer_path=tokenizer_path,
            worker_id=worker_id,
            mode=mode,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            image_field=image_field,
            text_field=text_field,
            min_image_pixels=min_image_pixels,
            max_image_pixels=max_image_pixels
        )

        # Store output directory for per-shard files
        self.output_dir = output_dir


    def process_shard(self, shard_id: int, dataset_info: Dict, num_shards: int) -> Dict:
        """
        Process a complete shard and save to separate text/image files.

        Args:
            shard_id: Index of the shard to process
            dataset_info: Dataset metadata
            num_shards: Total number of shards

        Returns:
            Processing statistics for this shard
        """
        self.logger.info(f"Processing shard {shard_id}/{num_shards}")
        start_time = time.time()

        # Load the shard using HuggingFace's efficient shard method
        from datasets import load_dataset
        dataset = load_dataset(
            dataset_info['name'],
            name=dataset_info.get('config'),
            split=dataset_info['split'],
            cache_dir=dataset_info.get('cache_dir')
        )

        # Get this specific shard
        shard = dataset.shard(num_shards=num_shards, index=shard_id)

        # Setup output paths with subdirectories
        from vision_tokenization.pipelines.indexed_dataset_megatron import DType, IndexedDatasetBuilder
        from pathlib import Path

        output_base = Path(self.output_dir)
        shard_filename = f"rank_{self.worker_id}_shard_{shard_id}_{num_shards}"

        # Create subdirectories for text and image
        text_dir = output_base / "text"
        image_dir = output_base / "image"
        metadata_dir = output_base / "metadata"

        # Create directories (only text if not image_only mode)
        if self.mode != "image_only":
            text_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        # Create metadata directory for SFT mode
        if self.mode == "sft":
            metadata_dir.mkdir(parents=True, exist_ok=True)

        # Create builders for both modalities
        dtype = DType.optimal_dtype(len(self.tokenizer.text_tokenizer))

        text_builder = None
        if self.mode != "image_only":
            text_builder = IndexedDatasetBuilder(
                str(text_dir / f"{shard_filename}.bin"),
                dtype=dtype
            )

        image_builder = IndexedDatasetBuilder(
            str(image_dir / f"{shard_filename}.bin"),
            dtype=dtype
        )

        # Process all samples in the shard
        stats = {
            'samples': 0,
            'tokens': 0,
            'image_tokens': 0,
            'text_tokens': 0,
            'errors': 0,
            'skipped': 0,
            'resolution_skipped': 0
        }

        # Collect per-sample metadata for SFT mode
        shard_metadata = []

        for sample in shard:
            # Extract data
            image, text = self._extract_data(sample)

            # Check sample status
            status = self.get_sample_status(image, text)

            if status == 'resolution_skip':
                stats['resolution_skipped'] += 1
                continue
            elif status == 'data_skip':
                stats['skipped'] += 1
                continue

            # Tokenize
            try:
                result = self.tokenize_sample(image, text)
                if result is None:
                    stats['errors'] += 1
                    continue

                # Extract components
                text_tokens = result.get("text")
                image_tokens = result.get("image")
                metadata = result.get("metadata", {})

                # Validate we have image tokens (always required)
                if image_tokens is None or len(image_tokens) == 0:
                    self.logger.warning("No image tokens generated, skipping sample")
                    stats['errors'] += 1
                    continue

                # Write text tokens (with empty document for alignment if needed)
                if text_builder is not None:
                    if text_tokens is not None and len(text_tokens) > 0:
                        text_builder.add_document(text_tokens, [len(text_tokens)])
                        stats['text_tokens'] += len(text_tokens)
                    else:
                        # Add empty document to maintain alignment with image
                        text_builder.add_document([], [0])

                # Write image tokens (always)
                image_builder.add_document(image_tokens, [len(image_tokens)])
                stats['image_tokens'] += len(image_tokens)

                # Update overall stats
                stats['samples'] += 1
                stats['tokens'] += (len(text_tokens) if text_tokens is not None else 0) + len(image_tokens)

                # Collect metadata for SFT mode
                if self.mode == "sft" and metadata:
                    shard_metadata.append({
                        "idx": stats['samples'] - 1,
                        "text_before_len": metadata.get("text_before_len", 0),
                        "text_after_len": metadata.get("text_after_len", 0),
                        "image_position": metadata.get("image_position", 0)
                    })

            except Exception as e:
                self.logger.warning(f"Failed to process sample: {e}")
                stats['errors'] += 1

        # Finalize both builders
        if text_builder is not None:
            text_builder.finalize(str(text_dir / f"{shard_filename}.idx"))
        image_builder.finalize(str(image_dir / f"{shard_filename}.idx"))

        # Write metadata file for SFT mode
        if self.mode == "sft" and shard_metadata:
            import json
            metadata_path = metadata_dir / f"{shard_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(shard_metadata, f)

        elapsed = time.time() - start_time
        self.logger.info(
            f"Completed shard {shard_id}: {stats['samples']} samples, "
            f"{stats['tokens']} tokens ({stats['text_tokens']} text, {stats['image_tokens']} image) "
            f"in {elapsed:.1f}s"
        )

        return {
            'shard_id': shard_id,
            'time': elapsed,
            **stats
        }

    def run_shards(self, shard_queue, dataset_info, num_shards, progress_actor=None) -> Dict:
        """
        Main worker loop for shard-based processing.

        Args:
            shard_queue: Ray remote shard queue for distribution
            dataset_info: Dataset metadata
            num_shards: Total number of shards
            progress_actor: Optional progress tracking actor

        Returns:
            Final worker statistics
        """
        self.logger.info("Starting shard processing loop")

        while True:
            # Get next shard from queue
            shard_id = ray.get(shard_queue.get_next_shard.remote(self.worker_id))

            if shard_id is None:
                self.logger.info("No more shards, finishing")
                break

            # Process the shard
            try:
                result = self.process_shard(shard_id, dataset_info, num_shards)
                ray.get(shard_queue.mark_completed.remote(shard_id, result))

                # Report progress if actor provided
                if progress_actor:
                    progress_actor.update.remote(result['samples'])

                # Update global statistics
                self.update_stats(
                    samples=result['samples'],
                    tokens=result['tokens'],
                    errors=result['errors'],
                    skipped=result.get('skipped', 0),
                    resolution_skipped=result.get('resolution_skipped', 0),
                    image_tokens=result.get('image_tokens', 0),
                    text_tokens=result.get('text_tokens', 0)
                )

            except Exception as e:
                self.logger.error(f"Failed to process shard {shard_id}: {e}")
                ray.get(shard_queue.mark_failed.remote(shard_id, str(e)))

        # Return final statistics
        return self.get_final_stats()

