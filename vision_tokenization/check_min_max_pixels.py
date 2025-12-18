#!/usr/bin/env python3
"""
Analyze image sizes in a dataset and output recommended tokenization parameters.

Usage:
    python check_min_max_pixels.py \
        --dataset-name HuggingFaceM4/FineVision \
        --dataset-split train[:1000] \
        --image-field images \
        --mode sft

    # With config name for datasets with multiple configs
    python check_min_max_pixels.py \
        --dataset-name HuggingFaceM4/FineVision \
        --config-name default \
        --dataset-split train[:1000] \
        --image-field images \
        --mode sft
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def get_image_size(image):
    """Get width and height from various image formats."""
    if image is None:
        return None, None

    # Handle PIL Image
    if isinstance(image, Image.Image):
        return image.width, image.height

    # Handle dict with width/height keys (some datasets store metadata)
    if isinstance(image, dict):
        if 'width' in image and 'height' in image:
            return image['width'], image['height']
        # Try to load from bytes if present
        if 'bytes' in image:
            try:
                from io import BytesIO
                img = Image.open(BytesIO(image['bytes']))
                return img.width, img.height
            except Exception:
                pass

    # Handle numpy array
    if hasattr(image, 'shape'):
        if len(image.shape) == 3:
            # (H, W, C) format
            return image.shape[1], image.shape[0]
        elif len(image.shape) == 2:
            # (H, W) grayscale
            return image.shape[1], image.shape[0]

    return None, None


def analyze_dataset(
    dataset_name: str,
    dataset_split: str,
    image_field: str,
    config_name: str = None,
    max_samples: int = None,
    cache_dir: str = None,
):
    """
    Analyze image sizes in a dataset.

    Returns:
        dict with statistics about image sizes
    """
    print(f"\nLoading dataset: {dataset_name}")
    print(f"  Split: {dataset_split}")
    print(f"  Image field: {image_field}")
    if config_name:
        print(f"  Config: {config_name}")

    # Load dataset
    try:
        dataset = load_dataset(
            dataset_name,
            name=config_name,
            split=dataset_split,
            cache_dir=cache_dir,
        )
    except ValueError as e:
        error_msg = str(e)
        # Check if this is a missing config error
        if "Config name is missing" in error_msg or "pick one among" in error_msg.lower():
            print(f"\nError: This dataset requires a --config-name argument.")
            print(f"\n{error_msg}")
            print("\nExample usage:")
            print(f"  python check_min_max_pixels.py \\")
            print(f"      --dataset-name {dataset_name} \\")
            print(f"      --config-name <CONFIG_NAME> \\")
            print(f"      --dataset-split \"{dataset_split}\" \\")
            print(f"      --image-field {image_field} \\")
            print(f"      --mode <MODE>")
        else:
            print(f"\nError loading dataset: {e}")
            print("\nTry specifying --config-name if the dataset has multiple configurations.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nTry specifying --config-name if the dataset has multiple configurations.")
        sys.exit(1)

    # Limit samples if requested
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    print(f"  Samples to analyze: {len(dataset)}")

    # Collect image sizes
    widths = []
    heights = []
    pixels = []
    aspect_ratios = []
    errors = 0

    for sample in tqdm(dataset, desc="Analyzing images"):
        # Get image(s) from sample
        images = sample.get(image_field)

        if images is None:
            errors += 1
            continue

        # Handle single image or list of images
        if not isinstance(images, list):
            images = [images]

        for img in images:
            w, h = get_image_size(img)
            if w is not None and h is not None:
                widths.append(w)
                heights.append(h)
                pixels.append(w * h)
                aspect_ratios.append(w / h)
            else:
                errors += 1

    if not pixels:
        print("\nNo valid images found!")
        print(f"  Errors: {errors}")
        print(f"\nCheck that '{image_field}' is the correct field name.")
        print(f"Available fields: {list(dataset.features.keys())}")
        sys.exit(1)

    # Calculate statistics
    stats = {
        'count': len(pixels),
        'errors': errors,
        'pixels': {
            'min': int(np.min(pixels)),
            'max': int(np.max(pixels)),
            'mean': int(np.mean(pixels)),
            'median': int(np.median(pixels)),
            'p5': int(np.percentile(pixels, 5)),
            'p25': int(np.percentile(pixels, 25)),
            'p75': int(np.percentile(pixels, 75)),
            'p95': int(np.percentile(pixels, 95)),
        },
        'width': {
            'min': int(np.min(widths)),
            'max': int(np.max(widths)),
            'mean': int(np.mean(widths)),
            'median': int(np.median(widths)),
        },
        'height': {
            'min': int(np.min(heights)),
            'max': int(np.max(heights)),
            'mean': int(np.mean(heights)),
            'median': int(np.median(heights)),
        },
        'aspect_ratio': {
            'min': float(np.min(aspect_ratios)),
            'max': float(np.max(aspect_ratios)),
            'mean': float(np.mean(aspect_ratios)),
        },
    }

    # Size distribution (for understanding the data)
    size_buckets = defaultdict(int)
    for p in pixels:
        if p < 256*256:
            size_buckets['< 256x256'] += 1
        elif p < 512*512:
            size_buckets['256x256 - 512x512'] += 1
        elif p < 1024*1024:
            size_buckets['512x512 - 1024x1024'] += 1
        elif p < 2048*2048:
            size_buckets['1024x1024 - 2048x2048'] += 1
        else:
            size_buckets['> 2048x2048'] += 1

    stats['distribution'] = dict(size_buckets)

    return stats


def recommend_parameters(stats: dict) -> dict:
    """
    Recommend tokenization parameters based on image statistics.

    Returns:
        dict with recommended parameter values
    """
    p5 = stats['pixels']['p5']
    p95 = stats['pixels']['p95']
    median = stats['pixels']['median']

    # Tokenizer pixels: Use common vision model resolutions
    # Most vision tokenizers work well with 512x512 to 1024x1024

    # Min tokenizer pixels: Use 512x512 as baseline (good for most tokenizers)
    min_tok = 512 * 512

    # Max tokenizer pixels: Use 1024x1024 as baseline
    # If dataset has larger images (p95 > 1M pixels), could go to 1536x1536
    if p95 > 2048 * 2048:
        max_tok = 1536 * 1536
    else:
        max_tok = 1024 * 1024

    # Image filtering: Based on actual data distribution
    # Min image pixels: Use p5 or 256x256, whichever is larger
    # This filters out very small images that won't tokenize well
    min_img = max(256 * 256, p5)

    # Max image pixels: Use p95 or 2048x2048, whichever is smaller
    # This filters out unusually large images that slow down processing
    max_img = min(2048 * 2048, int(p95 * 1.5))

    # Round to nice numbers
    def round_to_nice(val):
        """Round to a nice power-of-2 or common resolution."""
        nice_values = [
            128*128, 192*192, 256*256, 384*384, 512*512,
            640*640, 768*768, 896*896, 1024*1024,
            1280*1280, 1536*1536, 2048*2048, 3072*3072, 4096*4096
        ]
        # Find closest nice value
        return min(nice_values, key=lambda x: abs(x - val))

    return {
        'min_tokenizer_pixels': round_to_nice(min_tok),
        'max_tokenizer_pixels': round_to_nice(max_tok),
        'min_image_pixels': round_to_nice(min_img),
        'max_image_pixels': round_to_nice(max_img),
    }


def format_pixels(pixels: int) -> str:
    """Format pixel count as dimension string (e.g., 512*512)."""
    import math
    sqrt = int(math.sqrt(pixels))
    # Check if it's a perfect square
    if sqrt * sqrt == pixels:
        return f"{sqrt}*{sqrt}"
    # Otherwise just return the number
    return str(pixels)


def generate_command(
    dataset_name: str,
    dataset_split: str,
    mode: str,
    params: dict,
    tokenizer_path: str = "my_omni_tokenizer/",
    output_dir: str = "my_tokenized_data/",
    num_gpus: int = 1,
    num_shards: int = 100,
    config_name: str = None,
    image_field: str = "images",
    text_field: str = "texts",
    cache_dir: str = None,
) -> str:
    """Generate the tokenization command.

    Note: tokenize.py uses argparse with subparsers. Common arguments (tokenizer-path,
    output-dir, num-gpus, device, pixel settings) must come BEFORE the 'hf' subcommand.
    Subparser-specific arguments (mode, dataset-name, etc.) come AFTER 'hf'.
    """
    # Common arguments (defined on main parser) - must come before 'hf'
    cmd_parts = [
        "python vision_tokenization/tokenize.py",
        f"    --tokenizer-path {tokenizer_path}",
        f"    --output-dir {output_dir}",
        f"    --num-gpus {num_gpus}",
        f"    --device cuda",
        f"    --min-tokenizer-pixels \"{format_pixels(params['min_tokenizer_pixels'])}\"",
        f"    --max-tokenizer-pixels \"{format_pixels(params['max_tokenizer_pixels'])}\"",
        f"    --min-image-pixels \"{format_pixels(params['min_image_pixels'])}\"",
        f"    --max-image-pixels \"{format_pixels(params['max_image_pixels'])}\"",
        # Subcommand
        "    hf",
        # Subparser-specific arguments (defined on hf subparser) - must come after 'hf'
        f"    --mode {mode}",
        f"    --dataset-name {dataset_name}",
    ]

    if config_name:
        cmd_parts.append(f"    --config-name {config_name}")

    cmd_parts.extend([
        f"    --dataset-split \"{dataset_split}\"",
        f"    --num-shards {num_shards}",
        f"    --image-field {image_field}",
        f"    --text-field {text_field}",
    ])

    # Add cache-dir if specified (subparser argument)
    if cache_dir:
        cmd_parts.append(f"    --cache-dir {cache_dir}")

    return " \\\n".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset image sizes and recommend tokenization parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-split", type=str, default="train[:1000]",
                        help="Dataset split to analyze (use small subset for speed)")
    parser.add_argument("--config-name", type=str, default=None,
                        help="Dataset configuration/subset name")
    parser.add_argument("--image-field", type=str, default="images",
                        help="Name of the image field in the dataset")
    parser.add_argument("--text-field", type=str, default="texts",
                        help="Name of the text field in the dataset")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory for dataset download")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to analyze (overrides split limit)")

    # Tokenization arguments (for command generation)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["image_only", "image2text", "text2image", "sft"],
                        help="Tokenization mode")
    parser.add_argument("--tokenizer-path", type=str, default="my_omni_tokenizer/",
                        help="Path to tokenizer (for generated command)")
    parser.add_argument("--output-dir", type=str, default="my_tokenized_data/",
                        help="Output directory (for generated command)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (for generated command)")
    parser.add_argument("--num-shards", type=int, default=100,
                        help="Number of shards (for generated command)")

    args = parser.parse_args()

    # Analyze dataset
    stats = analyze_dataset(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        image_field=args.image_field,
        config_name=args.config_name,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
    )

    # Print statistics
    print("\n" + "=" * 60)
    print("IMAGE SIZE STATISTICS")
    print("=" * 60)
    print(f"Images analyzed: {stats['count']}")
    print(f"Errors/skipped:  {stats['errors']}")

    print("\nPixel counts:")
    print(f"  Min:    {stats['pixels']['min']:>12,} ({format_pixels(stats['pixels']['min'])})")
    print(f"  5th %:  {stats['pixels']['p5']:>12,} ({format_pixels(stats['pixels']['p5'])})")
    print(f"  25th %: {stats['pixels']['p25']:>12,} ({format_pixels(stats['pixels']['p25'])})")
    print(f"  Median: {stats['pixels']['median']:>12,} ({format_pixels(stats['pixels']['median'])})")
    print(f"  Mean:   {stats['pixels']['mean']:>12,} ({format_pixels(stats['pixels']['mean'])})")
    print(f"  75th %: {stats['pixels']['p75']:>12,} ({format_pixels(stats['pixels']['p75'])})")
    print(f"  95th %: {stats['pixels']['p95']:>12,} ({format_pixels(stats['pixels']['p95'])})")
    print(f"  Max:    {stats['pixels']['max']:>12,} ({format_pixels(stats['pixels']['max'])})")

    print("\nDimensions:")
    print(f"  Width:  {stats['width']['min']} - {stats['width']['max']} (median: {stats['width']['median']})")
    print(f"  Height: {stats['height']['min']} - {stats['height']['max']} (median: {stats['height']['median']})")
    print(f"  Aspect: {stats['aspect_ratio']['min']:.2f} - {stats['aspect_ratio']['max']:.2f} (mean: {stats['aspect_ratio']['mean']:.2f})")

    print("\nSize distribution:")
    for bucket, count in sorted(stats['distribution'].items()):
        pct = 100 * count / stats['count']
        print(f"  {bucket:25s}: {count:>6} ({pct:>5.1f}%)")

    # Get recommendations
    params = recommend_parameters(stats)

    print("\n" + "=" * 60)
    print("RECOMMENDED PARAMETERS")
    print("=" * 60)
    print(f"Tokenizer pixels (preprocessing):")
    print(f"  --min-tokenizer-pixels \"{format_pixels(params['min_tokenizer_pixels'])}\"")
    print(f"  --max-tokenizer-pixels \"{format_pixels(params['max_tokenizer_pixels'])}\"")
    print(f"\nImage filtering (skip images outside range):")
    print(f"  --min-image-pixels \"{format_pixels(params['min_image_pixels'])}\"")
    print(f"  --max-image-pixels \"{format_pixels(params['max_image_pixels'])}\"")

    # Generate command
    cmd = generate_command(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        mode=args.mode,
        params=params,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        num_gpus=args.num_gpus,
        num_shards=args.num_shards,
        config_name=args.config_name,
        image_field=args.image_field,
        text_field=args.text_field,
        cache_dir=args.cache_dir,
    )

    print("\n" + "=" * 60)
    print("TOKENIZATION COMMAND")
    print("=" * 60)
    print(cmd)
    print("\n")


if __name__ == "__main__":
    main()
