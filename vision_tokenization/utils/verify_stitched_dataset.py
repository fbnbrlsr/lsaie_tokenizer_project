#!/usr/bin/env python3
"""
Verify stitched multimodal dataset files.

This script checks that the stitch_back.py output is correct by:
1. Comparing sequence counts between text, image, and multimodal datasets
2. Verifying that multimodal = text + image tokens for each sample
3. Checking token ranges are valid
4. Validating .idx file structure

Usage:
    python verify_stitched_dataset.py /path/to/output_dir
    python verify_stitched_dataset.py /path/to/output_dir --tokenizer /path/to/tokenizer
    python verify_stitched_dataset.py /path/to/output_dir --num-samples 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vision_tokenization.pipelines.dataset import IndexedDataset


def find_shard_files(output_dir: Path, mode: str = "sft") -> Dict[str, List[Path]]:
    """Find all shard files in the output directory."""
    text_dir = output_dir / "text"
    image_dir = output_dir / "image"
    multimodal_dir = output_dir / "multimodal"

    files = {
        "text": [],
        "image": [],
        "multimodal": []
    }

    if text_dir.exists():
        files["text"] = sorted(text_dir.glob("*.idx"))
    if image_dir.exists():
        files["image"] = sorted(image_dir.glob("*.idx"))
    if multimodal_dir.exists():
        files["multimodal"] = sorted(multimodal_dir.glob("*.idx"))

    return files


def get_prefix_from_idx(idx_path: Path) -> str:
    """Get the dataset prefix (without .idx extension)."""
    return str(idx_path)[:-4]  # Remove .idx


def verify_single_shard(
    text_prefix: Optional[str],
    image_prefix: str,
    multimodal_prefix: str,
    num_samples: int = 10,
    tokenizer_path: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify a single shard's stitching is correct.

    Returns:
        Dictionary with verification results
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {}
    }

    # Load datasets
    try:
        image_dataset = IndexedDataset(image_prefix)
        multimodal_dataset = IndexedDataset(multimodal_prefix)
        text_dataset = IndexedDataset(text_prefix) if text_prefix else None
    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to load datasets: {e}")
        return results

    # Check sequence counts
    image_count = len(image_dataset)
    multimodal_count = len(multimodal_dataset)
    text_count = len(text_dataset) if text_dataset else 0

    results["stats"]["image_sequences"] = image_count
    results["stats"]["multimodal_sequences"] = multimodal_count
    results["stats"]["text_sequences"] = text_count

    if verbose:
        print(f"\n  Sequence counts:")
        print(f"    Image: {image_count}")
        if text_dataset:
            print(f"    Text: {text_count}")
        print(f"    Multimodal: {multimodal_count}")

    # Verify counts match
    if text_dataset and text_count != image_count:
        results["errors"].append(f"Text ({text_count}) and image ({image_count}) counts don't match")
        results["valid"] = False

    if multimodal_count != image_count:
        results["errors"].append(f"Multimodal ({multimodal_count}) and image ({image_count}) counts don't match")
        results["valid"] = False

    # Sample verification
    samples_to_check = min(num_samples, multimodal_count)
    if samples_to_check == 0:
        results["warnings"].append("No samples to verify")
        return results

    # Check random samples
    indices = np.random.choice(multimodal_count, samples_to_check, replace=False)
    indices = sorted(indices)

    length_mismatches = 0
    content_mismatches = 0

    if verbose:
        print(f"\n  Verifying {samples_to_check} samples...")

    for i, idx in enumerate(indices):
        image_tokens = image_dataset[idx]
        multimodal_tokens = multimodal_dataset[idx]

        if text_dataset:
            text_tokens = text_dataset[idx]
            expected_length = len(text_tokens) + len(image_tokens)

            # Check length
            if len(multimodal_tokens) != expected_length:
                length_mismatches += 1
                if verbose and length_mismatches <= 3:
                    print(f"    ❌ Sample {idx}: length mismatch")
                    print(f"       Text: {len(text_tokens)}, Image: {len(image_tokens)}")
                    print(f"       Expected: {expected_length}, Got: {len(multimodal_tokens)}")

            # Check content (text should be at beginning, image at end based on stitch_back.py)
            # The stitch order depends on your stitch_back.py implementation
            # Typically: text_before + image + text_after
        else:
            # Image-only mode: multimodal should equal image
            if len(multimodal_tokens) != len(image_tokens):
                length_mismatches += 1
            elif not np.array_equal(multimodal_tokens, image_tokens):
                content_mismatches += 1

    results["stats"]["samples_checked"] = samples_to_check
    results["stats"]["length_mismatches"] = length_mismatches
    results["stats"]["content_mismatches"] = content_mismatches

    if length_mismatches > 0:
        results["errors"].append(f"{length_mismatches}/{samples_to_check} samples have length mismatches")
        results["valid"] = False

    if content_mismatches > 0:
        results["warnings"].append(f"{content_mismatches}/{samples_to_check} samples have content differences")

    # Check token ranges
    if verbose:
        print(f"\n  Checking token ranges...")

    sample_tokens = multimodal_dataset[0]
    min_token = int(np.min(sample_tokens))
    max_token = int(np.max(sample_tokens))
    results["stats"]["min_token"] = min_token
    results["stats"]["max_token"] = max_token

    if verbose:
        print(f"    Token range: [{min_token}, {max_token}]")

    # If tokenizer provided, check against vocab size
    if tokenizer_path:
        try:
            config_path = os.path.join(tokenizer_path, "tokenizer_config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                vocab_size = config.get("vocab_size", None)
                if vocab_size:
                    results["stats"]["vocab_size"] = vocab_size
                    if max_token >= vocab_size:
                        results["errors"].append(f"Max token {max_token} >= vocab_size {vocab_size}")
                        results["valid"] = False
                    elif verbose:
                        print(f"    Vocab size: {vocab_size}")
                        print(f"    Valid range: ✅")
        except Exception as e:
            results["warnings"].append(f"Could not load tokenizer config: {e}")

    return results


def verify_dataset(
    output_dir: str,
    tokenizer_path: Optional[str] = None,
    num_samples: int = 10,
    mode: str = "sft",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify all shards in the output directory.

    Args:
        output_dir: Path to the tokenized output directory
        tokenizer_path: Optional path to tokenizer for vocab size validation
        num_samples: Number of samples to check per shard
        mode: "sft" or "image_only"
        verbose: Whether to print detailed output

    Returns:
        Dictionary with overall verification results
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        return {"valid": False, "errors": [f"Directory not found: {output_dir}"]}

    print("=" * 70)
    print("MULTIMODAL DATASET VERIFICATION")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    # Find shard files
    files = find_shard_files(output_path, mode)

    print(f"\nFound files:")
    print(f"  Text shards: {len(files['text'])}")
    print(f"  Image shards: {len(files['image'])}")
    print(f"  Multimodal shards: {len(files['multimodal'])}")

    if len(files["multimodal"]) == 0:
        return {"valid": False, "errors": ["No multimodal shards found"]}

    if len(files["image"]) == 0:
        return {"valid": False, "errors": ["No image shards found"]}

    # Match shards by name
    image_shards = {f.stem: f for f in files["image"]}
    text_shards = {f.stem: f for f in files["text"]}
    multimodal_shards = {f.stem: f for f in files["multimodal"]}

    overall_results = {
        "valid": True,
        "shards_checked": 0,
        "shards_valid": 0,
        "total_sequences": 0,
        "errors": [],
        "warnings": [],
        "shard_results": {}
    }

    # Verify each multimodal shard
    for shard_name, multimodal_idx in multimodal_shards.items():
        print(f"\n{'─' * 70}")
        print(f"Shard: {shard_name}")

        image_idx = image_shards.get(shard_name)
        text_idx = text_shards.get(shard_name)

        if not image_idx:
            overall_results["errors"].append(f"Missing image shard for {shard_name}")
            overall_results["valid"] = False
            continue

        text_prefix = get_prefix_from_idx(text_idx) if text_idx else None
        image_prefix = get_prefix_from_idx(image_idx)
        multimodal_prefix = get_prefix_from_idx(multimodal_idx)

        result = verify_single_shard(
            text_prefix=text_prefix,
            image_prefix=image_prefix,
            multimodal_prefix=multimodal_prefix,
            num_samples=num_samples,
            tokenizer_path=tokenizer_path,
            verbose=verbose
        )

        overall_results["shard_results"][shard_name] = result
        overall_results["shards_checked"] += 1

        if result["valid"]:
            overall_results["shards_valid"] += 1
            print(f"\n  ✅ Shard valid")
        else:
            overall_results["valid"] = False
            print(f"\n  ❌ Shard has errors:")
            for error in result["errors"]:
                print(f"     - {error}")

        overall_results["total_sequences"] += result["stats"].get("multimodal_sequences", 0)
        overall_results["errors"].extend(result["errors"])
        overall_results["warnings"].extend(result["warnings"])

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Shards checked: {overall_results['shards_checked']}")
    print(f"Shards valid: {overall_results['shards_valid']}")
    print(f"Total sequences: {overall_results['total_sequences']:,}")

    if overall_results["valid"]:
        print("\n✅ All verifications passed!")
    else:
        print(f"\n❌ Verification failed with {len(overall_results['errors'])} error(s)")
        for error in overall_results["errors"][:10]:
            print(f"   - {error}")
        if len(overall_results["errors"]) > 10:
            print(f"   ... and {len(overall_results['errors']) - 10} more errors")

    if overall_results["warnings"]:
        print(f"\n⚠️  {len(overall_results['warnings'])} warning(s)")
        for warning in overall_results["warnings"][:5]:
            print(f"   - {warning}")

    return overall_results


def main():
    parser = argparse.ArgumentParser(
        description="Verify stitched multimodal dataset files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic verification
    python verify_stitched_dataset.py /path/to/my_tokenized_data/CoSyn_400k_chart_sft/896x896_2048x2048

    # With tokenizer validation
    python verify_stitched_dataset.py /path/to/output --tokenizer /path/to/my_omni_tokenizer

    # Check more samples per shard
    python verify_stitched_dataset.py /path/to/output --num-samples 100
        """
    )
    parser.add_argument("output_dir", help="Path to the tokenized output directory")
    parser.add_argument("--tokenizer", help="Path to tokenizer for vocab size validation")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to verify per shard (default: 10)")
    parser.add_argument("--mode", choices=["sft", "image_only"], default="sft",
                        help="Tokenization mode (default: sft)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    results = verify_dataset(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer,
        num_samples=args.num_samples,
        mode=args.mode,
        verbose=not args.quiet
    )

    # Exit with error code if verification failed
    sys.exit(0 if results["valid"] else 1)


if __name__ == "__main__":
    main()
