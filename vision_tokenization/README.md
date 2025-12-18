# Vision Tokenization Pipeline

Unified pipeline for tokenizing large-scale vision datasets with support for multiple modes and vision tokenizers.

**Important:** This pipeline requires an omni-tokenizer (extended text tokenizer with other modality vocabularies). The omni-tokenizer wraps token IDs from modality-specific tokenizers into the unified vocabulary. For example, vision token ID 100 becomes `<|visual token 00100|>` in the omni-tokenizer's vocabulary, with corresponding entries in the model's input and output embeddings. However, during our actual tokenization pipeline, we use the modality-specific tokenizer (e.g., Emu3VisionTokenizer) to get raw token indices, then apply a simple offset to map them to the omni-tokenizer's ID space.

**Why this matters:** During inference, the model can predict tokens from any modality (text, vision, audio, etc.). The omni-tokenizer handles the unified token space, while modality-specific tokenizers (vision, audio) decode their respective tokens back to images, audio, etc. See [`utils/omni_tokenizer/README.md`](utils/omni_tokenizer/README.md) for creating omni-tokenizers.

---

## Pipeline Overview

This pipeline implements **separate modality tokenized storage** for datasets, followed by a **stitching step** to merge them back into a unified multimodal format. The workflow is:

1. **Build the omni-tokenizer** (once)
2. **Tokenize the dataset** - produces separate `text/` and `image/` idx/bin files
3. **Stitch back** - merge separate modality files into a single `multimodal/` output

### Quick Start (3-Step Pipeline)

```bash
# Step 1: Build the omni-tokenizer (run once)
python my_build_omni_tokenizer.py

# Step 2: Tokenize with separate modality storage
python vision_tokenization/tokenize.py hf \
    --mode image_only \
    --dataset-name laion/laion-high-resolution \
    --dataset-split train[:10000] \
    --tokenizer-path /path/to/my_omni_tokenizer \
    --output-dir /path/to/my_tokenized_data \
    --num-gpus 1 \
    --num-shards 100 \
    --device cuda \
    --min-tokenizer-pixels "512*512" \
    --max-tokenizer-pixels "1024*1024" \
    --min-image-pixels "256*256" \
    --max-image-pixels "2048*2048"

# Step 3: Stitch back to create merged multimodal idx/bin files
python vision_tokenization/pipelines/stitch_back.py
```

---

## Token Structure Format

Images are tokenized into a structured sequence with special tokens marking boundaries and rows:

```
[BOS]
<|img_start|>
"32*32"                    # Image dimensions as text
<|img_token_start|>
<|visual token 00100|>     # Row 1, column 1
<|visual token 00523|>     # Row 1, column 2
...
<|img_end_of_row|>         # End of row 1
<|visual token 01244|>     # Row 2, column 1
...
<|img_end_of_row|>         # End of row 2
...                        # All H rows
<|img_end_of_frame|>       # End of entire image
<|img_end|>
[EOS]
```

This structure is created by the `encapsulate_image()` method in [`vokenizers/emu/image_only.py`](vokenizers/emu/image_only.py), which wraps raw vision indices with boundary markers and spatial information. The complete pipeline is orchestrated by `tokenize_image()` in the same file.

**Tokenizer Class Hierarchy:**
```
EMUImageOnlyTokenizer (base class)
├── tokenize_image()      # Core method for image tokenization
├── encapsulate_image()   # Wraps vision indices with structure tokens
│
├─→ EMUImageTextPairTokenizer
│   └── tokenize_image_text_pair()
│       ├── Calls self.tokenize_image() for image (GPU, parallel)
│       ├── Tokenizes text separately (CPU, parallel)
│       └── Concatenates based on mode (image2text or text2image)
│
└─→ EMUSftTokenizer
    └── tokenize_conversation()
        ├── Calls self.tokenize_image() for image (GPU, parallel)
        ├── Applies chat template and tokenizes text (CPU, parallel, includes <|image|> placeholder)
        └── Replaces <|image|> placeholder token with actual vision tokens
```

## Quick Start

### Image-Only Tokenization
```bash
python tokenize.py hf \
    --mode image_only \
    --dataset-name HuggingFaceM4/FineVision \
    --config-name CoSyn_400k_chart \
    --dataset-split train \
    --tokenizer-path /path/to/omni/tokenizer \
    --output-dir ./output \
    --num-gpus 4 \
    --num-shards 100 \
    --device cuda
```

### Resume from Checkpoint
```bash
# If processing was interrupted, simply add --resume to skip completed shards
python tokenize.py hf \
    --config config.json \
    --resume
```

### SFT Tokenization (Conversations with Images)
```bash
python tokenize.py hf \
    --mode sft \
    --dataset-name HuggingFaceM4/FineVision \
    --config-name CoSyn_400k_chart \
    --dataset-split train \
    --tokenizer-path /path/to/omni/instruct/tokenizer \
    --output-dir ./output \
    --num-gpus 4 \
    --num-shards 100 \
    --device cuda
```

## Supported Modes

**Note**: Currently supports **single image** per sample. Multi-image interleaving is not yet supported.

- **`image_only`** - Tokenize single images only (for pretraining)
- **`image2text`** - Single image followed by text caption
- **`text2image`** - Text prompt followed by single image
- **`sft`** - Supervised fine-tuning with conversations (single image + text)

## Token Separation Architecture

The pipeline stores **text and image tokens separately** in independent `.bin/.idx` file pairs. This design enables:

### Benefits

- **Easy Tokenizer Switching**: Replace text tokenizer without re-tokenizing images
- **Chat Template Updates**: Change chat templates by only re-processing text tokens
- **Modality-Specific Processing**: Apply different post-processing to text vs image tokens
- **Storage Flexibility**: Store text and image tokens on different storage tiers if needed

### How It Works

1. **Tokenization**: The tokenizer returns a dictionary with separate `text` and `image` tensors
2. **Storage**: Two `IndexedDatasetBuilder` instances write to `text/` and `image/` subdirectories
3. **Alignment**: Document N in text files corresponds to document N in image files (maintained via empty documents when needed)
4. **Special Tokens**: All special tokens (BOS, EOS, img_start, img_end, etc.) are stored with image tokens

### SFT Mode Specifics

For SFT mode, text tokens are split around the `<|image|>` placeholder:
- Text before and after the image are concatenated into a single sequence
- Metadata stores the split lengths (`text_before_len`, `text_after_len`, `image_position`) for reconstruction during stitching
- This enables flexible positioning of images within conversations

**SFT Output Structure:**
```
output_dir/
├── text/
│   └── rank_0_shard_0_100.bin/.idx
├── image/
│   └── rank_0_shard_0_100.bin/.idx
├── metadata/                                    # SFT mode only
│   └── rank_0_shard_0_100_metadata.json         # Per-sample metadata
└── dataset_info.json
```

**Metadata JSON format:**
```json
[
  {"idx": 0, "text_before_len": 50, "text_after_len": 150, "image_position": 50},
  {"idx": 1, "text_before_len": 75, "text_after_len": 120, "image_position": 75}
]
```

**SFT Stitching:** During stitching, metadata is used to reconstruct the correct token order:
- `text_before_len` determines where to split text tokens
- Final sequence: `text_before + image + text_after`

## Configuration Options

### Common Arguments

- `--tokenizer-path` - Path to omni-tokenizer (base for pretraining, instruct for sft). The vision tokenizer type and path are automatically loaded from the omni-tokenizer's config.
- `--output-dir` - Output directory for tokenized data
- `--num-gpus` - Number of GPUs for distributed processing
- `--device` - Device to use (cuda or cpu)

### Image Resolution Control

**Tokenizer Resolution** (controls vision tokenizer behavior):
- `--min-tokenizer-pixels` - Minimum pixels for tokenizer (e.g., "512*512" or "262144")
- `--max-tokenizer-pixels` - Maximum pixels for tokenizer (e.g., "1024*1024" or "1048576")

**Dataset Filtering** (filters which images to process):
- `--min-image-pixels` - Filter out images smaller than this
- `--max-image-pixels` - Filter out images larger than this

### Dataset Options

- `--dataset-name` - HuggingFace dataset name
- `--config-name` - Dataset configuration/subset
- `--dataset-split` - Dataset split (train/validation/test)
- `--cache-dir` - Cache directory for datasets
- `--max-samples` - Maximum samples to process (for testing)

### Processing Options

- `--num-shards` - Number of shards for distributed processing and checkpointing (required)
- `--num-proc` - Number of processes for dataset loading
- `--image-field` - Name of image field in dataset (default: "images")
- `--text-field` - Name of text field in dataset (default: "texts")
- `--resume` - Resume from existing checkpoint, skipping completed shards

## Configuration Files

You can use JSON configuration files instead of CLI arguments:

```json
{
  "dataset_name": "HuggingFaceM4/FineVision",
  "config_name": "CoSyn_400k_chart",
  "dataset_split": "train",
  "mode": "sft",
  "tokenizer_path": "/path/to/tokenizer",
  "output_dir": "./output",
  "num_gpus": 4,
  "num_shards": 100,
  "device": "cuda",
  "min_tokenizer_pixels": "512*512",
  "max_tokenizer_pixels": "1024*1024"
}
```

### Important: Shards in HuggingFace vs WebDataset

**Note:** The concept of "shards" differs between HuggingFace and WebDataset formats:

- **WebDataset shards**: Physical `.tar` files on disk. Each shard is a separate file containing a subset of data.
- **HuggingFace shards**: Logical views of the dataset created using `.shard()` method. The entire dataset stays as one unit, but we create efficient strided views for parallel processing.

When using `--num-shards` with HuggingFace datasets:
- It doesn't create physical shard files for input data
- It creates logical shards for processing, which results in separate output files (`rank_XXX_shard_YYYYY.bin/idx`)
- Each logical shard is processed atomically for checkpointing
- `num_shards` should be a multiple of `num_gpus` (workers) for optimal load balancing
  - Example: 8 workers → use 8, 16, 24, 32, 40, 48... shards
  - The pipeline will automatically adjust if `num_shards < num_gpus` to ensure each worker has work

## Processing Large Datasets via Config

For large datasets, simply modify the `dataset_split` field to process subsets:

```json
{
  "dataset_name": "massive_dataset",
  "config_name": "some_config",
  "dataset_split": "train[:10000000]",  // Process first 10M samples
  "mode": "image_only",
  "tokenizer_path": "/path/to/tokenizer",
  "output_dir": "./output",
  "num_gpus": 8,
  "num_shards": 1000,  // Enable shard-based checkpointing
  "device": "cuda"
}
```

**Dataset split examples:**
- `"dataset_split": "train[:10000000]"` - First 10M samples
- `"dataset_split": "train[10000000:20000000]"` - Samples 10M to 20M
- `"dataset_split": "train[:50%]"` - First half of dataset
- `"dataset_split": "train[-5000000:]"` - Last 5M samples

**Note:** Vision tokenizer type and path are automatically loaded from the omni-tokenizer's `tokenizer_config.json`.

Use with:
```bash
python tokenize.py hf --config config.json
```

CLI arguments override config file values.

## Output Format

The pipeline creates Megatron-LM IndexedDataset format with **separated text and image storage**. Text and image tokens are stored in separate subdirectories with matching shard filenames:

### For image2text, text2image, and sft modes:

```
output_dir/
└── config_name_{mode}/
    ├── text/
    │   ├── rank_0_shard_0_32.bin   # Text tokens only
    │   ├── rank_0_shard_0_32.idx   # Text index
    │   ├── rank_0_shard_1_32.bin
    │   ├── rank_0_shard_1_32.idx
    │   └── ...
    ├── image/
    │   ├── rank_0_shard_0_32.bin   # Image tokens only (with special tokens)
    │   ├── rank_0_shard_0_32.idx   # Image index
    │   ├── rank_0_shard_1_32.bin
    │   ├── rank_0_shard_1_32.idx
    │   └── ...
    └── dataset_info.json           # Processing metadata
```

### For image_only mode:

```
output_dir/
└── config_name_image_only/
    ├── image/
    │   ├── rank_0_shard_0_32.bin   # Image tokens only
    │   ├── rank_0_shard_0_32.idx
    │   └── ...
    └── dataset_info.json
```

**Note**: In image_only mode, no `text/` directory is created since there are no text tokens.

### Key Features:

- **Separated Storage**: Text and image tokens are stored independently, enabling easy tokenizer/chat template switching
- **Document Alignment**: Document N in text files corresponds to document N in image files (maintained via empty documents when needed)
- **Special Token Handling**: All special tokens (BOS, EOS, img_start, img_end, etc.) are stored with image tokens to preserve encapsulation
- **Atomic Checkpointing**: Each shard is saved independently with both text and image files
- **Easy Resume**: The total shard count is embedded in filenames
- **Clear Tracking**: You can see progress at a glance

The filename format `rank_{worker}_shard_{id}_{total}` enables efficient distributed processing and checkpointing.

## Checkpoint and Resume

### How It Works

1. Each shard saves to both `text/rank_{worker}_shard_{id}_{total}.bin/.idx` and `image/rank_{worker}_shard_{id}_{total}.bin/.idx`
2. The `.idx` files mark completion (written last)
3. For non-image_only modes: Resume detects completed shards by checking for **BOTH** text and image `.idx` files
4. For image_only mode: Resume checks only image `.idx` files
5. Only uncompleted shards are reprocessed

### Resume Usage

```bash
# Initial run
python tokenize.py hf --config config.json

# Resume after interruption
python tokenize.py hf --config config.json --resume
```

### Edge Cases

**Incomplete shards** (only text OR only image files exist):
- Logged as warnings during resume
- Automatically reprocessed to create both files

**Missing text directory** (for non-image_only modes):
- Resume will not find any completed shards
- All shards will be processed from scratch

**Inconsistent shard counts**:
- Program stops with error if existing files have different total shards
- Example error:
  ```
  ERROR: Inconsistent total shard counts found: [32, 64]
    32 total shards: 20 files
    64 total shards: 10 files
  Clean the output directory or use a different output path
  ```

**Mismatched resume**:
- If resuming with different `num_shards`, program stops:
  ```
  ERROR: No existing shards match expected count (100). Found shard counts: [32]
  To resume, use --num-shards 32 or start fresh without --resume
  ```


## Examples

### Example 1: Image-Only Pretraining Dataset

```bash
python tokenize.py hf \
    --mode image_only \
    --dataset-name laion/laion-high-resolution \
    --dataset-split train \
    --tokenizer-path ./llama3_emu3_base \
    --output-dir ./tokenized_data \
    --num-gpus 8 \
    --num-shards 800 \
    --device cuda \
    --min-tokenizer-pixels "512*512" \
    --max-tokenizer-pixels "1024*1024" \
    --min-image-pixels "256*256" \
    --max-image-pixels "2048*2048"
```

### Example 2: SFT with FineVision

```bash
python tokenize.py hf \
    --mode sft \
    --dataset-name HuggingFaceM4/FineVision \
    --config-name CoSyn_400k_chart \
    --dataset-split train \
    --tokenizer-path ./llama3_emu3_instruct \
    --output-dir ./sft_data \
    --num-gpus 4 \
    --num-shards 100 \
    --device cuda \
    --max-samples 1000  # For testing
```

### Example 3: Using Emu3.5

```bash
# Just use an Emu3.5 omni-tokenizer - vision tokenizer is auto-loaded!
python tokenize.py hf \
    --mode image_only \
    --dataset-name ... \
    --tokenizer-path ./llama3_emu3.5_base \
    --output-dir ./emu3.5_data \
    --num-gpus 4 \
    --num-shards 100 \
    --device cuda
```

## Performance Tips

1. **Shard Count**: Choose appropriate `--num-shards` based on dataset size (see recommendations above)
2. **GPU Count**: More GPUs = faster processing with Ray's work-stealing
3. **Resolution**: Lower max pixels = faster tokenization but lower quality
4. **Filtering**: Use min/max image pixels to skip unwanted images early
5. **Load Balancing**: Set `num_shards` as a multiple of `num_gpus` for optimal distribution

## Troubleshooting

### Out of Memory
- Reduce `--max-tokenizer-pixels`
- Increase `--num-shards` to reduce samples per shard
- Use fewer GPUs

### Slow Processing
- Ensure `--num-shards` is a multiple of `--num-gpus`
- Use more GPUs with `--num-gpus`
- Check GPU utilization with `nvidia-smi`

### Missing Images or Text
- Check `--image-field` and `--text-field` match your dataset
- Use `--max-samples 10` to test on small subset first

## Architecture

The pipeline uses:
- **Ray** for distributed GPU processing with dynamic work-stealing
- **HuggingFace Datasets** for efficient data loading via `.shard()` method
- **Megatron-LM IndexedDataset** for training-optimized output format
- **Shard-based processing** for atomic checkpointing and resume capability

Workers pull shards dynamically from a shared queue, ensuring optimal GPU utilization. Each shard is processed independently and saved as a separate file pair (`rank_XXX_shard_YYYYY.bin/idx`), enabling robust checkpointing and recovery.

## Related Tools

- **Omni-Tokenizer Creation**: See `utils/omni_tokenizer/README.md`
- **Old WebDataset Pipeline**: See `README_OLD_WEBDATASET.md` (legacy)
- **Image Size Analyzer**: See "Step 2: Tokenize with Separate Modality Storage" below

---

## Detailed Pipeline Steps

This section provides detailed instructions for the 3-step pipeline.

### Step 1: Build the Omni-Tokenizer

The omni-tokenizer extends a base text tokenizer (e.g., LLaVA) with vision tokens. Run this once before tokenizing any datasets.

**Script:** [`my_build_omni_tokenizer.py`](../my_build_omni_tokenizer.py)

```python
# Configure these paths in my_build_omni_tokenizer.py:
TEXT_TOKENIZER_PATH = "llava-hf/llava-1.5-7b-hf"
OUTPUT_PATH = "/path/to/my_omni_tokenizer"
VISION_TOKENIZER_PATH = 'BAAI/Emu3-VisionTokenizer'
VISION_TOKENIZER = 'Emu3'
```

```bash
# Run once to create the tokenizer
python my_build_omni_tokenizer.py
```

**Output:** Creates a tokenizer directory with:

- `tokenizer.json` - Main tokenizer file
- `tokenizer_config.json` - Config with vocab sizes and vision tokenizer info
- `added_tokens.json` - Special and vision tokens mapping
- `special_tokens_map.json` - Special token definitions
- `vision_token_mapping.json` - Vision token ID mappings

### Step 2: Tokenize with Separate Modality Storage

Tokenize the dataset into separate `text/` and `image/` subdirectories. This separation enables easy tokenizer switching without re-tokenizing images.

**Script:** [`vision_tokenization/tokenize.py`](tokenize.py)

#### Finding Optimal Pixel Parameters

Before tokenizing, use `check_min_max_pixels.py` to analyze your dataset's image sizes and get recommended parameters:

```bash
# Analyze dataset and get recommended parameters + ready-to-use command
python vision_tokenization/check_min_max_pixels.py \
    --dataset-name laion/laion-high-resolution \
    --dataset-split "train[:1000]" \
    --image-field image \
    --mode image_only \
    --tokenizer-path /path/to/my_omni_tokenizer \
    --output-dir /path/to/my_tokenized_data

# For datasets with multiple configs (e.g., FineVision)
python vision_tokenization/check_min_max_pixels.py \
    --dataset-name HuggingFaceM4/FineVision \
    --config-name CoSyn_400k_chart \
    --dataset-split "train[:1000]" \
    --image-field images \
    --mode sft
```

This outputs image size statistics and a complete tokenization command you can copy-paste.

#### Running Tokenization

```bash
python vision_tokenization/tokenize.py hf \
    --mode image_only \
    --dataset-name laion/laion-high-resolution \
    --dataset-split train[:10000] \
    --tokenizer-path /path/to/my_omni_tokenizer \
    --output-dir /path/to/my_tokenized_data \
    --num-gpus 1 \
    --num-shards 100 \
    --device cuda \
    --min-tokenizer-pixels "512*512" \
    --max-tokenizer-pixels "1024*1024" \
    --min-image-pixels "256*256" \
    --max-image-pixels "2048*2048"
```

**Output Structure:**

```
my_tokenized_data/
├── text/
│   ├── rank_0_shard_0_100.bin
│   ├── rank_0_shard_0_100.idx
│   └── ...
├── image/
│   ├── rank_0_shard_0_100.bin
│   ├── rank_0_shard_0_100.idx
│   └── ...
└── dataset_info.json
```

### Step 3: Stitch Back to Multimodal Format

Merge the separate text and image token files into a unified multimodal format suitable for training.

**Script:** [`vision_tokenization/pipelines/stitch_back.py`](pipelines/stitch_back.py)

```python
# Edit stitch_back.py to configure:
results = stitch_shards(
    output_dir='/path/to/my_tokenized_data/output_folder',
    mode='image_only'  # or 'image2text', 'text2image', 'sft'
)
```

```bash
python vision_tokenization/pipelines/stitch_back.py
```

**Output:** Creates a `multimodal/` subdirectory with merged token files:

```
my_tokenized_data/
├── text/
├── image/
├── multimodal/
│   ├── rank_0_shard_0_100.bin
│   ├── rank_0_shard_0_100.idx
│   └── ...
└── dataset_info.json
```

The `stitch_back.py` script provides two functions:

- `stitch_shards()` - Merges text/image shards into multimodal shards (recommended)
- `stich_together_tokens()` - Alternative function that batches samples into multimodal datasets

#### Mode-Specific Stitching Behavior

| Mode | Stitching Order | Metadata Used |
|------|-----------------|---------------|
| `image_only` | `[image]` | No |
| `image2text` | `[image] + [text]` | No |
| `text2image` | `[image] + [text]` (fallback) | No |
| `sft` | `[text_before] + [image] + [text_after]` | Yes - uses `metadata/` JSON files |

**SFT Stitching Details:**

For SFT mode, `stitch_shards()` reads per-sample metadata from `metadata/*.json` files to correctly reconstruct token order:

1. Loads `metadata/{shard}_metadata.json` for each shard
2. For each sample, reads `text_before_len` from metadata
3. Splits text tokens: `text_before = text[:text_before_len]`, `text_after = text[text_before_len:]`
4. Reconstructs: `combined = text_before + image + text_after`

If metadata file is missing, falls back to simple `[image] + [text]` concatenation with a warning.

---

## New Utility Classes

### IndexedDataset

Class for reading tokenized data from `.bin/.idx` files.

**Location:** [`vision_tokenization/pipelines/dataset.py`](pipelines/dataset.py)

```python
from vision_tokenization.pipelines.dataset import IndexedDataset

# Load a tokenized shard
dataset = IndexedDataset("/path/to/shard_prefix")  # without .bin/.idx extension

# Access samples
sample = dataset[0]  # Returns numpy array of token IDs
print(f"Number of samples: {len(dataset)}")
print(f"Sample token count: {len(sample)}")
```

### IO Utilities

Low-level readers for index and binary files.

**Location:** [`vision_tokenization/io_utils/readers.py`](io_utils/readers.py)

- `IndexReader` - Reads `.idx` files (sequence lengths, pointers, document indices)
- `MMapBinReader` - Memory-mapped reader for `.bin` files (efficient for large files)
- `FileBinReader` - File-based reader for `.bin` files (lower memory usage)

### DType Utilities

Data type utilities for token storage.

**Location:** [`vision_tokenization/utils/types.py`](utils/types.py)

```python
from vision_tokenization.utils.types import DType

# Get optimal dtype for vocabulary size
dtype = DType.optimal_dtype(vocab_size=128256)

# Convert dtype code to numpy dtype
dtype = DType.dtype_from_code(code)
```
