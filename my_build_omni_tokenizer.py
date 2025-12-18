from vision_tokenization.utils.omni_tokenizer.core import create_base_tokenizer
import argparse
import sys
import os
from datasets import load_dataset
from Tokenizer.Emu3VisionTokenizer import Emu3VisionTokenizer
from vision_tokenization.vokenizers.emu.image_only import EMUImageOnlyTokenizer
from vision_tokenization.vokenizers.emu.image_text_pair import EMUImageTextPairTokenizer
from vision_tokenization.vokenizers.emu.sft import EMUSftTokenizer



def build_tokenizer(
        text_tokenizer_path,
        output_path,
        vision_tokenizer_path,
        vision_tokenizer
):

    # Create omni-tokenizer
    tokenizer, stats = create_base_tokenizer(
        text_tokenizer_path = text_tokenizer_path,
        output_path = output_path,
        vision_tokenizer_path = vision_tokenizer_path,
        vision_tokenizer = vision_tokenizer
    )

    print("\n" + "="*60)
    print("OMNI-TOKENIZER CREATION SUMMARY")
    print("="*60)
    print(f"Text tokenizer:           {stats['text_tokenizer']}")
    print(f"Vision tokenizer:         {stats['vision_tokenizer']}")
    print(f"Tokenizer type:           {stats['tokenizer_type']}")
    print(f"Original vocabulary size: {stats['original_vocab_size']:,}")
    print(f"Structure tokens added:   {stats['structure_tokens_added']:,}")
    print(f"Reserved tokens added:    {stats['reserved_tokens_added']:,}")
    print(f"Visual tokens added:      {stats['visual_tokens_added']:,}")
    print(f"Final vocabulary size:    {stats['final_vocab_size']:,}")
    print(f"Total tokens added:       {stats['final_vocab_size'] - stats['original_vocab_size']:,}")
    print("="*60)
    print("\n✅ Base omni-tokenizer created successfully!")
    print(f"   Saved to: {output_path}")

def load_coco_dataset(num_samples = 10):

    dataset = load_dataset(
        path="Open-Bee/Honey-Data-15M",
        name="COCO",
        split=f"train[:{num_samples}]",
        streaming=False,
    )

    return dataset

def get_image_caption_pairs(init_dataset, num_pairs = 10):

    image_caption_list = []
    for idx, sample in enumerate(init_dataset):

        if len(image_caption_list) >= num_pairs:
            break

        caption = None
        image = None

        images = sample["images"]
        if isinstance(images, list) and len(images) > 0:
            image = images[0]
        else:
            image = images

        convs = sample["conversations"]
        if isinstance(convs, list) and len(convs) > 0:
            # Get the first human message
            for conv in convs:
                if isinstance(conv, dict) and conv.get("from") == "human":
                    caption = conv.get("value", "")
                    break

        if image is not None and caption is not None:
            d = {
                "image": image,
                "caption": caption
            }
            image_caption_list.append(d)

    return image_caption_list

def create_tokenizer(
    mode: str,
    tokenizer_path: str,
    device: str = "cuda",
    min_pixels: int = 512 * 512,
    max_pixels: int = 1024 * 1024, 
):
    tokenizers = {
        "image_only": EMUImageOnlyTokenizer,
        "image2text": EMUImageTextPairTokenizer,  # image->text (captioning)
        "text2image": EMUImageTextPairTokenizer,  # text->image (generation)
        "sft": EMUSftTokenizer
    }

    tokenizer_class = tokenizers[mode]

    if mode in ["image2text", "text2image"]:
        # EMUImageTextPairTokenizer requires mode parameter
        return tokenizer_class(
            text_tokenizer_path=tokenizer_path,
            device=device,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            mode=mode
        )
    elif mode in ["image_only", "sft"]:
        # EMUImageOnlyTokenizer and EMUSftTokenizer don't use mode parameter
        return tokenizer_class(
            text_tokenizer_path=tokenizer_path,
            device=device,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
    else:
        raise ValueError(f"Unknown mode: {mode}. Supported modes: image_only, image2text, text2image, sft")
    

if __name__ == "__main__":

    ### BUILD OMNI TOKENIZER ###

    # Config
    TEXT_TOKENIZER_PATH = "llava-hf/llava-1.5-7b-hf"
    OUTPUT_PATH = "/users/alllau/scratch/ApertusProject/my_omni_tokenizer"
    VISION_TOKENIZER_PATH = 'BAAI/Emu3-VisionTokenizer'
    VISION_TOKENIZER = 'Emu3'
    MODE = 'sft'
    MAX_SAMPLES_TO_TOKENIZE = 5
    
    build_tok = True

    # Build tokenizer
    if build_tok:
        build_tokenizer(
            TEXT_TOKENIZER_PATH,
            OUTPUT_PATH,
            VISION_TOKENIZER_PATH,
            VISION_TOKENIZER
        )
        print("TOKENIZER BUILD FINISHED")


    ### TOKENIZE DATASET ###

    tokenize_dataset = False
    # Load dataset
    if tokenize_dataset:
        dataset = load_coco_dataset(100)
        dataset = get_image_caption_pairs(init_dataset=dataset, num_pairs=MAX_SAMPLES_TO_TOKENIZE)

        # Load omni-tokenizer
        tokenizer = create_tokenizer(
            mode = MODE,
            tokenizer_path = OUTPUT_PATH
        )

        print("TOKENIZER CREATION FINSIHED")
        print("type:", type(tokenizer))
        print("mode:", MODE)


        #### TOKENIZE DATA
        for i in range(len(dataset)):
            sample = dataset[i]
            print(f"\n --- SAMPLE {i} ---")

            text = sample["caption"]
            image = sample["image"]
            print(" > Text:", text)
            print(" > Image:", image)

            combined_tokens = tokenizer.tokenize(image=image, text=text)
            print(f" >> Combined tokens [shape={combined_tokens.shape}]:", combined_tokens)

    


"""
python tokenize.py hf \
--mode image_only \
--dataset-name laion/laion-high-resolution \
--dataset-split train[:10000] \
--tokenizer-path /users/fbrulisauer/scratch/ApertusProject/lsaie_tokenizer_project/my_omni_tokenizer \
--output-dir /users/fbrulisauer/scratch/ApertusProject/lsaie_tokenizer_project/my_tokenized_data \
--num-gpus 1 \
--num-shards 100 \
--device cuda \
--min-tokenizer-pixels "512*512" \
--max-tokenizer-pixels "1024*1024" \
--min-image-pixels "256*256" \
--max-image-pixels "2048*2048"
"""