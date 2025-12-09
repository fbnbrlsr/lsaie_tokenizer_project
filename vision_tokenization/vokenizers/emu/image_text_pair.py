#!/usr/bin/env python3
"""
EMU tokenizer for image-text pairs with parallel GPU/CPU processing.
"""

import torch
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from .image_only import EMUImageOnlyTokenizer


class EMUImageTextPairTokenizer(EMUImageOnlyTokenizer):
    """
    Extended tokenizer for image-text pairs with parallel GPU/CPU processing.
    Image tokenization happens on GPU while text tokenization happens on CPU in parallel.
    """

    def __init__(self, *args, mode=None, **kwargs):
        """Initialize with same parameters as parent class."""
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="TokenizerPool")

    def tokenize_image_text_pair(
        self,
        image,
        text: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize an image-text pair with parallel processing using ThreadPoolExecutor.
        Image is processed on GPU while text is processed on CPU simultaneously.

        Args:
            image: PIL Image to tokenize
            text: Text string to append after image

        Returns:
            Dictionary with keys:
            - "text": Text tokens (no special tokens)
            - "image": Image tokens with all special tokens (BOS, img_start, img_end, EOS)
            - "metadata": Dict with mode and token counts
        """
        def tokenize_text_cpu():
            """CPU thread for text tokenization."""
            # Force text tokenization to CPU
            with torch.cuda.device(-1):  # Use CPU
                text_tokens_dict = self.text_tokenizer(
                    text,
                    truncation=False,
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                return text_tokens_dict['input_ids'].squeeze(0)

        # Submit both tasks to executor
        # Image on GPU (usually the bottleneck)
        image_future = self.executor.submit(self.tokenize_image, image)

        # Text on CPU (fast, runs in parallel)
        text_future = self.executor.submit(tokenize_text_cpu)

        # Wait for both and get results
        image_tokens = image_future.result()
        text_tokens = text_future.result()

        # Move text tokens to same device as image tokens
        text_tokens = text_tokens.to(image_tokens.device)

        # Return separated tokens as dictionary
        # Image tokens contain full structure: BOS, img_start, img_end, EOS
        return {
            "text": text_tokens,
            "image": image_tokens,
            "metadata": {
                "mode": self.mode,
                "image_token_count": len(image_tokens),
                "text_token_count": len(text_tokens)
            }
        }

    def tokenize(self, image, text) -> Dict[str, torch.Tensor]:
        """
        Unified tokenization interface for image-text pair mode.

        Args:
            image: PIL Image to tokenize (required)
            text: Text string to append after image (required)

        Returns:
            Dictionary with separated text and image tokens
        """
        # Both image and text are required for image-text pair tokenizer
        return self.tokenize_image_text_pair(image, text)