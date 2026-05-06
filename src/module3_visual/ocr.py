"""
TrOCR Integration for Module 3

Extracts on-screen text from Critical and Important slide frames
using TrOCR (Transformer-based OCR).

Owner: Fazly (214008C)
Reference: Li et al. 2023

NOTE: This module uses TrOCR (microsoft/trocr-base-printed), NOT Tesseract.
TrOCR was validated for OCR tasks in the literature review.
"""

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from typing import List, Optional
from pathlib import Path


class SlideOCR:
    """
    Extract text from slide images using TrOCR.
    
    Run on Critical and Important frames only (Skip frames excluded).
    
    Usage:
        ocr = SlideOCR()
        text = ocr.extract_text('slide_001.png')
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        device: str = "cuda",
    ):
        """
        Args:
            model_name: TrOCR model from Hugging Face.
            device: Device for inference ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Loading TrOCR model: {model_name}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from a single slide image.
        
        Args:
            image_path: Path to the slide image.
        
        Returns:
            Extracted text string.
        """
        image = Image.open(image_path).convert("RGB")
        return self._process_image(image)
    
    def extract_text_from_pil(self, image: Image.Image) -> str:
        """
        Extract text from a PIL Image object.
        
        Args:
            image: PIL Image in RGB mode.
        
        Returns:
            Extracted text string.
        """
        return self._process_image(image.convert("RGB"))
    
    def _process_image(self, image: Image.Image) -> str:
        """Internal: run TrOCR on a PIL image."""
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=200)
        
        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return text.strip()
    
    def batch_extract(
        self,
        image_paths: List[str],
        skip_labels: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Extract text from multiple slide images.
        
        Args:
            image_paths: List of image file paths.
            skip_labels: Optional list of labels to skip (e.g., 'Skip' frames).
        
        Returns:
            List of dicts with 'image_path' and 'ocr_text'.
        """
        results = []
        for path in image_paths:
            text = self.extract_text(path)
            results.append({
                "image_path": path,
                "ocr_text": text,
            })
        return results


if __name__ == "__main__":
    print("SlideOCR (TrOCR) ready.")
    print("Usage: ocr = SlideOCR()")
    print("       text = ocr.extract_text('slide.png')")
