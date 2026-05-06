"""
Pipeline B — Synthetic Slideshow Video Generator (slideshow.mp4)

Produces a fully synthetic narrated slideshow video:
    - Renders slides as PNG images using Pillow (1920x1080)
    - Generates narration script per slide using GPT-4o
    - Synthesizes MP3 audio per slide using edge-tts AriaNeural
    - Holds each slide for max(audio_duration, 3 seconds)
    - Concatenates all slide clips and encodes as H.264 MP4

No raw lecture footage is needed — fully AI-generated output.

Owner: Lathisana (214116F)
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class NarrationGenerator:
    """
    Generate narration scripts using GPT-4o.
    
    Usage:
        narrator = NarrationGenerator(api_key='...')
        script = narrator.generate_script(slide_text='Binary Search Trees...')
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        Args:
            api_key: OpenAI API key.
            model: Model string. Use 'gpt-4o', NOT 'gpt-4.0'.
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Run: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_script(
        self,
        slide_text: str,
        max_tokens: int = 100,
    ) -> str:
        """
        Generate a narration script for a single slide.
        
        Args:
            slide_text: OCR text from the slide.
            max_tokens: Maximum tokens for the narration (cost control).
        
        Returns:
            Narration script string.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a university lecturer narrating an IT theory lecture. "
                        "Given the text from a lecture slide, generate a clear, concise "
                        "narration explaining the content. Keep it educational and engaging."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Narrate this slide content:\n\n{slide_text}",
                },
            ],
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content.strip()
    
    def batch_generate(
        self,
        slides: List[Dict],
    ) -> List[Dict]:
        """
        Generate narration scripts for multiple slides.
        
        Args:
            slides: List of dicts with 'ocr_text' key.
        
        Returns:
            List of dicts with added 'narration_script' key.
        """
        results = []
        for slide in slides:
            if slide.get("ocr_text"):
                script = self.generate_script(slide["ocr_text"])
            else:
                script = ""
            
            results.append({
                **slide,
                "narration_script": script,
            })
        
        return results


class TTSSynthesizer:
    """
    Synthesize speech from narration scripts using edge-tts AriaNeural.
    
    Usage:
        tts = TTSSynthesizer()
        await tts.synthesize('Hello world', 'output.mp3')
    """
    
    def __init__(self, voice: str = "en-US-AriaNeural"):
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts required. Run: pip install edge-tts")
        self.voice = voice
    
    async def synthesize(self, text: str, output_path: str) -> str:
        """
        Synthesize speech and save as MP3.
        
        Args:
            text: Text to synthesize.
            output_path: Path to save the MP3 file.
        
        Returns:
            Path to the saved MP3.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output))
        
        return str(output)
    
    def synthesize_sync(self, text: str, output_path: str) -> str:
        """Synchronous wrapper for synthesize()."""
        return asyncio.run(self.synthesize(text, output_path))


class SlideshowVideoGenerator:
    """
    Generate a synthetic slideshow video (Pipeline B).
    
    Usage:
        generator = SlideshowVideoGenerator(api_key='...')
        generator.generate(slides=critical_slides, output_path='slideshow.mp4')
    """
    
    def __init__(
        self,
        api_key: str = "",
        voice: str = "en-US-AriaNeural",
        min_slide_duration: float = 3.0,
    ):
        self.narrator = NarrationGenerator(api_key) if api_key else None
        self.tts = TTSSynthesizer(voice)
        self.min_slide_duration = min_slide_duration
    
    def generate(
        self,
        slides: List[Dict],
        output_path: str,
    ) -> str:
        """
        Generate the slideshow video.
        
        Args:
            slides: List of slide dicts with 'image_path' and 'ocr_text'.
            output_path: Path to save the output video.
        
        Returns:
            Path to the generated video.
        """
        # TODO: Implement full pipeline
        # 1. Generate narration scripts (GPT-4o)
        # 2. Synthesize audio (edge-tts)
        # 3. Create ImageClip per slide, hold for max(audio_duration, 3s)
        # 4. Concatenate and encode as H.264 MP4
        raise NotImplementedError(
            "Pipeline B implementation in progress. "
            "Target completion: Week 6-7."
        )


if __name__ == "__main__":
    print("SlideshowVideoGenerator (Pipeline B) ready.")
    print("Requires: openai, edge-tts, MoviePy, Pillow")
