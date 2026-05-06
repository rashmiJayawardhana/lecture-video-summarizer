"""
Whisper Transcription Integration for Module 2

Transcribes lecture audio using OpenAI Whisper (large-v3 preferred)
and produces word-level timestamped transcripts.

Owner: Ravindu (214095L)
Reference: Radford et al. 2023
"""

import json
import whisper
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional


class LectureTranscriber:
    """
    Transcribe lecture videos using OpenAI Whisper.
    
    Produces word-level timestamps for every spoken sentence,
    which Module 4 uses for subtitle alignment.
    
    Usage:
        transcriber = LectureTranscriber(model_size='large-v3')
        result = transcriber.transcribe('lecture_001.mp4')
        transcriber.save_transcript(result, 'transcript_001.json')
    """
    
    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        """
        Args:
            model_size: Whisper model size. Options: tiny, base, small, medium,
                        large, large-v2, large-v3. Default: large-v3 (preferred).
            device: Device to run on ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper {model_size} on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        self.model_size = model_size
    
    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        word_timestamps: bool = True,
    ) -> Dict[str, Any]:
        """
        Transcribe an audio/video file.
        
        Args:
            audio_path: Path to audio or video file.
            language: Language code (default: 'en').
            word_timestamps: Whether to include word-level timestamps.
        
        Returns:
            Dictionary with 'text', 'segments', and 'language' keys.
            Each segment has 'start', 'end', 'text', and optionally 'words'.
        """
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            verbose=False,
        )
        return result
    
    def extract_sentences(
        self,
        transcription: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract sentence-level data from Whisper transcription.
        
        Args:
            transcription: Output from self.transcribe().
        
        Returns:
            List of sentence dicts with keys:
                - sentence: str
                - timestamp_start: float
                - timestamp_end: float
        """
        sentences = []
        for segment in transcription.get("segments", []):
            sentences.append({
                "sentence": segment["text"].strip(),
                "timestamp_start": round(segment["start"], 2),
                "timestamp_end": round(segment["end"], 2),
            })
        return sentences
    
    def save_transcript(
        self,
        transcription: Dict[str, Any],
        output_path: str,
    ) -> None:
        """
        Save transcription to a JSON file.
        
        Args:
            transcription: Output from self.transcribe().
            output_path: Path to save the JSON file.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract clean data for saving
        save_data = {
            "text": transcription["text"],
            "language": transcription.get("language", "en"),
            "model": self.model_size,
            "segments": [
                {
                    "id": seg["id"],
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip(),
                }
                for seg in transcription.get("segments", [])
            ],
        }
        
        with open(output, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Transcript saved to: {output}")


if __name__ == "__main__":
    print("LectureTranscriber ready.")
    print("Usage: transcriber = LectureTranscriber('large-v3')")
    print("       result = transcriber.transcribe('lecture.mp4')")
