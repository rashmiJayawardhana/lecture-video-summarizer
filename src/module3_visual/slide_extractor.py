"""
Slide Extraction using OpenCV for Module 3

Extracts unique slide frames from lecture videos using SSIM-based
frame difference detection.

Owner: Fazly (214008C)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim


class SlideExtractor:
    """
    Extract unique slide frames from lecture videos.
    
    Uses SSIM (Structural Similarity Index) to detect frame changes
    and identify when a new slide appears on screen.
    
    Usage:
        extractor = SlideExtractor(similarity_threshold=0.95)
        slides = extractor.extract_slides('lecture_001.mp4')
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        sample_rate: int = 30,
        min_slide_duration: float = 2.0,
    ):
        """
        Args:
            similarity_threshold: SSIM threshold. Frames with SSIM below this
                                  vs. previous slide are considered new slides.
            sample_rate: Check every N frames (30 = 1 per second at 30fps).
            min_slide_duration: Minimum seconds a slide must be shown to count.
        """
        self.similarity_threshold = similarity_threshold
        self.sample_rate = sample_rate
        self.min_slide_duration = min_slide_duration
    
    def extract_slides(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
    ) -> List[dict]:
        """
        Extract unique slide frames from a video.
        
        Args:
            video_path: Path to the input video file.
            output_dir: Optional directory to save extracted slide images.
        
        Returns:
            List of dicts with keys: 'frame_time', 'frame_idx', 'image_path'.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        slides = []
        prev_frame_gray = None
        frame_idx = 0
        
        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240))
                
                is_new_slide = False
                if prev_frame_gray is None:
                    is_new_slide = True
                else:
                    score = ssim(prev_frame_gray, gray)
                    if score < self.similarity_threshold:
                        is_new_slide = True
                
                if is_new_slide:
                    timestamp = frame_idx / fps
                    slide_info = {
                        "frame_time": round(timestamp, 2),
                        "frame_idx": frame_idx,
                    }
                    
                    if output_dir:
                        img_name = f"slide_{len(slides):04d}_{timestamp:.1f}s.png"
                        img_path = str(out_path / img_name)
                        cv2.imwrite(img_path, frame)
                        slide_info["image_path"] = img_path
                    
                    slides.append(slide_info)
                    prev_frame_gray = gray
            
            frame_idx += 1
        
        cap.release()
        
        # Filter by minimum duration
        filtered = self._filter_by_duration(slides, fps)
        
        print(f"Extracted {len(filtered)} unique slides from {video_path}")
        return filtered
    
    def _filter_by_duration(
        self,
        slides: List[dict],
        fps: float,
    ) -> List[dict]:
        """Remove slides shown for less than min_slide_duration."""
        if len(slides) <= 1:
            return slides
        
        filtered = []
        for i in range(len(slides)):
            if i < len(slides) - 1:
                duration = slides[i + 1]["frame_time"] - slides[i]["frame_time"]
            else:
                duration = self.min_slide_duration  # Keep last slide
            
            if duration >= self.min_slide_duration:
                filtered.append(slides[i])
        
        return filtered


if __name__ == "__main__":
    extractor = SlideExtractor(similarity_threshold=0.95)
    print("SlideExtractor ready.")
    print("Usage: slides = extractor.extract_slides('lecture.mp4', 'output/slides/')")
