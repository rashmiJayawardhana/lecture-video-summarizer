"""
Pipeline A — Highlight Video Generator (summarized_video.mp4)

Produces a highlight video from real lecture footage: extracts the
top-scoring segments selected by ScoreFusion and concatenates them
(original footage + original audio, hard cuts) into a single condensed
MP4 using MoviePy/FFmpeg, capped at a target duration.
"""

from pathlib import Path
from typing import List, Dict

try:
    from moviepy import VideoFileClip, concatenate_videoclips
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not installed. Run: pip install moviepy")


class HighlightVideoGenerator:
    """
    Generate a highlight video (Pipeline A) from selected segments.

    Usage:
        generator = HighlightVideoGenerator(max_duration=300)
        generator.generate(
            video_path='lecture.mp4',
            segments=selected_segments,
            output_path='highlight.mp4'
        )
    """

    def __init__(
        self,
        max_duration: float = 300.0,
        fps: int = 24,
        codec: str = "libx264",
    ):
        self.max_duration = max_duration
        self.fps = fps
        self.codec = codec

    def generate(
        self,
        video_path: str,
        segments: List[Dict],
        output_path: str,
    ) -> str:
        """
        Generate the highlight video.

        Args:
            video_path: Path to the original lecture video.
            segments: List of selected segment dicts (from ScoreFusion),
                each with at least timestamp_start/timestamp_end. Assumed
                already sorted chronologically by the caller.
            output_path: Path to save the output video.

        Returns:
            Path to the generated video.
        """
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy is required for Pipeline A.")

        if not segments:
            raise ValueError("No segments were selected for Pipeline A - nothing to render.")

        source = VideoFileClip(str(video_path))
        clips = []

        try:
            for seg in segments:
                start = max(0.0, float(seg["timestamp_start"]))
                end = min(source.duration, float(seg["timestamp_end"]))
                if end <= start:
                    continue
                clips.append(source.subclipped(start, end))

            if not clips:
                raise ValueError("All selected segments fell outside the source video's duration.")

            final = concatenate_videoclips(clips, method="compose")

            if final.duration > self.max_duration:
                final = final.subclipped(0, self.max_duration)

            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)

            final.write_videofile(
                str(output),
                fps=self.fps,
                codec=self.codec,
                audio_codec="aac",
                logger=None,
            )

            print(f"Pipeline A output: {output} ({final.duration:.1f}s)")
            return str(output)
        finally:
            source.close()


if __name__ == "__main__":
    print("HighlightVideoGenerator (Pipeline A) ready.")
    print("Requires: MoviePy, FFmpeg")
