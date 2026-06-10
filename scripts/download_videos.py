"""
Download sample lecture videos from sources
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess


# Sample lecture video URLs (MIT OCW and other open sources)
SAMPLE_VIDEOS = [
    {
        "title": "MIT 6.006 Introduction to Algorithms - Lecture 1",
        "url": "https://www.youtube.com/watch?v=HtSuA80QTyo",
        "duration": "47:56",
        "subject": "Computer Science"
    },
    {
        "title": "MIT 18.01 Single Variable Calculus - Lecture 1",
        "url": "https://www.youtube.com/watch?v=7K1sB05pE0A",
        "duration": "49:36",
        "subject": "Mathematics"
    },
    {
        "title": "MIT 8.01 Physics I - Lecture 1",
        "url": "https://www.youtube.com/watch?v=wWnfJ0-xXRE",
        "duration": "49:11",
        "subject": "Physics"
    },
    # Add more videos as needed
]


def check_yt_dlp():
    """Check if yt-dlp is installed"""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_yt_dlp():
    """Install yt-dlp"""
    print("Installing yt-dlp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)
    print("yt-dlp installed successfully!")


def download_video(url, output_dir, video_id):
    """Download a single video"""
    output_template = os.path.join(output_dir, f"lecture_{video_id:03d}.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720]",  # Max 720p to save space
        "-o", output_template,
        "--no-playlist",
        url
    ]
    
    print(f"\nDownloading video {video_id}...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Download sample lecture videos")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/raw",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--num-videos",
        type=int,
        default=3,
        help="Number of videos to download"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check/install yt-dlp
    if not check_yt_dlp():
        print("yt-dlp not found. Installing...")
        install_yt_dlp()
    
    # Download videos
    num_to_download = min(args.num_videos, len(SAMPLE_VIDEOS))
    
    print(f"\nDownloading {num_to_download} sample lecture videos...")
    print(f"Output directory: {output_dir.absolute()}\n")
    
    for i, video_info in enumerate(SAMPLE_VIDEOS[:num_to_download], 1):
        print(f"\n{'='*60}")
        print(f"Video {i}/{num_to_download}")
        print(f"Title: {video_info['title']}")
        print(f"Duration: {video_info['duration']}")
        print(f"Subject: {video_info['subject']}")
        print(f"{'='*60}")
        
        try:
            download_video(video_info['url'], str(output_dir), i)
            print(f"✓ Successfully downloaded video {i}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to download video {i}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Download complete! Videos saved to: {output_dir.absolute()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
