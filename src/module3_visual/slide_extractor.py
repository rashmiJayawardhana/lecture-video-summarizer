import cv2
import os
import argparse
import json


def seconds_to_time(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}m{secs:02d}s"


def get_lecture_id(video_path):
    """
    Example:
    data/raw/lecture_001.mp4 -> lecture_001
    data/raw/lecture_002.mp4 -> lecture_002
    """
    filename = os.path.basename(video_path)
    lecture_id = os.path.splitext(filename)[0]
    return lecture_id


def extract_frames(video_path, output_dir, interval_seconds=5):
    os.makedirs(output_dir, exist_ok=True)

    lecture_id = get_lecture_id(video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise ValueError("Could not read FPS from video.")

    frame_interval = int(fps * interval_seconds)

    frame_count = 0
    saved_count = 0
    metadata = []

    while True:
        if frame_count % frame_interval == 0:
            # This is a frame we actually want: grab() + retrieve() decodes it,
            # same as read() would.
            ret, frame = cap.read()
            if not ret:
                break

            timestamp_seconds = frame_count / fps
            time_text = seconds_to_time(timestamp_seconds)

            filename = f"{lecture_id}_frame_{saved_count:05d}_{time_text}.jpg"
            output_path = os.path.join(output_dir, filename)

            cv2.imwrite(output_path, frame)

            metadata.append({
                "lecture_id": lecture_id,
                "frame_path": output_path,
                "frame_time": round(timestamp_seconds, 2)
            })

            saved_count += 1
        else:
            # Frames we're skipping don't need to be decoded at all - grab()
            # advances the reader without doing the (expensive) decode step
            # that read()/retrieve() does. Same frames get saved, same pixel
            # data, this just skips decoding frames we were throwing away.
            ret = cap.grab()
            if not ret:
                break

        frame_count += 1

    cap.release()

    metadata_path = os.path.join(output_dir, f"{lecture_id}_frames_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Frame extraction completed.")
    print(f"Lecture ID: {lecture_id}")
    print(f"Video path: {video_path}")
    print(f"FPS: {fps}")
    print(f"Total frames in video: {total_frames}")
    print(f"Saved frames: {saved_count}")
    print(f"Output folder: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract slide frames from lecture video.")

    parser.add_argument("--video", required=True, help="Path to lecture video")
    parser.add_argument("--output", required=True, help="Output directory for extracted frames")
    parser.add_argument("--interval", type=int, default=30, help="Frame extraction interval in seconds")

    args = parser.parse_args()

    extract_frames(
        video_path=args.video,
        output_dir=args.output,
        interval_seconds=args.interval
    )

