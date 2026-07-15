import os
import json
import argparse


def select_visual_anchors(input_json, output_json, min_gap=30, max_records=80):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Keep only Critical and Important frames
    candidates = [
        item for item in data
        if item.get("label") in ["Critical", "Important"]
    ]

    # Sort by time
    candidates = sorted(candidates, key=lambda x: x["frame_time"])

    selected = []
    last_time = -999999

    for item in candidates:
        current_time = item["frame_time"]

        if current_time - last_time >= min_gap:
            selected.append(item)
            last_time = current_time

        if len(selected) >= max_records:
            break

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=4)

    print(f"Input records: {len(data)}")
    print(f"Candidate Critical/Important records: {len(candidates)}")
    print(f"Selected visual anchors: {len(selected)}")
    print(f"Saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_gap", type=int, default=30)
    parser.add_argument("--max_records", type=int, default=80)

    args = parser.parse_args()

    select_visual_anchors(
        input_json=args.input_json,
        output_json=args.output_json,
        min_gap=args.min_gap,
        max_records=args.max_records
    )
