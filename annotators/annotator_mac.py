"""
Mac-optimized video annotator for tennis/pickleball shots.
Press keys at the moment the player hits the ball:

F → Forehand
B → Backhand
S → Serve
ESC → Exit

Outputs CSV: annotation_<video_name>.csv
"""

import cv2
import pandas as pd
import time
from pathlib import Path
from argparse import ArgumentParser

KEY_FOREHAND = ord('f')
KEY_BACKHAND = ord('b')
KEY_SERVE = ord('s')
KEY_ESC = 27  # ESC key


if __name__ == "__main__":
    parser = ArgumentParser(description="Mac-optimized video annotation tool")
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()

    video_path = args.video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video '{video_path}'")
        exit(1)

    # ---- Get video FPS for real-time playback ----
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 120:
        fps = 30  # fallback default for sports videos

    frame_delay = 1.0 / fps

    # Adjust delay to compensate for macOS waitKey timing issues
    # This value is tuned for smooth real-time playback
    COMPENSATION = 0.006   # calibrated for M1/M2/M3 Macs
    adjusted_delay = max(0, frame_delay - COMPENSATION)

    frame_id = 0
    annotations = []

    print("\n" + "="*60)
    print("🎾 Mac Video Annotator (Real-Time Calibrated)")
    print("="*60)
    print("Controls:")
    print("  F → Forehand")
    print("  B → Backhand")
    print("  S → Serve")
    print("  ESC → Exit")
    print("="*60)
    print(f"Annotating: {Path(video_path).name}")
    print("Press keys when the player hits the ball.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Annotator (Mac)", frame)

            # --- Correct, smooth real-time playback for macOS ---
            time.sleep(adjusted_delay)   # enforce frame timing
            k = cv2.waitKey(1) & 0xFF    # capture key without delaying playback

            if k == KEY_FOREHAND:
                annotations.append({"Shot": "forehand", "FrameId": frame_id})
                print(f"Frame {frame_id}: FOREHAND")

            elif k == KEY_BACKHAND:
                annotations.append({"Shot": "backhand", "FrameId": frame_id})
                print(f"Frame {frame_id}: BACKHAND")

            elif k == KEY_SERVE:
                annotations.append({"Shot": "serve", "FrameId": frame_id})
                print(f"Frame {frame_id}: SERVE")

            elif k == KEY_ESC:
                print("Exiting…")
                break

            frame_id += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if annotations:
            df = pd.DataFrame(annotations)
            out_name = f"annotation_{Path(video_path).stem}.csv"
            df.to_csv(out_name, index=False)
            print("\n" + "="*60)
            print("✅ Annotation complete!")
            print(f"📝 Saved: {out_name}")
            print(f"📊 Total annotated events: {len(annotations)}")
            print("="*60)
        else:
            print("\n⚠ No annotations recorded.")
