"""
Windows-optimized script for shot annotation on a pickleball/tennis video.
It will output a csv file containing frame id and shot name by pressing keyboard keys:
- 'F' key to mark a shot as FOREHAND
- 'B' key to mark a shot as BACKHAND
- 'S' key to mark a shot as SERVE
- Arrow keys also work (LEFT=backhand, UP=serve, RIGHT=forehand)
Press ESC to exit.

We advise you to hit the key when the player hits the ball.
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2

# Letter keys (most reliable on Windows)
KEY_FOREHAND = ord('f')
KEY_BACKHAND = ord('b')
KEY_SERVE = ord('s')

# Arrow keys for Windows (extended key codes)
# These may vary by OpenCV version, so letter keys are preferred
LEFT_ARROW_KEY = 2424832   # Windows left arrow
UP_ARROW_KEY = 2490368     # Windows up arrow
RIGHT_ARROW_KEY = 2555904  # Windows right arrow


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Annotate a video and write a csv file containing tennis shots (Windows version)"
    )
    parser.add_argument("video")
    parser.add_argument(
        "--use-arrows",
        action="store_true",
        help="Use arrow keys instead of letter keys (F/B/S)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.video}'")
        print("Please check that the file exists and is a valid video file.")
        exit(1)

    FRAME_ID = 0
    annotations = []

    print("=" * 60)
    print("Windows Video Annotator")
    print("=" * 60)
    if args.use_arrows:
        print("Controls (Arrow Keys):")
        print("  RIGHT ARROW -> Forehand")
        print("  LEFT ARROW  -> Backhand")
        print("  UP ARROW    -> Serve")
    else:
        print("Controls (Letter Keys - Recommended):")
        print("  F -> Forehand")
        print("  B -> Backhand")
        print("  S -> Serve")
    print("  ESC -> Exit")
    print("=" * 60)
    print(f"Annotating: {Path(args.video).name}")
    print("Press the keys when the player hits the ball.\n")

    # Read until video is completed
    try:
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Frame - Press keys to annotate", frame)
            
            # Get key press (wait 30ms for smooth playback)
            k = cv2.waitKey(30)
            
            # Check for letter keys (case-insensitive)
            k_lower = k & 0xFF  # Get lower byte for ASCII keys
            
            if args.use_arrows:
                # Arrow key mode (with letter key fallback)
                if k == RIGHT_ARROW_KEY or k_lower == ord('f'):  # forehand
                    annotations.append({"Shot": "forehand", "FrameId": FRAME_ID})
                    print(f"Frame {FRAME_ID}: Added FOREHAND")
                elif k == LEFT_ARROW_KEY or k_lower == ord('b'):  # backhand
                    annotations.append({"Shot": "backhand", "FrameId": FRAME_ID})
                    print(f"Frame {FRAME_ID}: Added BACKHAND")
                elif k == UP_ARROW_KEY or k_lower == ord('s'):  # serve
                    annotations.append({"Shot": "serve", "FrameId": FRAME_ID})
                    print(f"Frame {FRAME_ID}: Added SERVE")
            else:
                # Letter key mode (default, more reliable)
                # Convert to lowercase for case-insensitive matching
                key_char = chr(k_lower).lower() if 32 <= k_lower <= 126 else None
                if key_char == 'f':  # 'f' or 'F'
                    annotations.append({"Shot": "forehand", "FrameId": FRAME_ID})
                    print(f"Frame {FRAME_ID}: Added FOREHAND")
                elif key_char == 'b':  # 'b' or 'B'
                    annotations.append({"Shot": "backhand", "FrameId": FRAME_ID})
                    print(f"Frame {FRAME_ID}: Added BACKHAND")
                elif key_char == 's':  # 's' or 'S'
                    annotations.append({"Shot": "serve", "FrameId": FRAME_ID})
                    print(f"Frame {FRAME_ID}: Added SERVE")

            # Press ESC on keyboard to exit
            if k == 27 or k_lower == 27:
                print("\nExiting annotation...")
                break

            FRAME_ID += 1

    except KeyboardInterrupt:
        print("\n\nAnnotation interrupted by user.")
    except Exception as e:
        print(f"\nError during annotation: {e}")
    finally:
        # Create DataFrame and save
        if annotations:
            df = pd.DataFrame(annotations)
            out_file = f"annotation_{Path(args.video).stem}.csv"
            df.to_csv(out_file, index=False)
            print(f"\n{'=' * 60}")
            print(f"Annotation complete!")
            print(f"Total annotations: {len(annotations)}")
            print(f"Output file: {out_file}")
            print(f"{'=' * 60}")
        else:
            print("\nNo annotations were recorded.")

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

