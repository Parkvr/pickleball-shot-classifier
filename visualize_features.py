"""
Visualize a human pose sequence extracted by extract_pose_as_features.py.

Usage:
    python visualize_features.py path/to/shot.csv
    python visualize_features.py path/to/shot.csv --gif output.gif
"""

from argparse import ArgumentParser
import numpy as np
import cv2
import imageio
import pandas as pd

HEIGHT = 500
WIDTH = 500

# (indexA, indexB): color key
EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}

COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}

KEYPOINT_DICT = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


def draw_key_point(frame, inst, name):
    """Draw a green keypoint dot."""
    x = inst[f"{name}_x"]
    y = inst[f"{name}_y"]
    cv2.circle(
        frame,
        (int(x * WIDTH), int(y * HEIGHT)),
        radius=4,
        color=(0, 255, 0),
        thickness=-1,
    )


def draw_edge(frame, inst, edge):
    """Draw a colored limb between two keypoints."""
    a, b = edge

    pointA = None
    pointB = None

    # look up names by matching index values
    for k, idx in KEYPOINT_DICT.items():
        if idx == a:
            pointA = k
        if idx == b:
            pointB = k

    if pointA is None or pointB is None:
        return

    cv2.line(
        frame,
        (int(inst[f"{pointA}_x"] * WIDTH), int(inst[f"{pointA}_y"] * HEIGHT)),
        (int(inst[f"{pointB}_x"] * WIDTH), int(inst[f"{pointB}_y"] * HEIGHT)),
        COLORS[EDGES[edge]],
        thickness=2,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize human pose CSV")
    parser.add_argument("shots", nargs="*", help="CSV file(s)")
    parser.add_argument("--gif", type=str, help="Export animation as GIF")
    args = parser.parse_args()

    for shot_path in args.shots:
        df = pd.read_csv(shot_path)

        # remove "shot" label column if it exists
        if "shot" in df.columns:
            df = df.drop(columns=["shot"])

        frames = []
        for i in range(len(df)):
            inst = df.iloc[i]
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # draw all keypoints
            for name in KEYPOINT_DICT.keys():
                draw_key_point(frame, inst, name)

            # draw limbs
            for edge in EDGES.keys():
                draw_edge(frame, inst, edge)

            # display filename on screen
            cv2.putText(
                frame,
                shot_path,
                (10, HEIGHT - 10),
                fontScale=0.6,
                color=(255, 255, 255),
                thickness=1,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            )

            frames.append(frame)

            cv2.imshow("Pose Visualization", frame)
            k = cv2.waitKey(40)  # play animation automatically (40ms per frame)

            if k == 27:  # ESC
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()

        if args.gif:
            print(f"Saving GIF to {args.gif} ...")
            imageio.mimsave(args.gif, frames, fps=30)
            print("GIF saved.")
