"""
Extract human pose keypoints from tennis match images using MediaPipe.
Draws a region of interest (ROI) around the player and extracts joint positions,
exporting features to a CSV file.
"""

from argparse import ArgumentParser
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class PoseExtractor:
    """Extract and track human pose in tennis images using MediaPipe."""

    # Joint names from MediaPipe Pose (33 landmarks)
    JOINT_NAMES = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    # Indices for key body parts (for ROI calculation)
    SHOULDER_INDICES = [11, 12]  # left and right shoulder
    ANKLE_INDICES = [27, 28]     # left and right ankle

    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize the pose extractor.

        Args:
            min_confidence: Minimum confidence threshold for pose detection.
        """
        self.min_confidence = min_confidence
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_pose(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detect pose in the image.

        Args:
            image: Input image (BGR format from OpenCV).

        Returns:
            Tuple of (landmarks in pixel coordinates, detection confidence) or None if no pose detected.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        # Convert normalized landmarks to pixel coordinates
        h, w, _ = image.shape
        landmarks_pixels = []
        overall_confidence = 0

        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z
            confidence = landmark.visibility
            landmarks_pixels.append([x, y, z, confidence])
            overall_confidence += confidence

        overall_confidence /= len(results.pose_landmarks.landmark)
        landmarks_array = np.array(landmarks_pixels)

        return landmarks_array, overall_confidence

    def draw_pose(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw pose skeleton on the image.

        Args:
            image: Input image.
            landmarks: Landmarks in pixel coordinates [x, y, z, confidence].

        Returns:
            Image with pose drawn on it.
        """
        h, w, _ = image.shape

        # Normalize landmarks for drawing
        normalized_landmarks = self.mp_pose.PoseLandmark
        pose_landmarks = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()

        for landmark in landmarks:
            norm_landmark = pose_landmarks.landmark.add()
            norm_landmark.x = landmark[0] / w
            norm_landmark.y = landmark[1] / h
            norm_landmark.z = landmark[2]
            norm_landmark.visibility = landmark[3]

        self.mp_drawing.draw_landmarks(
            image,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

        return image

    def calculate_roi(self, landmarks: np.ndarray, image_shape: Tuple[int, int],
                     padding: float = 0.2) -> Tuple[int, int, int, int]:
        """
        Calculate region of interest bounding box around the player.

        Args:
            landmarks: Landmarks in pixel coordinates.
            image_shape: Shape of the image (height, width, channels).
            padding: Padding factor to expand the bounding box.

        Returns:
            Bounding box as (x1, y1, x2, y2).
        """
        h, w, _ = image_shape

        # Filter landmarks by confidence
        confident_landmarks = landmarks[landmarks[:, 3] >= self.min_confidence]

        if len(confident_landmarks) == 0:
            # Fallback to full image
            return 0, 0, w, h

        x_coords = confident_landmarks[:, 0]
        y_coords = confident_landmarks[:, 1]

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Add padding
        width = x_max - x_min
        height = y_max - y_min

        x_min = max(0, int(x_min - width * padding))
        x_max = min(w, int(x_max + width * padding))
        y_min = max(0, int(y_min - height * padding))
        y_max = min(h, int(y_max + height * padding))

        return x_min, y_min, x_max, y_max

    def draw_roi(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Draw ROI rectangle on the image.

        Args:
            image: Input image.
            roi: Bounding box as (x1, y1, x2, y2).

        Returns:
            Image with ROI drawn.
        """
        x1, y1, x2, y2 = roi
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        return image

    def landmarks_to_csv_row(self, landmarks: np.ndarray, frame_id: int = 0) -> dict:
        """
        Convert landmarks to a dictionary suitable for CSV export.

        Args:
            landmarks: Landmarks in pixel coordinates.
            frame_id: Frame identifier.

        Returns:
            Dictionary with frame_id and joint features.
        """
        row = {"frame_id": frame_id}

        for i, joint_name in enumerate(self.JOINT_NAMES):
            if i < len(landmarks):
                x, y, z, confidence = landmarks[i]
                row[f"{joint_name}_x"] = x
                row[f"{joint_name}_y"] = y
                row[f"{joint_name}_z"] = z
                row[f"{joint_name}_confidence"] = confidence

        return row

    def close(self):
        """Release resources."""
        self.pose.close()


def process_image(image_path: str, output_csv: str = None, display: bool = True) -> Optional[pd.DataFrame]:
    """
    Process a single image and extract pose.

    Args:
        image_path: Path to the input image.
        output_csv: Path to save the CSV file (optional).
        display: Whether to display the image with pose and ROI.

    Returns:
        DataFrame with pose data, or None if no pose detected.
    """
    extractor = PoseExtractor()

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    print(f"Processing image: {image_path}")
    print(f"Image shape: {image.shape}")

    # Detect pose
    result = extractor.detect_pose(image)
    if result is None:
        print("No pose detected in the image.")
        extractor.close()
        return None

    landmarks, confidence = result
    print(f"Pose detected with confidence: {confidence:.2f}")

    # Calculate and draw ROI
    roi = extractor.calculate_roi(landmarks, image.shape)
    image_with_roi = extractor.draw_roi(image.copy(), roi)

    # Draw pose
    image_with_pose = extractor.draw_pose(image_with_roi, landmarks)

    # Create DataFrame
    row = extractor.landmarks_to_csv_row(landmarks, frame_id=0)
    df = pd.DataFrame([row])

    # Save CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Pose data saved to: {output_csv}")

    # Display image
    if display:
        cv2.imshow("Pose Detection with ROI", image_with_pose)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    extractor.close()
    return df


def process_video(video_path: str, output_csv: str = None, display: bool = False,
                  skip_frames: int = 1) -> Optional[pd.DataFrame]:
    """
    Process a video and extract poses from each frame.

    Args:
        video_path: Path to the input video.
        output_csv: Path to save the CSV file (optional).
        display: Whether to display frames with pose and ROI.
        skip_frames: Process every nth frame (1 = process all frames).

    Returns:
        DataFrame with pose data from all frames, or None if no poses detected.
    """
    extractor = PoseExtractor()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}, FPS: {fps:.2f}")

    all_data = []
    frame_id = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if requested
        if frame_id % skip_frames != 0:
            frame_id += 1
            continue

        # Detect pose
        result = extractor.detect_pose(frame)
        if result is not None:
            landmarks, confidence = result

            # Calculate ROI
            roi = extractor.calculate_roi(landmarks, frame.shape)

            # Prepare data
            row = extractor.landmarks_to_csv_row(landmarks, frame_id=frame_id)
            row["confidence"] = confidence
            all_data.append(row)

            processed_frames += 1

            # Display if requested
            if display:
                frame_display = extractor.draw_roi(frame.copy(), roi)
                frame_display = extractor.draw_pose(frame_display, landmarks)
                cv2.imshow("Pose Detection with ROI", frame_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_id += 1

        if frame_id % 30 == 0:
            print(f"Processed {frame_id} frames ({processed_frames} with valid poses)...")

    cap.release()
    if display:
        cv2.destroyAllWindows()

    if not all_data:
        print("No poses detected in the video.")
        extractor.close()
        return None

    df = pd.DataFrame(all_data)

    # Save CSV if requested
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Pose data saved to: {output_csv}")
        print(f"Total frames with valid poses: {len(df)}")

    extractor.close()
    return df


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Extract human pose from tennis match images or videos using MediaPipe"
    )
    parser.add_argument("input", help="Path to input image or video")
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path (default: input_pose.csv)",
        default=None
    )
    parser.add_argument(
        "-d", "--display",
        help="Display frames with pose and ROI",
        action="store_true"
    )
    parser.add_argument(
        "-s", "--skip-frames",
        help="Process every nth frame (default: 1)",
        type=int,
        default=1
    )
    parser.add_argument(
        "-c", "--confidence",
        help="Minimum confidence threshold for pose detection (default: 0.5)",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    # Determine output CSV path
    if args.output:
        output_csv = args.output
    else:
        output_csv = str(input_path.parent / f"{input_path.stem}_pose.csv")

    # Process based on file type
    if input_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        df = process_image(str(input_path), output_csv, display=args.display)
    elif input_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        df = process_video(
            str(input_path),
            output_csv,
            display=args.display,
            skip_frames=args.skip_frames
        )
    else:
        print(f"Error: Unsupported file format: {input_path.suffix}")
        exit(1)

    if df is not None:
        print(f"\nExtraction complete!")
        print(f"Shape of data: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
    else:
        print("Failed to extract pose data.")
