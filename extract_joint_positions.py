"""
Read a video with opencv and infer movenet to display human pose.
Note that we perform tracking of the tennis player to feed the neural network
with a more specific search area dennotated as RoI (Region of Interest).
If the player is lost, we reset the RoI.
"""

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import cv2


class RoI:
    """
    Define the Region of Interest around the tennis player.
    At each frame, we refine it and use current position to feed the
    movenet neural network.
    """

class RoI:
    """
    Region of Interest around the player with smoothing and hysteresis
    to avoid frequent resets on small/low-confidence detections (pickleball-friendly).
    """

    # Tunables for pickleball singles (player is small/far)
    MIN_SIDE = 80             # was effectively ~150; lower it
    MAX_SIDE_FACTOR = 1.0     # 1.0 * frame smaller side
    ZOOM_MARGIN = 1.35        # inflate bbox (keeps context)
    EMA_ALPHA = 0.35          # smoothing for center/size (0=no smooth, 1=instant)
    MIN_CONF = 0.20           # min confidence for keypoints considered "good"
    MIN_GOOD_KP = 5           # need at least this many confident keypoints
    BAD_FRAMES_BEFORE_RESET = 6  # grace period before hard reset

    def __init__(self, shape):
        self.frame_height = shape[0]
        self.frame_width = shape[1]

        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        side0 = min(self.frame_width, self.frame_height)
        self.width = side0
        self.height = side0

        self._max_side = int(side0 * self.MAX_SIDE_FACTOR)
        self.valid = False
        self._bad_frames = 0  # how many consecutive bad frames

    def _bounds(self):
        x1 = max(0, self.center_x - self.width // 2)
        y1 = max(0, self.center_y - self.height // 2)
        x2 = min(self.frame_width, self.center_x + self.width // 2)
        y2 = min(self.frame_height, self.center_y + self.height // 2)
        return x1, y1, x2, y2

    def extract_subframe(self, frame):
        x1, y1, x2, y2 = self._bounds()
        if x2 <= x1 or y2 <= y1:
            return frame  # fallback (prevents gray frames)
        return frame[y1:y2, x1:x2]

    def transform_to_subframe_coordinates(self, keypoints_from_tf):
        # Multiply Y by height, X by width (TF outputs normalized [y, x, score])
        return np.squeeze(np.multiply(keypoints_from_tf, [self.height, self.width, 1.0]))

    def transform_to_frame_coordinates(self, keypoints_from_tf):
        kps_sub = self.transform_to_subframe_coordinates(keypoints_from_tf)
        x1, y1, x2, y2 = self._bounds()
        kps_frame = kps_sub.copy()
        kps_frame[:, 0] += y1  # y
        kps_frame[:, 1] += x1  # x
        return kps_frame

    def _ema(self, old, new):
        return int(round(self.EMA_ALPHA * new + (1 - self.EMA_ALPHA) * old))

    def update(self, keypoints_pixels):
        """Update RoI using robust rules: min size, smoothing, and grace period."""
        if keypoints_pixels is None or keypoints_pixels.ndim != 2:
            self._note_bad_and_maybe_reset(reason="no keypoints array")
            return

        # Keep only confident keypoints
        conf_mask = keypoints_pixels[:, 2] >= self.MIN_CONF
        good_count = int(np.sum(conf_mask))
        if good_count < self.MIN_GOOD_KP:
            self._note_bad_and_maybe_reset(reason=f"too few good kps ({good_count})")
            return

        pts = keypoints_pixels[conf_mask][:, :2]
        min_y, min_x = np.min(pts[:, 0]).astype(int), np.min(pts[:, 1]).astype(int)
        max_y, max_x = np.max(pts[:, 0]).astype(int), np.max(pts[:, 1]).astype(int)

        # Inflate box a bit for context
        w = int((max_x - min_x) * self.ZOOM_MARGIN)
        h = int((max_y - min_y) * self.ZOOM_MARGIN)
        side = max(w, h, self.MIN_SIDE)
        side = min(side, self._max_side)

        cx_new = (min_x + max_x) // 2
        cy_new = (min_y + max_y) // 2

        # Smooth center/size to avoid jitter
        self.center_x = self._ema(self.center_x, cx_new)
        self.center_y = self._ema(self.center_y, cy_new)
        self.width = self._ema(self.width, side)
        self.height = self.width  # keep square

        # Keep in-bounds
        half = self.width // 2
        self.center_x = min(max(self.center_x, half + 1), self.frame_width - half - 1)
        self.center_y = min(max(self.center_y, half + 1), self.frame_height - half - 1)

        # If after clamping the ROI collapses, treat as bad
        x1, y1, x2, y2 = self._bounds()
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            self._note_bad_and_maybe_reset(reason="collapsed after clamp")
            return

        # Good update
        self._bad_frames = 0
        self.valid = True

    def _note_bad_and_maybe_reset(self, reason=""):
        self._bad_frames += 1
        if self._bad_frames >= self.BAD_FRAMES_BEFORE_RESET:
            # print(f"Lost player track --> reset ROI ({reason})")
            self.reset()

    def reset(self):
        side0 = min(self.frame_width, self.frame_height)
        self.width = side0
        self.height = side0
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        self.valid = False
        self._bad_frames = 0

    def draw_shot(self, frame, shot):
        cv2.putText(
            frame, shot,
            (self.center_x - 50, self.center_y - self.height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 255, 255), 2
        )

class HumanPoseExtractor:
    """
    Defines mapping between movenet key points and human readable body points
    with realistic edges to be drawn"""

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

    # Dictionary that maps from joint names to keypoint indices.
    KEYPOINT_DICT = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
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

    def __init__(self, shape):
        # Initialize the TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=r"movenet.tflite\movenet_singlepose_lightning.tflite")
        self.interpreter.allocate_tensors()

        self.roi = RoI(shape)

    def extract(self, frame):
        """Run inference model on subframe"""
        # Reshape image
        subframe = self.roi.extract_subframe(frame)

        img = subframe.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.uint8)
        # input_image = tf.cast(img, dtype=tf.int32)

        # Setup input and output
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Make predictions
        self.interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        self.interpreter.invoke()
        self.keypoints_with_scores = self.interpreter.get_tensor(
            output_details[0]["index"]
        )
        self.keypoints_pixels_frame = self.roi.transform_to_frame_coordinates(
            self.keypoints_with_scores
        )

    def discard(self, list_of_keypoints):
        """Discard some points like eyes or ears (useless for our application)"""
        for keypoint in list_of_keypoints:
            self.keypoints_with_scores[0, 0, self.KEYPOINT_DICT[keypoint], 2] = 0
            self.keypoints_pixels_frame[self.KEYPOINT_DICT[keypoint], 2] = 0

    def draw_results_subframe(self):
        """Draw key points and eges on subframe (roi)"""
        subframe = self.roi.extract_subframe(frame)
        keypoints_pixels_subframe = self.roi.transform_to_subframe_coordinates(
            self.keypoints_with_scores
        )

        # Rendering
        draw_edges(subframe, keypoints_pixels_subframe, self.EDGES, 0.2)
        draw_keypoints(subframe, keypoints_pixels_subframe, 0.2)

        return subframe

    def draw_results_frame(self, frame):
        """Draw key points and eges on frame"""
        if not self.roi.valid:
            return

        draw_edges(frame, self.keypoints_pixels_frame, self.EDGES, 0.01)
        draw_keypoints(frame, self.keypoints_pixels_frame, 0.01)
        draw_roi(self.roi, frame)


def draw_keypoints(frame, shaped, confidence_threshold):
    """Draw key points with green dots"""
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_edges(frame, shaped, edges, confidence_threshold):
    """Draw edges with cyan for the right side, magenta for the left side, rest in yellow"""
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=HumanPoseExtractor.COLORS[color],
                thickness=2,
            )


def draw_roi(roi, frame):
    """Draw RoI with a yellow square"""
    cv2.line(
        frame,
        (roi.center_x - roi.width // 2, roi.center_y - roi.height // 2),
        (roi.center_x - roi.width // 2, roi.center_y + roi.height // 2),
        (0, 255, 255),
        3,
    )
    cv2.line(
        frame,
        (roi.center_x + roi.width // 2, roi.center_y + roi.height // 2),
        (roi.center_x - roi.width // 2, roi.center_y + roi.height // 2),
        (0, 255, 255),
        3,
    )
    cv2.line(
        frame,
        (roi.center_x + roi.width // 2, roi.center_y + roi.height // 2),
        (roi.center_x + roi.width // 2, roi.center_y - roi.height // 2),
        (0, 255, 255),
        3,
    )
    cv2.line(
        frame,
        (roi.center_x - roi.width // 2, roi.center_y - roi.height // 2),
        (roi.center_x + roi.width // 2, roi.center_y - roi.height // 2),
        (0, 255, 255),
        3,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Display human pose on a video")
    parser.add_argument("video")
    parser.add_argument(
        "--debug",
        action="store_const",
        const=True,
        default=False,
        help="Show sub frame (RoI)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    assert cap.isOpened()

    ret, frame = cap.read()

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    FRAME_ID = 0

    while cap.isOpened():
        ret, frame = cap.read()

        FRAME_ID += 1

        human_pose_extractor.extract(frame)

        # dont draw non-significant points/edges by setting probability to 0
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        # Extract subframe (roi) and display results
        if args.debug:
            subframe = human_pose_extractor.draw_results_subframe()
            cv2.imshow("Subframe", subframe)

        # Display results on original frame
        human_pose_extractor.draw_results_frame(frame)
        cv2.imshow("Frame", frame)
        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

        # cv2.imwrite(f"videos/image_{FRAME_ID:05d}.png", frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()