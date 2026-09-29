"""ChArUco board detection and pose estimation (OpenCV 3.2 aruco API)."""

import cv2
import numpy as np


DICTIONARIES = {name: getattr(cv2.aruco, name) for name in dir(cv2.aruco) if name.startswith("DICT_")}


class Detection(object):
    """ChArUco corners found in one image."""

    def __init__(self, ids, corners, marker_corners, marker_ids):
        self.ids = ids                  # (N,) int, ChArUco corner ids
        self.corners = corners          # (N, 2) float, pixel coordinates
        self.marker_corners = marker_corners
        self.marker_ids = marker_ids
        self.rvec = None                # board pose in the camera frame, if estimated
        self.tvec = None
        self.reprojection_rms = None    # px, of the single-image PnP fit


class CharucoBoard(object):
    def __init__(self, squares_x, squares_y, square_length, marker_length, dictionary,
                 min_corners=8):
        if dictionary not in DICTIONARIES:
            raise ValueError("unknown aruco dictionary {}".format(dictionary))
        if not 0.0 < marker_length < square_length:
            raise ValueError("marker_length must be positive and smaller than square_length")
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARIES[dictionary])
        self.board = cv2.aruco.CharucoBoard_create(
            int(squares_x), int(squares_y), float(square_length), float(marker_length),
            self.dictionary)
        # Board frame: origin at the outer corner of the first square, z out of the board.
        self.object_points = np.asarray(self.board.chessboardCorners, dtype=np.float64).reshape(-1, 3)
        self.min_corners = max(6, int(min_corners))
        self.square_length = float(square_length)
        self.size = (squares_x * square_length, squares_y * square_length)

    @classmethod
    def from_params(cls, params):
        return cls(params["squares_x"], params["squares_y"], params["square_length"],
                   params["marker_length"], params["dictionary"], params.get("min_corners", 8))

    def detect(self, image, K, D):
        """Return a Detection, or None without markers. The pose is set only with enough corners."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if marker_ids is None or len(marker_ids) < 2:
            return None
        count, corners, ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.board, cameraMatrix=K, distCoeffs=D)
        if ids is None or count == 0:
            return Detection(np.zeros(0, int), np.zeros((0, 2)), marker_corners, marker_ids)
        detection = Detection(ids.reshape(-1).astype(int), corners.reshape(-1, 2).astype(np.float64),
                              marker_corners, marker_ids)
        if count >= self.min_corners:
            self.estimate_pose(detection, K, D)
        return detection

    def estimate_pose(self, detection, K, D):
        points = self.object_points[detection.ids]
        # Corners on a single row are degenerate for PnP.
        if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-6) < 2:
            return False
        ok, rvec, tvec = cv2.solvePnP(points, detection.corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return False
        projected, _ = cv2.projectPoints(points, rvec, tvec, K, D)
        error = projected.reshape(-1, 2) - detection.corners
        detection.rvec, detection.tvec = rvec.reshape(3), tvec.reshape(3)
        detection.reprojection_rms = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
        return True

    def is_valid(self, detection):
        return (detection is not None and len(detection.ids) >= self.min_corners
                and detection.rvec is not None)

    def draw(self, image, detection, K, D):
        """Draw markers, corners and the board axes in place."""
        if detection is None:
            return image
        if detection.marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(image, detection.marker_corners, detection.marker_ids)
        if len(detection.ids):
            cv2.aruco.drawDetectedCornersCharuco(
                image, detection.corners.reshape(-1, 1, 2).astype(np.float32),
                detection.ids.reshape(-1, 1), (0, 255, 0))
        if detection.rvec is not None:
            cv2.aruco.drawAxis(image, K, D, detection.rvec, detection.tvec, 2.0 * self.square_length)
        return image
