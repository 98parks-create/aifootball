import numpy as np
import cv2

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# Stuttgart Mercedes-Benz Arena Sample calibration
# 화면상 4개 좌표 -> 실제 경기장 평면 좌표 (미터 단위)
# 반코트 기준 (가로 105m, 세로 68m) -> 원점 (52.5, 34)
SOURCE_POINTS = np.array([
    [933, 283],   # 1. 왼쪽 상단 (터치라인 부근)
    [1236, 283],  # 2. 오른쪽 상단
    [1537, 898],  # 3. 오른쪽 하단
    [644, 898]    # 4. 왼쪽 하단 
])

TARGET_POINTS = np.array([
    [0, 0],
    [100, 0],
    [100, 50],
    [0, 50]
])

def get_transformer():
    return ViewTransformer(SOURCE_POINTS, TARGET_POINTS)
