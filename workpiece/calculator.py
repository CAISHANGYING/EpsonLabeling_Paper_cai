import numpy as np

class WorkpieceCalculator:

    @staticmethod
    def calculateNorm(vector: list) -> float:
        # 使用 NumPy 計算範數
        return np.linalg.norm(vector)

    @staticmethod
    def calAngle(v1: list, v2: list, xAngle: float = 0, yAngle: float = 0) -> float:
        # 將角度轉換為弧度
        alpha_x = np.radians(xAngle)  # x 軸旋轉角度
        beta_y = np.radians(yAngle)   # y 軸旋轉角度

        # 應用旋轉角度計算新向量
        v1_rotated = np.array([v1[0] * np.cos(alpha_x), v1[1] * np.cos(beta_y)])
        v2_rotated = np.array([v2[0] * np.cos(alpha_x), v2[1] * np.cos(beta_y)])

        # 計算內積和行列式來得到角度
        dotProduct = np.dot(v1_rotated, v2_rotated)
        determinant = np.linalg.det([v1_rotated, v2_rotated])

        # 使用 atan2 計算角度並轉換為度數
        angle = np.degrees(np.arctan2(determinant, dotProduct))

        return angle

    @staticmethod
    def calLength(v: list, xAngle: float = 0, yAngle: float = 0) -> float:
        # 將角度轉換為弧度並計算旋轉後的向量長度
        alpha_x = np.radians(xAngle)
        beta_y = np.radians(yAngle)

        # 應用旋轉角度計算新向量
        v_rotated = np.array([v[0] * np.cos(alpha_x), v[1] * np.cos(beta_y)])

        # 使用 NumPy 的 norm 計算長度
        return np.linalg.norm(v_rotated)

    @staticmethod
    def calRatio(v1: list, v2: list, rollAngle: float = 0, pitchAngle: float = 0) -> float:
        # 計算兩個向量的長度
        length1 = WorkpieceCalculator.calLength(v1, rollAngle, pitchAngle)
        length2 = WorkpieceCalculator.calLength(v2, rollAngle, pitchAngle)

        # 安全地計算長度比，避免除以 0
        return length2 / length1 if length1 != 0 else 0
