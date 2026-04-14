import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import kde

class FootballAnalyzer:
    def __init__(self):
        self.data_dir = os.path.join("data", "analysis")
        os.makedirs(self.data_dir, exist_ok=True)

    def generate_heatmap(self, positions, output_filename="heatmap.png"):
        """
        positions: list of (u, v) coordinates in meters
        """
        if len(positions) < 2:
            print("Not enough data for heatmap.")
            return

        x = [p[0] for p in positions]
        y = [p[1] for p in positions]

        # KDE(Kernel Density Estimation) 기반 히트맵 생성
        plt.figure(figsize=(10, 6))
        plt.hist2d(x, y, bins=[50, 30], cmap='hot', range=[[0, 100], [0, 50]])
        
        # 경기장 배경 그리기 (단순 사각형)
        plt.plot([0, 100, 100, 0, 0], [0, 0, 50, 50, 0], color='white', linewidth=2)
        plt.plot([50, 50], [0, 50], color='white', linewidth=2) # 센터 라인
        
        plt.title("Player Activity Heatmap (Pitch Scale)")
        plt.xlabel("Meters")
        plt.ylabel("Meters")
        plt.gca().set_facecolor('green')
        
        save_path = os.path.join(self.data_dir, output_filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Heatmap saved to: {save_path}")
        return save_path

    def calculate_basic_stats(self, tracks):
        """
        tracks: dict mapping track_id to list of (u, v) positions
        """
        stats = {}
        for track_id, positions in tracks.items():
            if len(positions) < 2:
                continue
            
            # 총 이동 거리 계산 (유클리드 거리 합)
            distance = 0
            for i in range(1, len(positions)):
                p1 = positions[i-1]
                p2 = positions[i]
                d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                distance += d
            
            stats[track_id] = {
                "distance_meters": round(distance, 2),
                "avg_pos": np.mean(positions, axis=0).tolist()
            }
        return stats

if __name__ == "__main__":
    # 간단한 테스트
    analyzer = FootballAnalyzer()
    dummy_pos = np.random.rand(100, 2) * [100, 50]
    analyzer.generate_heatmap(dummy_pos, "test_heatmap.png")
