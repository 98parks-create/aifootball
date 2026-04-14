import os
import numpy as np
import supervision as sv
from ultralytics import YOLO
from src.transformer import get_transformer
from src.analyzer import FootballAnalyzer

class FootballDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.transformer = get_transformer()
        self.analyzer = FootballAnalyzer()
        self.all_positions = [] # For global heatmap
        self.player_tracks = {} # For individual stats {track_id: [pos1, pos2, ...]}

    def process_video(self, input_path, output_path):
        print(f"Processing video: {input_path}")
        video_info = sv.VideoInfo.from_video_path(input_path)
        
        def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
            results = self.model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[(detections.class_id == 0) | (detections.class_id == 32)]
            detections = self.tracker.update_with_detections(detections)
            
            # 좌표 변환 및 데이터 저장
            if len(detections) > 0:
                # 박스의 하단 중앙 좌표 추출 (발 위치)
                points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                transformed_points = self.transformer.transform_points(points)
                
                for i, track_id in enumerate(detections.tracker_id):
                    pos = transformed_points[i]
                    self.all_positions.append(pos)
                    if track_id not in self.player_tracks:
                        self.player_tracks[track_id] = []
                    self.player_tracks[track_id].append(pos)

            labels = [
                f"#{tracker_id} {self.model.model.names[class_id]}"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
            ]
            
            annotated_frame = frame.copy()
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            return annotated_frame

        sv.process_video(source_path=input_path, target_path=output_path, callback=process_frame)
        print(f"Video saved to: {output_path}")
        
        # 분석 결과 생성
        print("Generating analysis results...")
        self.analyzer.generate_heatmap(self.all_positions, "final_heatmap.png")
        stats = self.analyzer.calculate_basic_stats(self.player_tracks)
        print(f"Stats calculated for {len(stats)} players.")
        return stats

if __name__ == "__main__":
    input_video = os.path.join("data", "uploads", "sample.mp4")
    output_video = os.path.join("data", "processed", "detected_sample.mp4")
    
    if not os.path.exists(input_video):
        print(f"Input video not found: {input_video}")
    else:
        detector = FootballDetector('yolov8n.pt') 
        stats = detector.process_video(input_video, output_video)
        print("Analysis complete.")
