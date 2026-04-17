import os
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from src.transformer import ViewTransformer
from src.analyzer import FootballAnalyzer

class FootballDetector:
    def __init__(self, model_path='models/yolov8s.pt'): # 8n -> 8s 모델 업그레이드 (공 인식 정확도 향상)
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.analyzer = FootballAnalyzer()
        self.player_tracks = {} 

    def detect_players_for_selection(self, video_path):
        """첫 프레임에서 모든 선수들의 좌표와 ID를 추출하여 반환"""
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        
        if not success:
            return []
            
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        # 0: person, 32: sports ball
        detections = detections[(detections.class_id == 0) | (detections.class_id == 32)]
        
        players = []
        for i, (xyxy, mask, conf, class_id, tracker_id, data) in enumerate(detections):
            x1, y1, x2, y2 = xyxy
            players.append({
                "id": i, 
                "class": self.model.model.names[class_id],
                "x": float((x1 + x2) / 2),
                "y": float(y2), 
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        return players

    def process_video_v2(self, input_path, output_path, calibration_points, target_player_id, progress_callback=None):
        """보정 좌표와 타겟 플레이어를 사용하여 정밀 분석 수행"""
        print(f"Starting advanced ball-centric processing for {input_path}")
        
        target_field = np.array([[0,0], [100,0], [100,50], [0,50]], dtype=np.float32)
        source_field = np.array(calibration_points, dtype=np.float32)
        transformer = ViewTransformer(source_field, target_field)
        
        video_info = sv.VideoInfo.from_video_path(input_path)
        total_frames = video_info.total_frames
        target_track_id = None
        
        # 타켓 추적 상태 관리
        last_target_pos = None 

        def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
            nonlocal target_track_id, last_target_pos
            try:
                if progress_callback and index % 10 == 0:
                    percent = (index / total_frames) * 100
                    progress_callback(percent)

                results = self.model(frame, conf=0.15)[0] 
                detections = sv.Detections.from_ultralytics(results)
                detections = detections[(detections.class_id == 0) | (detections.class_id == 32)]
                detections = self.tracker.update_with_detections(detections)
                
                # --- 타겟 락 / 복구 로직 ---
                current_target_detected = False
                if len(detections) > 0:
                    # 1. 기존 ID 검색
                    if target_track_id is not None:
                        for i, tid in enumerate(detections.tracker_id):
                            if int(tid) == target_track_id:
                                xyxy = detections.xyxy[i]
                                last_target_pos = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
                                current_target_detected = True
                                break
                    
                    # 2. 로스트 시 반경 내 재탐색 (ID가 바뀌었을 가능성 대비)
                    if not current_target_detected and last_target_pos is not None:
                        min_dist = 150 # 150px 이내에서 재탐색
                        for i, (xyxy, _, _, _, tid, _) in enumerate(detections):
                            if detections.class_id[i] == 0:
                                cx, cy = (xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2
                                dist = np.sqrt((cx - last_target_pos[0])**2 + (cy - last_target_pos[1])**2)
                                if dist < min_dist:
                                    target_track_id = int(tid)
                                    last_target_pos = [cx, cy]
                                    current_target_detected = True
                                    print(f"Target re-acquired at frame {index}! ID: {target_track_id}")
                                    break

                    # 3. 최초 락 (Step 0 에서 선택한 좌표 기반)
                    if target_track_id is None:
                        min_dist = 120
                        for i, (xyxy, _, _, _, tid, _) in enumerate(detections):
                            if detections.class_id[i] == 0: 
                                cx, cy = (xyxy[0] + xyxy[2]) / 2, xyxy[3] 
                                dist = np.sqrt((cx - target_player_id['x'])**2 + (cy - target_player_id['y'])**2)
                                if dist < min_dist: 
                                    min_dist = dist
                                    target_track_id = int(tid)
                                    last_target_pos = [cx, cy]
                                    print(f"Initial Target locked! ID: {target_track_id}")

                if len(detections) > 0:
                    points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    transformed_points = transformer.transform_points(points)
                    
                    for i, track_id in enumerate(detections.tracker_id):
                        pos = transformed_points[i]
                        tid = int(track_id)
                        xyxy = detections.xyxy[i]
                        pos_px = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
                        
                        if tid not in self.player_tracks: self.player_tracks[tid] = []
                        self.player_tracks[tid].append({
                            "frame": int(index), "pos": pos.tolist(), "pos_px": pos_px,
                            "class": int(detections.class_id[i]), "conf": float(detections.confidence[i])
                        })

                # Spotlight 모드 가시화
                annotated_frame = frame.copy()
                if target_track_id is not None:
                    mask = (detections.tracker_id == target_track_id)
                    target_detections = detections[mask]
                    if len(target_detections) > 0:
                        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=target_detections)
                        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=target_detections, labels=[f"TARGET #{target_track_id}"])
                return annotated_frame
            except Exception as e:
                print(f"Error processing frame {index}: {e}")
                return frame

        # 비디오 처리 (OpenCV 직접 제어하여 코덱 호환성 확보)
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 프레임 처리 로직 호출
            annotated_frame = process_frame(frame, frame_index)
            out.write(annotated_frame)
            frame_index += 1
            
        cap.release()
        out.release()
        
        print(f"Processing complete. Target Track ID: {target_track_id}")
        print(f"Total tracked players: {len(self.player_tracks)}")
        
        # 분석 결과 생성
        stats = self.analyzer.calculate_individual_stats(self.player_tracks, target_track_id)
        highlights = self.analyzer.extract_combined_highlights(input_path, self.player_tracks, target_track_id)
        
        print(f"Stats generated: {stats}")
        print(f"Highlights count: {len(highlights)}")
        
        return {
            "stats": stats,
            "highlights": highlights,
            "target_track_id": int(target_track_id) if target_track_id is not None else None
        }
