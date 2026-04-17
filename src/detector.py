import os
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from src.transformer import ViewTransformer
from src.analyzer import FootballAnalyzer

try:
    import torch
    _DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    _DEVICE = 'cpu'

print(f"[TAD] Device: {_DEVICE}")

# Unified target track is stored under this key in player_tracks
UNIFIED_TARGET_ID = -1


class FootballDetector:
    def __init__(self, model_path='models/yolov8s.pt'):
        self.model = YOLO(model_path)
        self.model.to(_DEVICE)
        self.analyzer = FootballAnalyzer()
        self.player_tracks = {}
        self._target_hist = None
        self._target_hist_count = 0

    # ── Appearance Re-ID ─────────────────────────────────────────────────────

    def _extract_jersey_hist(self, frame, bbox):
        x1 = max(0, int(bbox[0]));  y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1], int(bbox[2]));  y2 = min(frame.shape[0], int(bbox[3]))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        torso = roi[:max(1, int(roi.shape[0] * 0.5)), :]
        if torso.size == 0:
            return None
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _update_target_hist(self, frame, bbox):
        h = self._extract_jersey_hist(frame, bbox)
        if h is None:
            return
        if self._target_hist is None:
            self._target_hist = h.copy()
            self._target_hist_count = 1
        else:
            n = min(self._target_hist_count, 15)
            self._target_hist = (self._target_hist * n + h) / (n + 1)
            self._target_hist_count += 1

    def _best_appearance_match(self, frame, dets, threshold=0.45):
        if self._target_hist is None:
            return None
        best_i, best_s = None, threshold
        for i in range(len(dets)):
            if dets.class_id[i] != 0:
                continue
            h = self._extract_jersey_hist(frame, dets.xyxy[i])
            if h is None:
                continue
            s = cv2.compareHist(self._target_hist, h, cv2.HISTCMP_CORREL)
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    # ── Public API ───────────────────────────────────────────────────────────

    def detect_players_for_selection(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return []
        results = self.model(frame, conf=0.2, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results)
        dets = dets[(dets.class_id == 0) | (dets.class_id == 32)]
        players = []
        for i, (xyxy, _, conf, class_id, _, _) in enumerate(dets):
            x1, y1, x2, y2 = xyxy
            players.append({
                "id": i,
                "class": self.model.model.names[int(class_id)],
                "x": float((x1 + x2) / 2),
                "y": float(y2),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })
        return players

    def process_video_v2(self, input_path, output_path, calibration_points,
                         target_player_id, progress_callback=None):
        print(f"[TAD] Starting tracking: {input_path}")

        src = np.array(calibration_points, dtype=np.float32)
        dst = np.array([[0,0],[100,0],[100,50],[0,50]], dtype=np.float32)
        transformer = ViewTransformer(src, dst)

        info = sv.VideoInfo.from_video_path(input_path)
        total_frames = max(info.total_frames, 1)
        fps = info.fps or 25

        cap = cv2.VideoCapture(input_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (W, H))
        if not out.isOpened():
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

        target_id = None
        last_px = None
        frames_lost = 0
        PROXY_WIN = int(fps * 2)
        APPEAR_WIN = int(fps * 4)

        # Unified target track — accumulates ALL frames where target is found,
        # regardless of tracker ID changes (re-locks, camera cuts, etc.)
        target_frames = []

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if progress_callback and idx % 15 == 0:
                progress_callback((idx / total_frames) * 100)

            try:
                results = self.model.track(
                    frame, persist=True, conf=0.2,
                    classes=[0, 32], verbose=False
                )[0]
                dets = sv.Detections.from_ultralytics(results)

                if dets.tracker_id is not None:
                    valid = dets.tracker_id != -1
                    dets = dets[valid]

                found = False
                found_i = None   # index in dets of the confirmed target this frame

                if len(dets) > 0 and dets.tracker_id is not None:

                    # ── 1. Current ID still visible ───────────────────────
                    if target_id is not None:
                        where = np.where(dets.tracker_id == target_id)[0]
                        if len(where):
                            found_i = where[0]
                            xyxy = dets.xyxy[found_i]
                            last_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            self._update_target_hist(frame, xyxy)
                            found = True
                            frames_lost = 0

                    # ── 2. Position-based recovery (< 2s, < 200px) ───────
                    if not found and last_px is not None and frames_lost < PROXY_WIN:
                        best_i, best_d = None, 200.0
                        for i in range(len(dets)):
                            if dets.class_id[i] != 0:
                                continue
                            xyxy = dets.xyxy[i]
                            cx = (xyxy[0]+xyxy[2])/2
                            cy = (xyxy[1]+xyxy[3])/2
                            d = np.hypot(cx - last_px[0], cy - last_px[1])
                            if d < best_d:
                                best_d, best_i = d, i
                        if best_i is not None:
                            target_id = int(dets.tracker_id[best_i])
                            found_i = best_i
                            xyxy = dets.xyxy[best_i]
                            last_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            self._update_target_hist(frame, xyxy)
                            found = True
                            frames_lost = 0
                            print(f"[TAD] Position re-lock f={idx} id={target_id}")

                    # ── 3. Appearance re-ID (after long absence) ─────────
                    if not found and frames_lost >= APPEAR_WIN:
                        best_i = self._best_appearance_match(frame, dets, threshold=0.40)
                        if best_i is not None:
                            target_id = int(dets.tracker_id[best_i])
                            found_i = best_i
                            xyxy = dets.xyxy[best_i]
                            last_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            self._update_target_hist(frame, xyxy)
                            found = True
                            frames_lost = 0
                            print(f"[TAD] Appearance re-lock f={idx} id={target_id}")

                    # ── 4. Initial lock ───────────────────────────────────
                    if target_id is None:
                        best_i, best_d = None, 140.0
                        for i in range(len(dets)):
                            if dets.class_id[i] != 0:
                                continue
                            xyxy = dets.xyxy[i]
                            cx = (xyxy[0]+xyxy[2])/2
                            cy = xyxy[3]
                            d = np.hypot(cx - target_player_id['x'], cy - target_player_id['y'])
                            if d < best_d:
                                best_d, best_i = d, i
                        if best_i is not None:
                            target_id = int(dets.tracker_id[best_i])
                            found_i = best_i
                            xyxy = dets.xyxy[best_i]
                            last_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            self._update_target_hist(frame, xyxy)
                            found = True
                            print(f"[TAD] Initial lock id={target_id}")

                    if not found:
                        frames_lost += 1

                    # ── Store all tracks ──────────────────────────────────
                    anchor_pts = dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    tpts = transformer.transform_points(anchor_pts)
                    for i in range(len(dets)):
                        if dets.tracker_id[i] is None or dets.tracker_id[i] == -1:
                            continue
                        tid = int(dets.tracker_id[i])
                        xyxy = dets.xyxy[i]
                        if tid not in self.player_tracks:
                            self.player_tracks[tid] = []
                        self.player_tracks[tid].append({
                            "frame": idx,
                            "pos": tpts[i].tolist(),
                            "pos_px": [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2],
                            "class": int(dets.class_id[i]),
                            "conf": float(dets.confidence[i]) if dets.confidence is not None else 0.5,
                            "bbox": xyxy.tolist(),
                        })

                    # ── Append to unified target track ────────────────────
                    if found and found_i is not None:
                        xyxy = dets.xyxy[found_i]
                        target_frames.append({
                            "frame": idx,
                            "pos": tpts[found_i].tolist(),
                            "pos_px": [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2],
                            "class": 0,
                            "conf": float(dets.confidence[found_i]) if dets.confidence is not None else 0.5,
                            "bbox": xyxy.tolist(),
                        })

                # ── Draw target overlay ───────────────────────────────────
                annotated = frame.copy()
                if target_id is not None and len(dets) > 0 and dets.tracker_id is not None:
                    where = np.where(dets.tracker_id == target_id)[0]
                    if len(where):
                        xyxy = dets.xyxy[where[0]]
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 40, 255), 3)
                        arm = min(35, (x2-x1)//3, (y2-y1)//3)
                        for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                            dx = arm if px == x1 else -arm
                            dy = arm if py == y1 else -arm
                            cv2.line(annotated, (px,py), (px+dx, py), (0,215,255), 4)
                            cv2.line(annotated, (px,py), (px, py+dy), (0,215,255), 4)
                        cv2.putText(annotated, "TARGET", (x1, max(y1-10, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,215,255), 2, cv2.LINE_AA)
                out.write(annotated)

            except Exception as e:
                print(f"[TAD] Frame {idx} error: {e}")
                import traceback; traceback.print_exc()
                out.write(frame)

            idx += 1

        cap.release()
        out.release()

        # Store unified target track under a fixed key so all downstream
        # functions (stats, highlights, heatmap) always use the complete trajectory
        self.player_tracks[UNIFIED_TARGET_ID] = target_frames
        print(f"[TAD] Done. target_id={target_id}, unified_frames={len(target_frames)}, players_tracked={len(self.player_tracks)}")

        stats = self.analyzer.calculate_individual_stats(
            self.player_tracks, UNIFIED_TARGET_ID, fps=fps
        )
        return {
            "stats": stats,
            "target_track_id": UNIFIED_TARGET_ID,
        }
