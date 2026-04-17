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
        self._team_centers = None   # K-means team color centroids
        self._target_velocity = np.array([0.0, 0.0])  # pixel/frame velocity

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

    def _best_appearance_match_team(self, frame, dets, target_team, tid_team, threshold=0.40):
        """Appearance re-ID with team-aware penalty."""
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
            t_id = int(dets.tracker_id[i]) if dets.tracker_id is not None else -1
            t_team = tid_team.get(t_id, -1)
            if target_team != -1 and t_team != -1 and t_team != target_team:
                s *= 0.15  # 상대팀 강력 패널티
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    # ── Team Color Clustering ────────────────────────────────────────────────

    def _cluster_teams_init(self, frame, dets):
        """K-means (k=2) on jersey histograms to learn 2 team color centroids."""
        hists = []
        for i in range(len(dets)):
            if dets.class_id[i] != 0:
                continue
            h = self._extract_jersey_hist(frame, dets.xyxy[i])
            if h is not None:
                hists.append(h.flatten().astype(np.float32))
        if len(hists) < 4:
            self._team_centers = None
            return
        data = np.array(hists)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, _, centers = cv2.kmeans(
            data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        self._team_centers = centers
        print(f"[TAD] Team clustering done: {len(hists)} players sampled")

    def _get_team(self, frame, bbox):
        """Classify player bbox as team 0 or 1 using stored centroids. -1 if unavailable."""
        if self._team_centers is None:
            return -1
        h = self._extract_jersey_hist(frame, bbox)
        if h is None:
            return -1
        hf = h.flatten().astype(np.float32)
        dists = [float(np.linalg.norm(hf - self._team_centers[k])) for k in range(2)]
        return int(np.argmin(dists))

    # ── Gap Interpolation ────────────────────────────────────────────────────

    def _interpolate_gaps(self, frames, fps, max_gap_sec=1.0):
        """Linear interpolation for tracking gaps up to max_gap_sec seconds."""
        if len(frames) < 2:
            return frames
        max_gap = int(fps * max_gap_sec)
        result = list(frames)
        for i in range(len(frames) - 1):
            f1, f2 = frames[i]['frame'], frames[i+1]['frame']
            gap = f2 - f1
            if gap <= 1 or gap > max_gap:
                continue
            p1, p2   = frames[i]['pos'],    frames[i+1]['pos']
            px1, px2 = frames[i]['pos_px'], frames[i+1]['pos_px']
            for f in range(f1 + 1, f2):
                a = (f - f1) / gap
                result.append({
                    "frame": f,
                    "pos":    [p1[0]  + a*(p2[0]-p1[0]),   p1[1]  + a*(p2[1]-p1[1])],
                    "pos_px": [px1[0] + a*(px2[0]-px1[0]), px1[1] + a*(px2[1]-px1[1])],
                    "class": 0, "conf": 0.0,
                    "bbox": frames[i]['bbox'],
                    "interpolated": True,
                })
        result.sort(key=lambda x: x['frame'])
        return result

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

        # ── Team color clustering (first frame) ───────────────────────────
        ok, first_frame = cap.read()
        if ok:
            r0 = self.model(first_frame, conf=0.25, verbose=False)[0]
            d0 = sv.Detections.from_ultralytics(r0)
            self._cluster_teams_init(first_frame, d0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind to start

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (W, H))
        if not out.isOpened():
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

        target_id   = None
        last_px     = None
        frames_lost = 0
        target_team = -1          # team id of the target player
        tid_team    = {}          # cache: tracker_id -> team_id
        PROXY_WIN   = int(fps * 2)
        APPEAR_WIN  = int(fps * 4)

        # Unified target track — ALL frames regardless of ID changes
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
                            new_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            if last_px is not None:
                                vel = np.array(new_px) - np.array(last_px)
                                self._target_velocity = 0.75 * self._target_velocity + 0.25 * vel
                            last_px = new_px
                            self._update_target_hist(frame, xyxy)
                            found = True
                            frames_lost = 0

                    # ── 2. Position + jersey + team re-lock (< 2s) ───────
                    if not found and last_px is not None and frames_lost < PROXY_WIN:
                        # 속도 예측으로 예상 위치 계산
                        pred = np.array(last_px) + self._target_velocity * min(frames_lost, PROXY_WIN)
                        pred = np.clip(pred, [0, 0], [W, H])

                        best_i, best_score = None, -1.0
                        for i in range(len(dets)):
                            if dets.class_id[i] != 0:
                                continue
                            xyxy = dets.xyxy[i]
                            cx = (xyxy[0] + xyxy[2]) / 2
                            cy = (xyxy[1] + xyxy[3]) / 2
                            d = np.hypot(cx - pred[0], cy - pred[1])
                            if d > 280:
                                continue
                            # 유니폼 색상 유사도
                            sim = 0.35
                            if self._target_hist is not None:
                                h = self._extract_jersey_hist(frame, xyxy)
                                if h is not None:
                                    sim = max(0.0, cv2.compareHist(
                                        self._target_hist, h, cv2.HISTCMP_CORREL))
                            # 상대팀 패널티
                            t_id = int(dets.tracker_id[i]) if dets.tracker_id is not None else -1
                            t_team = tid_team.get(t_id, -1)
                            if target_team != -1 and t_team != -1 and t_team != target_team:
                                sim *= 0.15
                            score = 0.35 * max(0.0, 1.0 - d / 280.0) + 0.65 * sim
                            if score > best_score:
                                best_score = score
                                best_i = i

                        if best_i is not None and best_score >= 0.22:
                            target_id = int(dets.tracker_id[best_i])
                            found_i = best_i
                            xyxy = dets.xyxy[best_i]
                            new_px = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
                            vel = (np.array(new_px) - np.array(last_px)) / max(frames_lost, 1)
                            self._target_velocity = 0.6 * self._target_velocity + 0.4 * vel
                            last_px = new_px
                            self._update_target_hist(frame, xyxy)
                            found = True
                            frames_lost = 0
                            print(f"[TAD] Pos re-lock f={idx} id={target_id} score={best_score:.2f}")

                    # ── 3. Appearance re-ID with team filter (> 4s) ──────
                    if not found and frames_lost >= APPEAR_WIN:
                        best_i = self._best_appearance_match_team(
                            frame, dets, target_team, tid_team, threshold=0.40)
                        if best_i is not None:
                            target_id = int(dets.tracker_id[best_i])
                            found_i = best_i
                            xyxy = dets.xyxy[best_i]
                            new_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            self._target_velocity = np.array([0.0, 0.0])  # 긴 공백 후 초기화
                            last_px = new_px
                            self._update_target_hist(frame, xyxy)
                            found = True
                            frames_lost = 0
                            print(f"[TAD] Appearance re-lock f={idx} id={target_id}")

                    # ── 4. Initial lock ───────────────────────────────────
                    if target_id is None:
                        best_i, best_d = None, 180.0
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
                            self._target_velocity = np.array([0.0, 0.0])
                            self._update_target_hist(frame, xyxy)
                            found = True
                            print(f"[TAD] Initial lock id={target_id} dist={best_d:.0f}px")

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
                        if tid not in tid_team:
                            tid_team[tid] = self._get_team(frame, xyxy)
                        if tid not in self.player_tracks:
                            self.player_tracks[tid] = []
                        self.player_tracks[tid].append({
                            "frame":   idx,
                            "pos":     tpts[i].tolist(),
                            "pos_px":  [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2],
                            "class":   int(dets.class_id[i]),
                            "conf":    float(dets.confidence[i]) if dets.confidence is not None else 0.5,
                            "bbox":    xyxy.tolist(),
                            "team_id": tid_team[tid],
                        })

                    # ── Append to unified target track ────────────────────
                    if found and found_i is not None:
                        xyxy = dets.xyxy[found_i]
                        # Capture target's team on first lock
                        if target_team == -1:
                            target_team = self._get_team(frame, xyxy)
                            print(f"[TAD] Target team: {target_team}")
                        target_frames.append({
                            "frame":   idx,
                            "pos":     tpts[found_i].tolist(),
                            "pos_px":  [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2],
                            "class":   0,
                            "conf":    float(dets.confidence[found_i]) if dets.confidence is not None else 0.5,
                            "bbox":    xyxy.tolist(),
                            "team_id": target_team,
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

        # Fill gaps with linear interpolation (≤1s gaps)
        target_frames = self._interpolate_gaps(target_frames, fps, max_gap_sec=1.0)

        # Store unified target track under fixed key
        self.player_tracks[UNIFIED_TARGET_ID] = target_frames
        print(f"[TAD] Done. target_id={target_id}, unified_frames={len(target_frames)}, "
              f"target_team={target_team}, players_tracked={len(self.player_tracks)}")

        stats = self.analyzer.calculate_individual_stats(
            self.player_tracks, UNIFIED_TARGET_ID, fps=fps
        )
        return {
            "stats":           stats,
            "target_track_id": UNIFIED_TARGET_ID,
            "target_team":     target_team,
        }
