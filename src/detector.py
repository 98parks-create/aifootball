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
    def __init__(self, model_path='models/yolov8s.pt', scan_model_path=None):
        self.model = YOLO(model_path)
        self.model.to(_DEVICE)
        # 480p 스캔 전용 경량 모델 (없으면 기본 모델 공유)
        if scan_model_path and os.path.exists(scan_model_path):
            self.scan_model = YOLO(scan_model_path)
            self.scan_model.to(_DEVICE)
            print(f"[TAD] 스캔 모델: {scan_model_path}")
        else:
            self.scan_model = self.model
        # GPU 최적화: FP16 + cudnn 자동 튜닝
        if _DEVICE == 'cuda':
            try:
                import torch
                torch.backends.cudnn.benchmark = True
                self.scan_model.model.half()
                if self.scan_model is not self.model:
                    self.model.model.half()
                print("[TAD] GPU FP16 활성화")
            except Exception as e:
                print(f"[TAD] FP16 비활성화 (fallback): {e}")
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

    @staticmethod
    def _cosine_sim(h1: np.ndarray, h2: np.ndarray) -> float:
        """두 히스토그램 배열 간 코사인 유사도 (0~1)."""
        a = h1.flatten().astype(np.float64)
        b = h2.flatten().astype(np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / (denom + 1e-9))

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
            s = self._cosine_sim(self._target_hist, h)
            if s > best_s:
                best_s, best_i = s, i
        return best_i

    def _best_appearance_match_team(self, frame, dets, target_team, tid_team, threshold=0.40):
        """코사인 유사도 기반 appearance re-ID (팀 패널티 적용)."""
        if self._target_hist is None:
            return None
        best_i, best_s = None, threshold
        for i in range(len(dets)):
            if dets.class_id[i] != 0:
                continue
            h = self._extract_jersey_hist(frame, dets.xyxy[i])
            if h is None:
                continue
            s = self._cosine_sim(self._target_hist, h)
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
        """
        2단계 처리:
          Stage-1 (0-100%): 480p 다운스케일로 빠른 전체 스캔 — 추적 데이터 수집
          Stage-2: app.py 에서 원본 해상도로 하이라이트 클립 추출
        """
        print(f"[TAD] Stage-1 480p 스캔 시작: {input_path}")

        info         = sv.VideoInfo.from_video_path(input_path)
        total_frames = max(info.total_frames, 1)
        fps          = info.fps or 25
        orig_w       = info.width  or 1280
        orig_h       = info.height or 720

        # ── 스캔 해상도 계산 (480p, 비율 유지) ──────────────────────────────
        TARGET_H   = 480
        scan_scale = min(TARGET_H / orig_h, 1.0)
        scan_w     = max(2, int(orig_w * scan_scale) - (int(orig_w * scan_scale) % 2))
        scan_h     = max(2, int(orig_h * scan_scale) - (int(orig_h * scan_scale) % 2))
        print(f"[TAD] 원본 {orig_w}×{orig_h} → 스캔 {scan_w}×{scan_h} (scale={scan_scale:.2f})")

        # ── 캘리브레이션 포인트 스케일 변환 ────────────────────────────────
        scaled_calib  = [[p[0] * scan_scale, p[1] * scan_scale] for p in calibration_points]
        scaled_target = {
            'x': target_player_id['x'] * scan_scale,
            'y': target_player_id['y'] * scan_scale,
        }
        src         = np.array(scaled_calib, dtype=np.float32)
        dst         = np.array([[0,0],[100,0],[100,50],[0,50]], dtype=np.float32)
        transformer = ViewTransformer(src, dst)

        # ── ByteTrack 설정 파일 경로 ────────────────────────────────────────
        _cfg_path   = os.path.join(os.path.dirname(__file__), '..', 'bytetrack.yaml')
        _tracker    = _cfg_path if os.path.exists(_cfg_path) else 'bytetrack.yaml'

        cap = cv2.VideoCapture(input_path)

        # ── 팀 색상 클러스터링 (첫 프레임) ──────────────────────────────────
        ok, first_frame = cap.read()
        if ok:
            small0 = cv2.resize(first_frame, (scan_w, scan_h))
            r0 = self.scan_model(small0, conf=0.25, verbose=False)[0]
            d0 = sv.Detections.from_ultralytics(r0)
            self._cluster_teams_init(small0, d0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        target_id     = None
        last_px       = None
        frames_lost   = 0
        target_team   = -1
        tid_team: dict = {}
        PROXY_WIN     = int(fps * 2)
        APPEAR_WIN    = int(fps * 4)
        target_frames: list = []
        # 타겟이 안정 추적 중일 때만 격 프레임 처리 (추적 불안정 시 매 프레임)
        STABLE_SKIP   = 2

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 타겟 안정 추적 중(frames_lost==0)에만 격 프레임 스킵
            if target_id is not None and frames_lost == 0 and idx % STABLE_SKIP != 0:
                idx += 1
                continue

            if progress_callback and idx % 8 == 0:
                progress_callback((idx / total_frames) * 100)

            # 480p 다운스케일
            small = cv2.resize(frame, (scan_w, scan_h))

            try:
                results = self.scan_model.track(
                    small, persist=True, conf=0.2,
                    classes=[0, 32], verbose=False,
                    tracker=_tracker,
                )[0]
                dets = sv.Detections.from_ultralytics(results)
                if dets.tracker_id is not None:
                    dets = dets[dets.tracker_id != -1]

                found   = False
                found_i = None

                if len(dets) > 0 and dets.tracker_id is not None:

                    # ── 1. 현재 ID 유지 ───────────────────────────────────
                    if target_id is not None:
                        where = np.where(dets.tracker_id == target_id)[0]
                        if len(where):
                            found_i = where[0]
                            xyxy    = dets.xyxy[found_i]
                            new_px  = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            if last_px is not None:
                                vel = np.array(new_px) - np.array(last_px)
                                self._target_velocity = 0.75 * self._target_velocity + 0.25 * vel
                            last_px = new_px
                            self._update_target_hist(small, xyxy)
                            found = True
                            frames_lost = 0

                    # ── 2. 위치 예측 + 유니폼 + 팀 재탐색 (< 2s) ─────────
                    if not found and last_px is not None and frames_lost < PROXY_WIN:
                        pred = np.array(last_px) + self._target_velocity * min(frames_lost, PROXY_WIN)
                        pred = np.clip(pred, [0, 0], [scan_w, scan_h])
                        best_i, best_score = None, -1.0
                        for i in range(len(dets)):
                            if dets.class_id[i] != 0:
                                continue
                            xyxy = dets.xyxy[i]
                            cx   = (xyxy[0] + xyxy[2]) / 2
                            cy   = (xyxy[1] + xyxy[3]) / 2
                            d    = np.hypot(cx - pred[0], cy - pred[1])
                            MAX_D = 280 * scan_scale
                            if d > MAX_D:
                                continue
                            sim = 0.35
                            if self._target_hist is not None:
                                h = self._extract_jersey_hist(small, xyxy)
                                if h is not None:
                                    sim = max(0.0, self._cosine_sim(self._target_hist, h))
                            t_id   = int(dets.tracker_id[i]) if dets.tracker_id is not None else -1
                            t_team = tid_team.get(t_id, -1)
                            if target_team != -1 and t_team != -1 and t_team != target_team:
                                sim *= 0.15
                            score = 0.35 * max(0.0, 1.0 - d / MAX_D) + 0.65 * sim
                            if score > best_score:
                                best_score = score
                                best_i = i
                        if best_i is not None and best_score >= 0.22:
                            target_id = int(dets.tracker_id[best_i])
                            found_i   = best_i
                            xyxy      = dets.xyxy[best_i]
                            new_px    = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            vel = (np.array(new_px) - np.array(last_px)) / max(frames_lost, 1)
                            self._target_velocity = 0.6 * self._target_velocity + 0.4 * vel
                            last_px = new_px
                            self._update_target_hist(small, xyxy)
                            found = True
                            frames_lost = 0
                            print(f"[TAD] Pos re-lock f={idx} id={target_id} score={best_score:.2f}")

                    # ── 3. 코사인 유사도 appearance re-ID (> 4s) ─────────
                    if not found and frames_lost >= APPEAR_WIN:
                        best_i = self._best_appearance_match_team(
                            small, dets, target_team, tid_team, threshold=0.40)
                        if best_i is not None:
                            target_id = int(dets.tracker_id[best_i])
                            found_i   = best_i
                            xyxy      = dets.xyxy[best_i]
                            new_px    = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            self._target_velocity = np.array([0.0, 0.0])
                            last_px = new_px
                            self._update_target_hist(small, xyxy)
                            found = True
                            frames_lost = 0
                            print(f"[TAD] Appearance re-lock f={idx} id={target_id}")

                    # ── 4. 최초 락온 ──────────────────────────────────────
                    if target_id is None:
                        best_i, best_d = None, 180.0 * scan_scale
                        for i in range(len(dets)):
                            if dets.class_id[i] != 0:
                                continue
                            xyxy = dets.xyxy[i]
                            cx = (xyxy[0]+xyxy[2])/2
                            cy = xyxy[3]
                            d  = np.hypot(cx - scaled_target['x'], cy - scaled_target['y'])
                            if d < best_d:
                                best_d, best_i = d, i
                        if best_i is not None:
                            target_id = int(dets.tracker_id[best_i])
                            found_i   = best_i
                            xyxy      = dets.xyxy[best_i]
                            last_px   = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            self._target_velocity = np.array([0.0, 0.0])
                            self._update_target_hist(small, xyxy)
                            found = True
                            print(f"[TAD] Initial lock id={target_id} dist={best_d:.0f}px")

                    if not found:
                        frames_lost += 1

                    # ── 모든 트랙 저장 ────────────────────────────────────
                    anchor_pts = dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    tpts       = transformer.transform_points(anchor_pts)
                    for i in range(len(dets)):
                        if dets.tracker_id[i] is None or dets.tracker_id[i] == -1:
                            continue
                        tid  = int(dets.tracker_id[i])
                        xyxy = dets.xyxy[i]
                        if tid not in tid_team:
                            tid_team[tid] = self._get_team(small, xyxy)
                        if tid not in self.player_tracks:
                            self.player_tracks[tid] = []
                        self.player_tracks[tid].append({
                            "frame":   idx,
                            "pos":     tpts[i].tolist(),
                            "pos_px":  [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2],
                            "class":   int(dets.class_id[i]),
                            "conf":    float(dets.confidence[i]) if dets.confidence is not None else 0.5,
                            "bbox":    xyxy.tolist(),
                            "team_id": tid_team.get(tid, -1),
                        })

                    # ── 타겟 통합 트랙 추가 ───────────────────────────────
                    if found and found_i is not None:
                        xyxy = dets.xyxy[found_i]
                        if target_team == -1:
                            target_team = self._get_team(small, xyxy)
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

            except Exception as e:
                print(f"[TAD] Frame {idx} error: {e}")
                import traceback; traceback.print_exc()

            idx += 1

        cap.release()

        # ── pos_px 480p → 원본 해상도로 역스케일 ────────────────────────────
        px_sx = orig_w / scan_w
        px_sy = orig_h / scan_h
        for _frames in self.player_tracks.values():
            for t in _frames:
                if 'pos_px' in t:
                    t['pos_px'] = [t['pos_px'][0] * px_sx, t['pos_px'][1] * px_sy]
        for t in target_frames:
            if 'pos_px' in t:
                t['pos_px'] = [t['pos_px'][0] * px_sx, t['pos_px'][1] * px_sy]

        # ── 갭 보간 (≤1s) → 통합 타겟 트랙 저장 ────────────────────────────
        target_frames = self._interpolate_gaps(target_frames, fps, max_gap_sec=1.0)
        self.player_tracks[UNIFIED_TARGET_ID] = target_frames
        print(f"[TAD] Stage-1 완료. target_id={target_id}, unified={len(target_frames)}, "
              f"team={target_team}, players={len(self.player_tracks)}")

        stats = self.analyzer.calculate_individual_stats(
            self.player_tracks, UNIFIED_TARGET_ID, fps=fps
        )
        return {
            "stats":           stats,
            "target_track_id": UNIFIED_TARGET_ID,
            "target_team":     target_team,
            "fps":             fps,
        }
