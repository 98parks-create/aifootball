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

UNIFIED_TARGET_ID = -1

# ── 외모 지문(Fingerprint) 설정 ────────────────────────────────────────────────
FP_HIST_BINS   = (32, 32)   # H, S 채널
FP_TORSO_RATIO = 0.50       # bbox 상위 50% → 상체(유니폼)
FP_SHORTS_TOP  = 0.50       # bbox 중간~하위 30% → 반바지
FP_SHORTS_BOT  = 0.80
FP_UPDATE_MAX  = 30         # EMA 누적 최대 개수
FP_SWITCH_THR  = 0.30       # 이 이하면 ID 스위치 의심 → 즉시 재탐색
FP_RELOCK_HIGH = 0.58       # 즉시 재탐색 성공 기준 (엄격)
FP_RELOCK_LOW  = 0.38       # 장시간 소실 후 재탐색 기준 (완화)
FP_TEAM_PEN    = 0.15       # 상대팀 패널티 배율


class FootballDetector:
    def __init__(self, model_path='models/yolov8s.pt', scan_model_path=None):
        self.model = YOLO(model_path)
        self.model.to(_DEVICE)
        # 스캔 전용 경량 모델 (없으면 자동 다운로드)
        _scan_path = scan_model_path or 'models/yolov8n.pt'
        try:
            if not os.path.exists(_scan_path):
                print(f"[TAD] 경량 스캔 모델 다운로드 중: {_scan_path}")
            self.scan_model = YOLO(_scan_path)
            self.scan_model.to(_DEVICE)
            print(f"[TAD] 스캔 모델: {_scan_path}")
        except Exception as e:
            print(f"[TAD] 스캔 모델 로드 실패({e}), 기본 모델 사용")
            self.scan_model = self.model

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

        # ── 타겟 Fingerprint (EMA) ───────────────────────────────────────────────
        self._fp_torso  = None
        self._fp_shorts = None
        self._fp_count  = 0
        self._fp_bbox_h     = None
        self._fp_bbox_ratio = None
        # ── Golden Fingerprint (초기 40프레임 고정 기준점 — 줌/이동 불변 앵커) ──
        self._fp_torso_golden  = None
        self._fp_shorts_golden = None
        self._fp_golden_count  = 0
        FP_GOLDEN_FRAMES       = 40   # 몇 프레임까지 golden 수집
        self._FP_GOLDEN_FRAMES = FP_GOLDEN_FRAMES
        # ── 상태 ─────────────────────────────────────────────────────────────
        self._team_centers = None
        self._target_velocity = np.array([0.0, 0.0])
        # ID 스위치 연속 카운터 (단발성 오탐 무시용)
        self._switch_suspect_count = 0
        # ── 클러스터(선수 밀집) 상태 — 밀집 시 지문 오염 방지 ───────────────
        self._in_cluster = False
        self._cluster_frames = 0

    # ══════════════════════════════════════════════════════════════════════════
    # 히스토그램 추출 헬퍼
    # ══════════════════════════════════════════════════════════════════════════

    def _roi_hist(self, frame, bbox, top_r, bot_r):
        """bbox 내 [top_r ~ bot_r] 세로 범위에서 HSV 히스토그램 반환."""
        x1 = max(0, int(bbox[0])); y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1], int(bbox[2])); y2 = min(frame.shape[0], int(bbox[3]))
        h = y2 - y1
        if h <= 0 or x2 <= x1:
            return None
        r_y1 = y1 + int(h * top_r)
        r_y2 = y1 + int(h * bot_r)
        roi = frame[r_y1:r_y2, x1:x2]
        if roi.size == 0:
            return None
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                            list(FP_HIST_BINS), [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def _extract_torso_hist(self, frame, bbox):
        return self._roi_hist(frame, bbox, 0.0, FP_TORSO_RATIO)

    def _extract_shorts_hist(self, frame, bbox):
        return self._roi_hist(frame, bbox, FP_SHORTS_TOP, FP_SHORTS_BOT)

    @staticmethod
    def _cosine_sim(h1: np.ndarray, h2: np.ndarray) -> float:
        a = h1.flatten().astype(np.float64)
        b = h2.flatten().astype(np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / (denom + 1e-9))

    # ══════════════════════════════════════════════════════════════════════════
    # Fingerprint 업데이트 / 스코어
    # ══════════════════════════════════════════════════════════════════════════

    def _update_fingerprint(self, frame, bbox):
        """타겟의 외모 지문을 EMA로 갱신. 초기 40프레임은 golden(불변 기준점)도 함께 구축."""
        ht  = self._extract_torso_hist(frame, bbox)
        hs  = self._extract_shorts_hist(frame, bbox)
        bh  = float(bbox[3] - bbox[1])
        bw  = float(bbox[2] - bbox[0])
        ratio = bw / (bh + 1e-6)

        n = min(self._fp_count, FP_UPDATE_MAX)

        if ht is not None:
            if self._fp_torso is None:
                self._fp_torso = ht.copy()
            else:
                self._fp_torso = (self._fp_torso * n + ht) / (n + 1)
            # golden 수집 (초기 N프레임만, 이후 고정)
            if self._fp_golden_count < self._FP_GOLDEN_FRAMES:
                g = self._fp_golden_count
                if self._fp_torso_golden is None:
                    self._fp_torso_golden = ht.copy()
                else:
                    self._fp_torso_golden = (self._fp_torso_golden * g + ht) / (g + 1)

        if hs is not None:
            if self._fp_shorts is None:
                self._fp_shorts = hs.copy()
            else:
                self._fp_shorts = (self._fp_shorts * n + hs) / (n + 1)
            if self._fp_golden_count < self._FP_GOLDEN_FRAMES:
                g = self._fp_golden_count
                if self._fp_shorts_golden is None:
                    self._fp_shorts_golden = hs.copy()
                else:
                    self._fp_shorts_golden = (self._fp_shorts_golden * g + hs) / (g + 1)

        if ht is not None and self._fp_golden_count < self._FP_GOLDEN_FRAMES:
            self._fp_golden_count += 1

        if self._fp_bbox_h is None:
            self._fp_bbox_h    = bh
            self._fp_bbox_ratio = ratio
        else:
            self._fp_bbox_h     = (self._fp_bbox_h * n + bh) / (n + 1)
            self._fp_bbox_ratio = (self._fp_bbox_ratio * n + ratio) / (n + 1)

        self._fp_count += 1

    def _fingerprint_score(self, frame, bbox) -> float:
        """
        외모 지문 유사도 (0~1).
        EMA 지문 65% + Golden(초기 앵커) 35% 결합.
        bbox 크기 패널티 없음 — 카메라 줌인/아웃 시 오탐 방지.
        """
        if self._fp_torso is None:
            return 0.5

        ht = self._extract_torso_hist(frame, bbox)
        hs = self._extract_shorts_hist(frame, bbox)

        # EMA 지문 스코어
        s_torso_ema  = self._cosine_sim(self._fp_torso, ht) if ht is not None else 0.0
        s_shorts_ema = (self._cosine_sim(self._fp_shorts, hs)
                        if (hs is not None and self._fp_shorts is not None) else 0.0)
        if self._fp_shorts is not None and hs is not None:
            score_ema = 0.70 * s_torso_ema + 0.30 * s_shorts_ema
        else:
            score_ema = s_torso_ema

        # Golden 지문 스코어 (있을 때만)
        if self._fp_torso_golden is not None and self._fp_golden_count >= 10:
            s_torso_g  = self._cosine_sim(self._fp_torso_golden, ht) if ht is not None else 0.0
            s_shorts_g = (self._cosine_sim(self._fp_shorts_golden, hs)
                          if (hs is not None and self._fp_shorts_golden is not None) else 0.0)
            if self._fp_shorts_golden is not None and hs is not None:
                score_golden = 0.70 * s_torso_g + 0.30 * s_shorts_g
            else:
                score_golden = s_torso_g
            score = 0.65 * score_ema + 0.35 * score_golden
        else:
            score = score_ema

        return float(np.clip(score, 0.0, 1.0))

    def _is_id_switch(self, frame, bbox) -> bool:
        """
        현재 tracker ID 에 할당된 bbox 가 우리 타겟과 외모가 너무 다르면
        ByteTrack ID 스위치 의심 → True 반환.
        지문이 충분히 쌓인 후에만 판정하며, 3프레임 연속으로 낮아야 스위치로 확정.
        카메라 줌인/아웃 시 일시적 점수 하락으로 인한 오탐 방지.
        """
        if self._fp_count < 20:
            self._switch_suspect_count = 0
            return False
        score = self._fingerprint_score(frame, bbox)
        if score < FP_SWITCH_THR:
            self._switch_suspect_count += 1
        else:
            self._switch_suspect_count = 0
        return self._switch_suspect_count >= 3

    # ══════════════════════════════════════════════════════════════════════════
    # 팀 클러스터링
    # ══════════════════════════════════════════════════════════════════════════

    def _cluster_teams_init(self, frame, dets):
        hists = []
        for i in range(len(dets)):
            if dets.class_id[i] != 0:
                continue
            h = self._extract_torso_hist(frame, dets.xyxy[i])
            if h is not None:
                hists.append(h.flatten().astype(np.float32))
        if len(hists) < 4:
            self._team_centers = None
            return
        data = np.array(hists)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, _, centers = cv2.kmeans(
            data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        self._team_centers = centers
        print(f"[TAD] 팀 클러스터링 완료: {len(hists)}명 샘플")

    def _get_team(self, frame, bbox) -> int:
        if self._team_centers is None:
            return -1
        h = self._extract_torso_hist(frame, bbox)
        if h is None:
            return -1
        hf    = h.flatten().astype(np.float32)
        dists = [float(np.linalg.norm(hf - self._team_centers[k])) for k in range(2)]
        return int(np.argmin(dists))

    # ══════════════════════════════════════════════════════════════════════════
    # 재탐색 (Re-ID) 핵심 로직
    # ══════════════════════════════════════════════════════════════════════════

    def _find_best_candidate(self, frame, dets, last_px, frames_lost,
                             scan_w, scan_h, target_team, tid_team):
        """
        소실 기간에 따라 전략을 다르게 적용:
        - 단기 소실 (< 30f): 위치만 사용 (같은 유니폼 색 구별 불가)
        - 중기 소실 (30~100f): 위치 70% + 외모 30%
        - 장기 소실 (> 100f): 위치 40% + 외모 60%
        """
        best_i, best_score = None, -1.0

        # 소실 단계별 설정
        if frames_lost < 30:
            # 단기: 위치 반경 120px 이내에서만 탐색, 외모 무시
            MAX_D      = 120.0
            pos_weight = 1.0
            app_weight = 0.0
        elif frames_lost < 100:
            # 중기: 반경 220px, 위치 우선
            MAX_D      = 220.0
            pos_weight = 0.70
            app_weight = 0.30
        else:
            # 장기: 반경 확대, 외모 비중 증가
            MAX_D      = min(300 + frames_lost, 500)
            pos_weight = 0.40
            app_weight = 0.60

        for i in range(len(dets)):
            if dets.class_id[i] != 0:
                continue

            bbox   = dets.xyxy[i]
            cx     = (bbox[0] + bbox[2]) / 2
            cy     = (bbox[1] + bbox[3]) / 2
            t_id   = int(dets.tracker_id[i]) if dets.tracker_id is not None else -1
            t_team = tid_team.get(t_id, -1)

            # ── 위치 스코어 ─────────────────────────────────────────────
            pos_score = 0.0
            if last_px is not None:
                pred = np.array(last_px) + self._target_velocity * min(frames_lost, 60)
                pred = np.clip(pred, [0, 0], [scan_w, scan_h])
                d    = np.hypot(cx - pred[0], cy - pred[1])
                if d > MAX_D:
                    continue  # 반경 밖은 아예 제외
                pos_score = max(0.0, 1.0 - d / MAX_D)
            elif app_weight == 0.0:
                continue  # 위치 정보 없고 단기 모드면 스킵

            # ── 외모 스코어 ─────────────────────────────────────────────
            app_score = 0.0
            if app_weight > 0:
                app_score = self._fingerprint_score(frame, bbox)
                # 상대팀 패널티
                if target_team != -1 and t_team != -1 and t_team != target_team:
                    app_score *= FP_TEAM_PEN

            score = pos_weight * pos_score + app_weight * app_score

            if score > best_score:
                best_score = score
                best_i     = i

        return best_i, best_score

    # ══════════════════════════════════════════════════════════════════════════
    # 갭 보간
    # ══════════════════════════════════════════════════════════════════════════

    def _interpolate_gaps(self, frames, fps, max_gap_sec=1.0):
        if len(frames) < 2:
            return frames
        max_gap = int(fps * max_gap_sec)
        result  = list(frames)
        for i in range(len(frames) - 1):
            f1, f2 = frames[i]['frame'], frames[i+1]['frame']
            gap    = f2 - f1
            if gap <= 1 or gap > max_gap:
                continue
            p1, p2   = frames[i]['pos'],    frames[i+1]['pos']
            px1, px2 = frames[i]['pos_px'], frames[i+1]['pos_px']
            for f in range(f1 + 1, f2):
                a = (f - f1) / gap
                result.append({
                    "frame":  f,
                    "pos":    [p1[0]  + a*(p2[0]-p1[0]),   p1[1]  + a*(p2[1]-p1[1])],
                    "pos_px": [px1[0] + a*(px2[0]-px1[0]), px1[1] + a*(px2[1]-px1[1])],
                    "class":  0, "conf": 0.0,
                    "bbox":   frames[i]['bbox'],
                    "interpolated": True,
                })
        result.sort(key=lambda x: x['frame'])
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    def detect_players_for_selection(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return []
        results = self.model(frame, conf=0.2, verbose=False)[0]
        dets    = sv.Detections.from_ultralytics(results)
        dets    = dets[(dets.class_id == 0) | (dets.class_id == 32)]
        players = []
        for i, (xyxy, _, conf, class_id, _, _) in enumerate(dets):
            x1, y1, x2, y2 = xyxy
            players.append({
                "id":    i,
                "class": self.model.model.names[int(class_id)],
                "x":     float((x1 + x2) / 2),
                "y":     float(y2),
                "bbox":  [float(x1), float(y1), float(x2), float(y2)],
            })
        return players

    def process_video_v2(self, input_path, output_path, calibration_points,
                         target_player_id, progress_callback=None):
        """
        2-Stage 처리:
          Stage-1 (0-100%): 480p 스캔 — 타겟 추적 데이터 수집
          Stage-2: app.py 에서 원본 해상도로 하이라이트 클립 추출
        """
        print(f"[TAD] Stage-1 480p 스캔 시작: {input_path}")

        info         = sv.VideoInfo.from_video_path(input_path)
        total_frames = max(info.total_frames, 1)
        fps          = info.fps or 25
        orig_w       = info.width  or 1280
        orig_h       = info.height or 720

        TARGET_H   = 480
        scan_scale = min(TARGET_H / orig_h, 1.0)
        scan_w     = max(2, int(orig_w * scan_scale) - (int(orig_w * scan_scale) % 2))
        scan_h     = max(2, int(orig_h * scan_scale) - (int(orig_h * scan_scale) % 2))
        print(f"[TAD] 원본 {orig_w}×{orig_h} → 스캔 {scan_w}×{scan_h} (scale={scan_scale:.2f})")

        # 캘리브레이션 스케일 변환
        scaled_calib  = [[p[0] * scan_scale, p[1] * scan_scale] for p in calibration_points]
        scaled_target = {
            'x': target_player_id['x'] * scan_scale,
            'y': target_player_id['y'] * scan_scale,
        }
        src         = np.array(scaled_calib, dtype=np.float32)
        dst         = np.array([[0,0],[100,0],[100,50],[0,50]], dtype=np.float32)
        transformer = ViewTransformer(src, dst)

        _cfg_path = os.path.join(os.path.dirname(__file__), '..', 'bytetrack.yaml')
        _tracker  = _cfg_path if os.path.exists(_cfg_path) else 'bytetrack.yaml'

        cap = cv2.VideoCapture(input_path)

        # ── 팀 색상 클러스터링 (첫 프레임) ──────────────────────────────────
        ok, first_frame = cap.read()
        if ok:
            small0 = cv2.resize(first_frame, (scan_w, scan_h))
            r0     = self.scan_model(small0, conf=0.25, verbose=False)[0]
            d0     = sv.Detections.from_ultralytics(r0)
            self._cluster_teams_init(small0, d0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ── 타겟 선수 사전 지문 추출 ──────────────────────────────────────────
        # 사용자가 선택한 정확한 bbox(yolov8s, 풀해상도)로 HSV 지문을 미리 구축.
        # 이후 추적 루프는 이 지문을 기준으로 초기 락온 + ID 스위치 판단에 활용한다.
        _pre_fp_ok = False
        _pre_team  = -1
        if ok and 'bbox' in target_player_id:
            orig_bbox = target_player_id['bbox']        # [x1,y1,x2,y2] 풀해상도
            sbbox = np.array([
                orig_bbox[0] * scan_scale,
                orig_bbox[1] * scan_scale,
                orig_bbox[2] * scan_scale,
                orig_bbox[3] * scan_scale,
            ], dtype=np.float32)
            self._update_fingerprint(first_frame if scan_scale >= 1.0
                                     else cv2.resize(first_frame, (scan_w, scan_h)), sbbox)
            # golden 즉시 고정 (이후 오염 방지)
            if self._fp_torso is not None:
                self._fp_torso_golden  = self._fp_torso.copy()
            if self._fp_shorts is not None:
                self._fp_shorts_golden = self._fp_shorts.copy()
            self._fp_golden_count = self._FP_GOLDEN_FRAMES
            # 팀 색상 사전 결정
            _pre_team = self._get_team(
                cv2.resize(first_frame, (scan_w, scan_h)), sbbox)
            _pre_fp_ok = True
            print(f"[TAD] 사전 지문 추출 완료 — team={_pre_team}")

        # ── 추적 상태 변수 ────────────────────────────────────────────────────
        target_id    = None
        last_px      = None
        frames_lost  = 0
        target_team  = _pre_team if _pre_fp_ok else -1
        tid_team     = {}
        target_frames = []

        # 격 프레임 스킵 설정 (안정 추적 중) — 4에서 2로 줄여야 ByteTrack ID 안정성 향상
        STABLE_SKIP = 2
        # 소실 허용 최대 프레임 수 (이후 완화 임계값 사용)
        LONG_LOST   = int(fps * 4)

        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 안정 추적 중 격 프레임 스킵 (처리 속도 향상)
            # ByteTrack에는 모든 프레임 전달 (ID 안정성), 데이터 수집만 스킵
            if target_id is not None and frames_lost == 0 and idx % STABLE_SKIP != 0:
                _small_skip = cv2.resize(frame, (scan_w, scan_h))
                try:
                    self.scan_model.track(
                        _small_skip, persist=True, conf=0.2,
                        classes=[0, 32], verbose=False, tracker=_tracker,
                    )
                except Exception:
                    pass
                idx += 1
                continue

            if progress_callback and idx % 8 == 0:
                progress_callback((idx / total_frames) * 100)

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

                    # ── 클러스터 상태 업데이트 (이전 프레임 타겟 위치 기준) ──────
                    # 타겟 100px 반경 내 선수가 3명 이상이면 밀집 상태로 판단.
                    # 밀집 중에는 지문 업데이트와 외모 재탐색을 차단해 오염 방지.
                    if last_px is not None:
                        nearby_cnt = sum(
                            1 for j in range(len(dets))
                            if dets.class_id[j] == 0
                            and np.hypot(
                                (dets.xyxy[j][0] + dets.xyxy[j][2]) / 2 - last_px[0],
                                (dets.xyxy[j][1] + dets.xyxy[j][3]) / 2 - last_px[1]
                            ) < 100.0
                        )
                        was_cluster = self._in_cluster
                        self._in_cluster = (nearby_cnt >= 3)
                        if self._in_cluster:
                            self._cluster_frames += 1
                            if not was_cluster:
                                print(f"[TAD] 클러스터 진입 f={idx} ({nearby_cnt}명 밀집)")
                        else:
                            if was_cluster:
                                print(f"[TAD] 클러스터 해소 f={idx}, "
                                      f"{self._cluster_frames}프레임 유지됨")
                            self._cluster_frames = 0
                    else:
                        self._in_cluster = False

                    # ══════════════════════════════════════════════════════
                    # STEP 1 : 현재 tracker_id 유지 & ID 스위치 감지
                    # ══════════════════════════════════════════════════════
                    if target_id is not None:
                        where = np.where(dets.tracker_id == target_id)[0]
                        if len(where):
                            idx_t = where[0]
                            xyxy  = dets.xyxy[idx_t]

                            if self._is_id_switch(small, xyxy):
                                if self._in_cluster:
                                    # 클러스터 밀집 중 ID 스위치 → 재탐색 건너뜀
                                    # 같은 유니폼 선수가 많아 재탐색 시 오매칭 위험
                                    # 속도 벡터 유지하며 현재 위치만 업데이트
                                    print(f"[TAD] 클러스터 중 스위치 의심 f={idx} → 위치 유지")
                                    new_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                                    if last_px is not None:
                                        vel = np.array(new_px) - np.array(last_px)
                                        self._target_velocity = (
                                            0.75 * self._target_velocity + 0.25 * vel)
                                    last_px = new_px
                                    found       = True
                                    found_i     = idx_t
                                    frames_lost = 0
                                    # 지문 업데이트 생략 (오염 방지)
                                else:
                                    # ByteTrack이 다른 선수에게 ID를 줬을 가능성
                                    print(f"[TAD] ⚠ ID 스위치 의심 f={idx}, 외모 재탐색 시작")
                                    # 현재 ID 무시하고 외모 기반 재탐색
                                    best_i, best_score = self._find_best_candidate(
                                        small, dets, last_px, 1,
                                        scan_w, scan_h, target_team, tid_team)
                                    if best_i is not None and best_score >= FP_RELOCK_HIGH:
                                        # 새 후보 채택
                                        target_id = int(dets.tracker_id[best_i])
                                        found_i   = best_i
                                        xyxy      = dets.xyxy[best_i]
                                        new_px    = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                                        self._target_velocity = (
                                            np.array(new_px) - np.array(last_px)
                                        ) if last_px else np.array([0.0, 0.0])
                                        last_px   = new_px
                                        self._update_fingerprint(small, xyxy)
                                        found = True
                                        frames_lost = 0
                                        self._switch_suspect_count = 0
                                        print(f"[TAD] 스위치 복구 → id={target_id} score={best_score:.2f}")
                                    else:
                                        # 복구 실패 → 원래 ID 유지 (오염 방지)
                                        new_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                                        if last_px is not None:
                                            vel = np.array(new_px) - np.array(last_px)
                                            self._target_velocity = (
                                                0.75 * self._target_velocity + 0.25 * vel)
                                        last_px = new_px
                                        found       = True
                                        found_i     = idx_t
                                        frames_lost = 0
                            else:
                                # 정상 추적 유지
                                self._switch_suspect_count = 0
                                new_px = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                                if last_px is not None:
                                    vel = np.array(new_px) - np.array(last_px)
                                    self._target_velocity = (
                                        0.75 * self._target_velocity + 0.25 * vel)
                                last_px = new_px
                                # 클러스터(밀집) 중에는 지문 업데이트 생략 (오염 방지)
                                if not self._in_cluster:
                                    self._update_fingerprint(small, xyxy)
                                found       = True
                                found_i     = idx_t
                                frames_lost = 0

                    # ══════════════════════════════════════════════════════
                    # STEP 2 : 소실 → 위치+외모 재탐색
                    # ══════════════════════════════════════════════════════
                    if not found and (target_id is not None or self._fp_torso is not None):
                        # 단기 소실: 위치만으로 판단 (같은 팀 유니폼 색 구별 불가)
                        # 중/장기 소실: 외모 점점 반영
                        if frames_lost < 30:
                            threshold = 0.55  # 단기 소실 재탐색 기준 강화 (오매칭 방지)
                        elif frames_lost >= LONG_LOST:
                            threshold = FP_RELOCK_LOW
                        else:
                            threshold = FP_RELOCK_HIGH

                        best_i, best_score = self._find_best_candidate(
                            small, dets, last_px, frames_lost,
                            scan_w, scan_h, target_team, tid_team)

                        if best_i is not None and best_score >= threshold:
                            new_id  = int(dets.tracker_id[best_i])
                            xyxy    = dets.xyxy[best_i]
                            new_px  = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            vel     = (np.array(new_px) - np.array(last_px)) / max(frames_lost, 1) \
                                      if last_px else np.array([0.0, 0.0])
                            self._target_velocity = 0.6 * self._target_velocity + 0.4 * vel
                            target_id   = new_id
                            found_i     = best_i
                            last_px     = new_px
                            # 신뢰도 높을 때만 지문 업데이트 (오염 방지)
                            if best_score >= 0.65:
                                self._update_fingerprint(small, xyxy)
                            found       = True
                            frames_lost = 0
                            print(f"[TAD] Re-lock f={idx} id={target_id} "
                                  f"score={best_score:.2f} thr={threshold:.2f}")

                    # ══════════════════════════════════════════════════════
                    # STEP 3 : 최초 락온
                    # 지문 있음 → 지문(60%) + 위치(40%) 결합
                    # 지문 없음 → 순수 좌표 기반 (fallback)
                    # ══════════════════════════════════════════════════════
                    if target_id is None:
                        best_i, best_score = None, 0.0
                        INIT_RADIUS = max(280.0 * scan_scale, 180.0)

                        for i in range(len(dets)):
                            if dets.class_id[i] != 0:
                                continue
                            xyxy = dets.xyxy[i]
                            cx   = (xyxy[0]+xyxy[2])/2
                            cy   = float(xyxy[3])
                            d    = np.hypot(cx - scaled_target['x'],
                                            cy - scaled_target['y'])
                            if d > INIT_RADIUS:
                                continue
                            pos_score = max(0.0, 1.0 - d / INIT_RADIUS)

                            if self._fp_torso is not None:
                                fp_score = self._fingerprint_score(small, xyxy)
                                score    = 0.60 * fp_score + 0.40 * pos_score
                            else:
                                score = pos_score

                            if score > best_score:
                                best_score, best_i = score, i

                        if best_i is not None and best_score >= (0.40 if self._fp_torso is not None else 0.20):
                            target_id = int(dets.tracker_id[best_i])
                            found_i   = best_i
                            xyxy      = dets.xyxy[best_i]
                            last_px   = [(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2]
                            self._target_velocity = np.array([0.0, 0.0])
                            self._update_fingerprint(small, xyxy)
                            if target_team == -1:
                                target_team = self._get_team(small, xyxy)
                            found = True
                            self._switch_suspect_count = 0
                            print(f"[TAD] 초기 락온 id={target_id} score={best_score:.2f}")

                    if not found:
                        frames_lost += 1
                        if frames_lost % 30 == 0:
                            print(f"[TAD] 타겟 소실 {frames_lost}프레임 (f={idx})")

                    # ══════════════════════════════════════════════════════
                    # 모든 트랙 저장
                    # ══════════════════════════════════════════════════════
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

                    # ══════════════════════════════════════════════════════
                    # 타겟 통합 트랙 추가
                    # ══════════════════════════════════════════════════════
                    if found and found_i is not None:
                        xyxy = dets.xyxy[found_i]
                        if target_team == -1:
                            target_team = self._get_team(small, xyxy)
                            print(f"[TAD] 타겟 팀: {target_team}")
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

        # ── pos_px 480p → 원본 해상도 역스케일 ──────────────────────────────
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

        total_tracked = len(target_frames)
        coverage = total_tracked / max(total_frames // STABLE_SKIP, 1) * 100
        print(f"[TAD] Stage-1 완료. target_id={target_id}, "
              f"unified={total_tracked}, 커버리지≈{coverage:.1f}%, "
              f"team={target_team}, players={len(self.player_tracks)}")

        stats = self.analyzer.calculate_individual_stats(
            self.player_tracks, UNIFIED_TARGET_ID, fps=fps)
        return {
            "stats":           stats,
            "target_track_id": UNIFIED_TARGET_ID,
            "target_team":     target_team,
            "fps":             fps,
        }
