import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
import cv2

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Normalized field (100x50 units) mapped to real dimensions (~105x68m)
SCALE_X = 105.0 / 100.0
SCALE_Y = 68.0 / 50.0


def _put_centered_text(frame, text, y, font_scale, color, thickness):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, (frame.shape[1] - tw) // 2)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


class FootballAnalyzer:
    def __init__(self):
        self.base_storage = r"D:\aifootball_data"
        self.data_dir = os.path.join(self.base_storage, "analysis")
        self.highlight_dir = os.path.join(self.base_storage, "highlights")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.highlight_dir, exist_ok=True)

    def _calc_speed_kmh(self, pos1, pos2, fps):
        dx = (pos2[0] - pos1[0]) * SCALE_X
        dy = (pos2[1] - pos1[1]) * SCALE_Y
        dist_m = np.sqrt(dx ** 2 + dy ** 2)
        return dist_m * fps * 3.6

    def calculate_individual_stats(self, tracks, target_id, position="ST", fps=25):
        if not tracks or target_id not in tracks or not tracks[target_id]:
            return {"PAC": 50, "PHY": 50, "DRI": 50, "PAS": 50, "SHO": 50,
                    "total_distance_km": 0, "top_speed_kmh": 0, "sprint_count": 0}

        target_data = tracks[target_id]
        positions = [t['pos'] for t in target_data]

        distance_m = 0.0
        speeds = []
        for i in range(1, len(positions)):
            spd = self._calc_speed_kmh(positions[i - 1], positions[i], fps)
            if spd < 40:  # filter tracking noise
                dx = (positions[i][0] - positions[i - 1][0]) * SCALE_X
                dy = (positions[i][1] - positions[i - 1][1]) * SCALE_Y
                distance_m += np.sqrt(dx ** 2 + dy ** 2)
                speeds.append(spd)

        top_speed = max(speeds) if speeds else 0
        avg_speed = float(np.mean(speeds)) if speeds else 0
        total_km = distance_m / 1000
        sprint_count = sum(1 for s in speeds if s > 20)

        pos = position.upper()
        pac = min(99, max(55, int(top_speed * 2.8)))
        phy = min(99, max(55, int(total_km * 12)))

        if any(p in pos for p in ["ST", "FW", "CF", "LW", "RW", "SS"]):
            sho = min(99, max(70, 70 + int(top_speed * 0.4)))
            dri = min(99, max(70, 72 + int(avg_speed * 1.5)))
            pas = min(99, max(65, 68))
        elif any(p in pos for p in ["MF", "CM", "AM", "DM", "CAM", "CDM"]):
            sho = 70
            dri = min(99, max(76, 78 + int(avg_speed * 1.2)))
            pas = min(99, max(76, 78 + int(total_km * 3)))
        else:  # DF / GK
            sho = 58
            dri = 66
            phy = min(99, phy + 10)
            pas = min(99, max(66, 70))

        return {
            "PAC": pac, "PHY": phy, "DRI": dri, "PAS": pas, "SHO": sho,
            "total_distance_km": round(total_km, 2),
            "top_speed_kmh": round(top_speed, 1),
            "sprint_count": sprint_count,
        }

    def generate_pitch_heatmap(self, tracks, target_id, session_id):
        if not tracks or target_id not in tracks:
            return None
        try:
            positions = np.array([t['pos'] for t in tracks[target_id]])
            heatmap, _, _ = np.histogram2d(
                positions[:, 0], positions[:, 1],
                bins=[60, 30], range=[[0, 100], [0, 50]]
            )
            hm = gaussian_filter(heatmap.T, sigma=2.5) if HAS_SCIPY else heatmap.T

            fig = Figure(figsize=(12, 6), facecolor='none')
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.set_facecolor('none')
            ax.imshow(hm, origin='lower', extent=[0, 100, 0, 50],
                      cmap='hot', interpolation='bilinear', alpha=0.88)
            ax.axis('off')

            filename = f"{session_id}_heatmap.png"
            fig.savefig(os.path.join(self.highlight_dir, filename),
                        bbox_inches='tight', transparent=True, dpi=150, pad_inches=0)
            plt.close('all')
            return f"/static/highlights/{filename}"
        except Exception as e:
            print(f"Heatmap Error: {e}")
            return None

    def generate_ai_comment(self, stats):
        report = {"pros": [], "cons": [], "scouting_note": ""}
        spd = stats.get('top_speed_kmh', 0)
        dist = stats.get('total_distance_km', 0)
        sprints = stats.get('sprint_count', 0)

        if spd >= 28:
            report['pros'].append(f"엘리트급 최고속도 {spd:.1f}km/h — 프로 레벨 상위 5% 스프린트")
        elif spd >= 23:
            report['pros'].append(f"우수한 순간 가속력 ({spd:.1f}km/h) — 수비 배후 공략 가능")
        elif spd >= 18:
            report['pros'].append(f"안정적인 이동 속도 ({spd:.1f}km/h) — 포지셔닝 유지 우수")

        if dist >= 8:
            report['pros'].append(f"왕성한 활동량 {dist:.1f}km — 팀의 엔진 역할 수행")
        elif dist >= 5:
            report['pros'].append(f"적정 이동거리 {dist:.1f}km — 포지션 밸런스 유지")

        if sprints >= 10:
            report['pros'].append(f"스프린트 {sprints}회 — 공격·수비 전환 시 폭발적 전진력")

        if stats.get('PAC', 0) >= 85:
            report['pros'].append("볼 없이도 공간을 위협하는 오프 더 볼 무브먼트")
        if stats.get('DRI', 0) >= 80:
            report['pros'].append("압박 상황에서도 흔들리지 않는 볼 킵 능력")

        if spd < 18:
            report['cons'].append("역습 시 스프린트 속도 향상 훈련 권장 (목표: 22km/h)")
        if dist < 4:
            report['cons'].append("전·후방 전환 시 이동거리 증가 — 체력 훈련 필요")
        if stats.get('PHY', 0) < 65:
            report['cons'].append("경기 후반 체력 유지력 강화 루틴 추가 권장")

        if len(report['pros']) >= 2:
            report['scouting_note'] = (
                f"TAD AI 측정 결과 최고속도 {spd:.1f}km/h, 이동거리 {dist:.1f}km. "
                f"실전 투입 즉시 팀에 기여할 수 있는 즉전감으로 평가됩니다."
            )
        elif len(report['pros']) == 1:
            report['scouting_note'] = (
                "기본기가 탄탄한 선수입니다. 강점을 더욱 발전시키면 "
                "한 단계 높은 레벨에서도 충분히 경쟁 가능합니다."
            )
        else:
            report['scouting_note'] = (
                "TAD AI는 이 선수의 성실함과 잠재력에 주목합니다. "
                "지속적인 데이터 축적을 통해 성장 궤적을 추적하겠습니다."
            )
        return report

    def _detect_highlight_events(self, tracks, target_id, ball_dict, fps=25):
        target_tracks = tracks[target_id]
        if len(target_tracks) < 2:
            return []

        # Build per-frame speed map
        speed_map = {}
        for i in range(1, len(target_tracks)):
            f = target_tracks[i]['frame']
            spd = self._calc_speed_kmh(target_tracks[i - 1]['pos'], target_tracks[i]['pos'], fps)
            if spd < 40:
                speed_map[f] = spd

        events = []
        min_gap = int(fps * 8)
        last_event = -min_gap

        # Sprint events (> 22 km/h)
        for i in range(1, len(target_tracks)):
            f = target_tracks[i]['frame']
            spd = speed_map.get(f, 0)
            if spd > 22 and f - last_event > min_gap:
                events.append({"frame": f, "category": "Sprint", "score": spd * 1.5, "speed_kmh": round(spd, 1)})
                last_event = f

        # Ball touch events (< 3m proximity)
        last_touch = -int(fps * 6)
        for t in target_tracks:
            f = t['frame']
            if f not in ball_dict:
                continue
            p = t.get('pos', [0, 0])
            dist = np.sqrt((p[0] - ball_dict[f][0]) ** 2 + (p[1] - ball_dict[f][1]) ** 2)
            if dist < 3.0 and f - last_touch > fps * 6:
                spd = speed_map.get(f, 0)
                events.append({"frame": f, "category": "Ball Action", "score": 200 - dist * 10, "speed_kmh": round(spd, 1)})
                last_touch = f

        # Activity fallback
        if len(events) < 4:
            window = int(fps * 6)
            for i in range(0, len(target_tracks) - window, window):
                f = target_tracks[i + window // 2]['frame']
                if any(abs(f - e['frame']) < fps * 8 for e in events):
                    continue
                seg_dist = sum(
                    self._calc_speed_kmh(target_tracks[j - 1]['pos'], target_tracks[j]['pos'], fps)
                    for j in range(i + 1, min(i + window, len(target_tracks)))
                    if self._calc_speed_kmh(target_tracks[j - 1]['pos'], target_tracks[j]['pos'], fps) < 40
                )
                events.append({"frame": f, "category": "Active Play", "score": seg_dist, "speed_kmh": 0})

        events = sorted(events, key=lambda x: x['score'], reverse=True)[:10]
        events = sorted(events, key=lambda x: x['frame'])
        return events

    def extract_combined_highlights(self, video_path, tracks, target_id, fps=25):
        if target_id not in tracks:
            return [], []

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            cap.release()
        except Exception:
            fps = 25

        target_dict = {t['frame']: t['pos_px'] for t in tracks[target_id] if 'pos_px' in t}
        ball_tracks = [t for tid, data in tracks.items()
                       if data and data[0].get('class') == 32 for t in data]
        ball_dict = {b['frame']: b['pos'] for b in ball_tracks}

        events = self._detect_highlight_events(tracks, target_id, ball_dict, fps)

        highlights = []
        pre = int(fps * 3)
        post = int(fps * 3)

        for i, ev in enumerate(events):
            start_f = max(0, int(ev['frame'] - pre))
            end_f = int(ev['frame'] + post)
            clip_name = f"tad_hl_{target_id}_{i}.mp4"
            clip_path = os.path.join(self.highlight_dir, clip_name)
            seg_tracks = {f: target_dict[f] for f in range(start_f, end_f + 1) if f in target_dict}
            if self._save_cinematic_clip(video_path, start_f, end_f, clip_path, seg_tracks, ev, fps):
                highlights.append({
                    "url": f"/static/highlights/{clip_name}",
                    "category": ev['category'],
                    "frame": int(ev['frame']),
                    "speed_kmh": ev.get("speed_kmh", 0),
                })

        return highlights, events

    def _make_title_frame(self, w, h, line1, line2=""):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for row in range(h):
            v = int(25 * (1 - row / h))
            frame[row, :] = [v, 0, 0]
        _put_centered_text(frame, "TAD AI", h // 2 - 40, 2.2, (220, 40, 40), 4)
        _put_centered_text(frame, line1.upper(), h // 2 + 20, 1.0, (255, 215, 0), 2)
        if line2:
            _put_centered_text(frame, line2.upper(), h // 2 + 65, 0.65, (180, 180, 180), 1)
        return frame

    def _save_cinematic_clip(self, video_path, start_f, end_f, out_path, tracks, event=None, fps=25):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            out_w, out_h = 1280, 720

            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (out_w, out_h))
            if not out.isOpened():
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))

            cat = (event.get('category', 'HIGHLIGHT') if event else 'HIGHLIGHT').upper()
            spd_str = f"{event.get('speed_kmh', 0):.0f} KM/H" if event and event.get('speed_kmh', 0) > 1 else ""
            is_action = event and event.get('category') in ('Ball Action', 'Sprint')

            # Title card fade-in (0.4s)
            title = self._make_title_frame(out_w, out_h, cat, spd_str)
            title_n = int(fps * 0.4)
            for k in range(title_n):
                a = min(1.0, k / max(1, fps * 0.2))
                out.write((title * a).astype(np.uint8))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            total = max(end_f - start_f, 1)
            lx = ly = None

            for f_idx in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret:
                    break
                h_f, w_f = frame.shape[:2]

                raw = tracks.get(f_idx)
                cx = float(raw[0]) if raw is not None else w_f / 2
                cy = float(raw[1]) if raw is not None else h_f / 2

                if lx is not None:
                    cx, cy = lx * 0.85 + cx * 0.15, ly * 0.85 + cy * 0.15
                lx, ly = cx, cy

                cs = min(w_f, h_f, 960)
                x1 = int(max(0, min(w_f - cs, cx - cs / 2)))
                y1 = int(max(0, min(h_f - cs, cy - cs / 2)))
                res = cv2.resize(frame[y1:y1 + cs, x1:x1 + cs], (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                prog = (f_idx - start_f) / total
                if prog < 0.1:
                    res = (res * (prog / 0.1)).astype(np.uint8)
                elif prog > 0.9:
                    res = (res * ((1 - prog) / 0.1)).astype(np.uint8)

                # Overlays
                cv2.putText(res, "TAD AI", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 40, 40), 2, cv2.LINE_AA)
                cv2.putText(res, cat, (24, out_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2, cv2.LINE_AA)
                if spd_str:
                    (sw, _), _ = cv2.getTextSize(spd_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(res, spd_str, (out_w - sw - 24, out_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Slow-mo at event peak (middle 30%)
                if is_action and 0.35 < prog < 0.65:
                    out.write(res)  # duplicate frame → 0.5x speed

                out.write(res)

            cap.release()
            out.release()
            return os.path.exists(out_path) and os.path.getsize(out_path) > 1000
        except Exception as e:
            print(f"Clip Error: {e}")
            import traceback; traceback.print_exc()
            return False

    def generate_master_sizzle_reel(self, video_path, events, tracks, target_id, session_id, player_name="PLAYER"):
        if not events or target_id not in tracks:
            return None
        target_dict = {t['frame']: t['pos_px'] for t in tracks[target_id] if 'pos_px' in t}
        output_name = f"{session_id}_tad_master.mp4"
        output_path = os.path.join(self.highlight_dir, output_name)

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            out_w, out_h = 1280, 720

            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (out_w, out_h))
            if not out.isOpened():
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))

            # Intro (1.5s)
            intro = self._make_title_frame(out_w, out_h, player_name, "OFFICIAL HIGHLIGHT FILM")
            for k in range(int(fps * 1.5)):
                a = min(1.0, k / max(1, fps * 0.5))
                out.write((intro * a).astype(np.uint8))

            pre = int(fps * 3)
            post = int(fps * 3)
            lx = ly = None

            for ev in events:
                start_f = max(0, int(ev['frame'] - pre))
                end_f = int(ev['frame'] + post)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                total = max(end_f - start_f, 1)
                cat = ev.get('category', 'HIGHLIGHT').upper()
                spd_str = f"{ev.get('speed_kmh', 0):.0f} KM/H" if ev.get('speed_kmh', 0) > 1 else ""
                is_action = ev.get('category') in ('Ball Action', 'Sprint')

                for f_idx in range(start_f, end_f):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    h_f, w_f = frame.shape[:2]
                    raw = target_dict.get(f_idx)
                    cx = float(raw[0]) if raw is not None else w_f / 2
                    cy = float(raw[1]) if raw is not None else h_f / 2
                    if lx is not None:
                        cx, cy = lx * 0.85 + cx * 0.15, ly * 0.85 + cy * 0.15
                    lx, ly = cx, cy

                    cs = min(w_f, h_f, 960)
                    x1 = int(max(0, min(w_f - cs, cx - cs / 2)))
                    y1 = int(max(0, min(h_f - cs, cy - cs / 2)))
                    res = cv2.resize(frame[y1:y1 + cs, x1:x1 + cs], (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

                    prog = (f_idx - start_f) / total
                    if prog < 0.08:
                        res = (res * (prog / 0.08)).astype(np.uint8)
                    elif prog > 0.92:
                        res = (res * ((1 - prog) / 0.08)).astype(np.uint8)

                    cv2.putText(res, "TAD AI", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220, 40, 40), 2, cv2.LINE_AA)
                    cv2.putText(res, cat, (24, out_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2, cv2.LINE_AA)
                    (pnw, _), _ = cv2.getTextSize(player_name.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(res, player_name.upper(), (out_w - pnw - 24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2, cv2.LINE_AA)
                    if spd_str:
                        (sw, _), _ = cv2.getTextSize(spd_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.putText(res, spd_str, (out_w - sw - 24, out_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    if is_action and 0.35 < prog < 0.65:
                        out.write(res)

                    out.write(res)

                # Cut transition (0.15s black)
                black = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                for _ in range(int(fps * 0.15)):
                    out.write(black)

            # Outro (1.5s fade)
            outro = self._make_title_frame(out_w, out_h, "ANALYSIS COMPLETE", "TAD AI POWERED")
            for k in range(int(fps * 1.5)):
                a = max(0.0, 1.0 - k / max(1, fps * 0.8))
                out.write((outro * a).astype(np.uint8))

            cap.release()
            out.release()
            return f"/static/highlights/{output_name}" if os.path.exists(output_path) else None
        except Exception as e:
            print(f"Master Reel Error: {e}")
            import traceback; traceback.print_exc()
            return None
