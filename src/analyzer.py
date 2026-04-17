import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import cv2

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

SCALE_X = 105.0 / 100.0   # meters per normalized unit (x)
SCALE_Y = 68.0  / 50.0    # meters per normalized unit (y)


def _safe(text):
    """Remove non-ASCII for OpenCV putText (Korean → empty)."""
    return ''.join(c if ord(c) < 128 else '' for c in str(text)).strip() or 'PLAYER'


def _put_center(frame, text, y, scale, color, thick):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text, (max(0, (frame.shape[1]-tw)//2), y),
                font, scale, color, thick, cv2.LINE_AA)


class FootballAnalyzer:
    def __init__(self):
        self.base_storage = r"D:\aifootball_data"
        self.data_dir     = os.path.join(self.base_storage, "analysis")
        self.highlight_dir = os.path.join(self.base_storage, "highlights")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.highlight_dir, exist_ok=True)

    # ── Speed helpers ─────────────────────────────────────────────────────────

    def _speed_kmh(self, pos1, pos2, fps):
        dx = (pos2[0]-pos1[0]) * SCALE_X
        dy = (pos2[1]-pos1[1]) * SCALE_Y
        return np.sqrt(dx**2+dy**2) * fps * 3.6

    def _build_speed_map(self, target_tracks, fps):
        """Per-frame speed (km/h), noise-filtered."""
        sm = {}
        for i in range(1, len(target_tracks)):
            f = target_tracks[i]['frame']
            spd = self._speed_kmh(target_tracks[i-1]['pos'], target_tracks[i]['pos'], fps)
            if spd < 40:
                sm[f] = spd
        return sm

    # ── Stats ─────────────────────────────────────────────────────────────────

    def calculate_individual_stats(self, tracks, target_id, position="ST", fps=25):
        empty = {"PAC":50,"PHY":50,"DRI":50,"PAS":50,"SHO":50,
                 "total_distance_km":0,"top_speed_kmh":0,"sprint_count":0}
        if not tracks or target_id not in tracks or not tracks[target_id]:
            return empty

        data = tracks[target_id]
        positions = [t['pos'] for t in data]
        dist_m, speeds = 0.0, []
        for i in range(1, len(positions)):
            spd = self._speed_kmh(positions[i-1], positions[i], fps)
            if spd < 40:
                dx = (positions[i][0]-positions[i-1][0]) * SCALE_X
                dy = (positions[i][1]-positions[i-1][1]) * SCALE_Y
                dist_m += np.sqrt(dx**2+dy**2)
                speeds.append(spd)

        top = max(speeds) if speeds else 0
        avg = float(np.mean(speeds)) if speeds else 0
        km  = dist_m / 1000
        sprints = sum(1 for s in speeds if s > 20)

        pos = position.upper()
        pac = min(99, max(55, int(top * 2.8)))
        phy = min(99, max(55, int(km * 12)))

        if any(p in pos for p in ["ST","FW","CF","LW","RW","SS"]):
            sho = min(99, max(70, 70+int(top*0.4)))
            dri = min(99, max(70, 72+int(avg*1.5)))
            pas = 70
        elif any(p in pos for p in ["MF","CM","AM","DM","CAM","CDM"]):
            sho = 70; dri = min(99,max(76,78+int(avg*1.2))); pas = min(99,max(76,78+int(km*3)))
        else:
            sho = 58; dri = 66; phy = min(99,phy+10); pas = 70

        return {"PAC":pac,"PHY":phy,"DRI":dri,"PAS":pas,"SHO":sho,
                "total_distance_km":round(km,2),"top_speed_kmh":round(top,1),"sprint_count":sprints}

    # ── Pitch Heatmap (OpenCV green pitch) ───────────────────────────────────

    def _draw_pitch(self, w=1050, h=680):
        img = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)
        lw, white, m = 3, (255,255,255), 15
        cv2.rectangle(img, (m,m), (w-m,h-m), white, lw)
        cx, cy = w//2, h//2
        cv2.line(img, (cx,m), (cx,h-m), white, lw)
        cv2.circle(img, (cx,cy), 91, white, lw)
        cv2.circle(img, (cx,cy), 6, white, -1)
        # Penalty areas
        cv2.rectangle(img, (m,148), (178,h-148), white, lw)
        cv2.rectangle(img, (w-178,148), (w-m,h-148), white, lw)
        # 6-yard boxes
        cv2.rectangle(img, (m,245), (70,h-245), white, lw)
        cv2.rectangle(img, (w-70,245), (w-m,h-245), white, lw)
        # Penalty spots
        cv2.circle(img, (m+115, cy), 6, white, -1)
        cv2.circle(img, (w-m-115, cy), 6, white, -1)
        # Corner arcs
        for px, py, a0 in [(m,m,0),(w-m,m,90),(w-m,h-m,180),(m,h-m,270)]:
            cv2.ellipse(img,(px,py),(18,18),0,a0,a0+90,white,lw)
        return img

    def generate_pitch_heatmap(self, tracks, target_id, session_id):
        if not tracks or target_id not in tracks:
            return None
        try:
            positions = np.array([t['pos'] for t in tracks[target_id]])
            pw, ph = 1050, 680
            pitch = self._draw_pitch(pw, ph)

            hm, _, _ = np.histogram2d(positions[:,0], positions[:,1],
                                      bins=[60,30], range=[[0,100],[0,50]])
            hm = gaussian_filter(hm.T, sigma=2.5) if HAS_SCIPY else hm.T
            hm_norm = (hm / (hm.max()+1e-9) * 255).astype(np.uint8)
            hm_resized = cv2.resize(hm_norm, (pw, ph))
            hm_colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_HOT)

            mask = hm_resized > 15
            result = pitch.copy()
            blend = cv2.addWeighted(pitch, 0.25, hm_colored, 0.75, 0)
            result[mask] = blend[mask]

            filename = f"{session_id}_heatmap.png"
            cv2.imwrite(os.path.join(self.highlight_dir, filename), result)
            return f"/static/highlights/{filename}"
        except Exception as e:
            print(f"Heatmap Error: {e}"); return None

    # ── AI Comment ────────────────────────────────────────────────────────────

    def generate_ai_comment(self, stats):
        report = {"pros":[], "cons":[], "scouting_note":""}
        spd  = stats.get('top_speed_kmh', 0)
        dist = stats.get('total_distance_km', 0)
        spr  = stats.get('sprint_count', 0)

        if spd >= 28:   report['pros'].append(f"엘리트급 최고속도 {spd:.1f}km/h — 리그 상위 5% 스프린트")
        elif spd >= 23: report['pros'].append(f"우수한 순간 가속력 ({spd:.1f}km/h) — 수비 배후 공략 가능")
        elif spd >= 18: report['pros'].append(f"안정적인 이동 속도 ({spd:.1f}km/h)")
        if dist >= 8:   report['pros'].append(f"왕성한 활동량 {dist:.1f}km — 팀의 엔진")
        elif dist >= 5: report['pros'].append(f"적정 이동거리 {dist:.1f}km — 포지션 밸런스 우수")
        if spr >= 10:   report['pros'].append(f"스프린트 {spr}회 — 폭발적인 전환력")
        if stats.get('DRI',0) >= 80: report['pros'].append("볼 킵 능력 우수 — 압박 상황에서도 안정적")

        if spd < 18:  report['cons'].append("스프린트 속도 향상 필요 (목표: 22km/h 이상)")
        if dist < 4:  report['cons'].append("전·후방 이동거리 증가 — 체력 훈련 권장")
        if stats.get('PHY',0) < 65: report['cons'].append("후반 체력 유지력 강화 훈련 권장")

        if len(report['pros']) >= 2:
            report['scouting_note'] = f"최고속도 {spd:.1f}km/h, 이동거리 {dist:.1f}km. 실전 투입 즉시 팀에 기여 가능한 즉전감으로 평가됩니다."
        elif len(report['pros']) == 1:
            report['scouting_note'] = "기본기 탄탄. 강점을 더 발전시키면 상위 레벨 경쟁 가능합니다."
        else:
            report['scouting_note'] = "데이터 축적을 통해 성장 궤적을 추적하겠습니다."
        return report

    # ── Event Detection ───────────────────────────────────────────────────────

    def _detect_events(self, tracks, target_id, ball_dict, fps=25):
        """
        모든 볼 터치(proximity < 3m) 클립 + 스프린트 이벤트.
        볼 터치는 전부 추출 (최소 간격 4초).
        이벤트마다 점수: 볼 터치 > 스프린트 > 활동량.
        """
        target_tracks = tracks[target_id]
        if len(target_tracks) < 2:
            return []

        speed_map = self._build_speed_map(target_tracks, fps)
        events = []
        min_gap = int(fps * 4)   # 4초 간격 (볼터치는 촘촘하게)
        last_touch = -min_gap
        last_sprint = -int(fps * 8)

        # ── 볼 터치 이벤트 (핵심) ──────────────────────────────────────────
        for t in target_tracks:
            f = t['frame']
            if f not in ball_dict:
                continue
            p = t.get('pos', [0,0])
            dist = np.hypot(p[0]-ball_dict[f][0], p[1]-ball_dict[f][1])
            if dist < 3.0 and f - last_touch > min_gap:
                spd = speed_map.get(f, 0)
                # 볼 터치 중 속도로 카테고리 분류
                if spd > 14:
                    cat = "Dribble"      # 뛰면서 볼 터치 = 드리블
                else:
                    cat = "Ball Touch"   # 정지/저속 = 패스/수비/트랩
                events.append({
                    "frame": f,
                    "category": cat,
                    "score": 300 - dist * 10,
                    "speed_kmh": round(spd, 1),
                })
                last_touch = f

        # ── 스프린트 이벤트 (볼 없을 때) ─────────────────────────────────
        for i in range(1, len(target_tracks)):
            f = target_tracks[i]['frame']
            spd = speed_map.get(f, 0)
            if spd > 22 and f - last_sprint > int(fps * 8) and f not in ball_dict:
                events.append({
                    "frame": f,
                    "category": "Sprint Run",
                    "score": spd * 1.5,
                    "speed_kmh": round(spd, 1),
                })
                last_sprint = f

        # ── 활동량 보완 (이벤트가 5개 미만일 때) ──────────────────────────
        if len(events) < 5:
            win = int(fps * 6)
            for i in range(0, len(target_tracks) - win, win):
                f = target_tracks[i + win//2]['frame']
                if any(abs(f - e['frame']) < fps*6 for e in events):
                    continue
                seg = sum(
                    self._speed_kmh(target_tracks[j-1]['pos'], target_tracks[j]['pos'], fps)
                    for j in range(i+1, min(i+win, len(target_tracks)))
                    if self._speed_kmh(target_tracks[j-1]['pos'], target_tracks[j]['pos'], fps) < 40
                )
                events.append({"frame": f, "category": "Active Play", "score": seg, "speed_kmh": 0})

        events = sorted(events, key=lambda x: x['score'], reverse=True)[:15]
        events = sorted(events, key=lambda x: x['frame'])
        return events

    # ── Clip extraction ───────────────────────────────────────────────────────

    def extract_combined_highlights(self, video_path, tracks, target_id, fps=25):
        if target_id not in tracks:
            return [], []
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            cap.release()
        except Exception:
            pass

        target_dict = {t['frame']: t['pos_px'] for t in tracks[target_id] if 'pos_px' in t}
        ball_tracks = [t for tid, data in tracks.items()
                       if data and data[0].get('class') == 32 for t in data]
        ball_dict = {b['frame']: b['pos'] for b in ball_tracks}

        events = self._detect_events(tracks, target_id, ball_dict, fps)

        pre  = int(fps * 3)
        post = int(fps * 3)
        highlights = []

        for i, ev in enumerate(events):
            start_f = max(0, int(ev['frame'] - pre))
            end_f   = int(ev['frame'] + post)
            clip_name = f"tad_hl_{target_id}_{i}.mp4"
            clip_path = os.path.join(self.highlight_dir, clip_name)
            seg = {f: target_dict[f] for f in range(start_f, end_f+1) if f in target_dict}
            if self._save_clip(video_path, start_f, end_f, clip_path, seg, ev, fps):
                highlights.append({
                    "url": f"/static/highlights/{clip_name}",
                    "category": ev['category'],
                    "frame": int(ev['frame']),
                    "speed_kmh": ev.get("speed_kmh", 0),
                })

        return highlights, events

    # ── Cinematic clip ────────────────────────────────────────────────────────

    def _title_frame(self, w, h, line1, line2=""):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for r in range(h):
            img[r,:] = [int(30*(1-r/h)), 0, 0]
        _put_center(img, "TAD AI", h//2-40, 2.2, (220,40,40), 4)
        _put_center(img, _safe(line1).upper(), h//2+20, 1.0, (255,215,0), 2)
        if line2:
            _put_center(img, _safe(line2).upper(), h//2+65, 0.65, (180,180,180), 1)
        return img

    def _save_clip(self, video_path, start_f, end_f, out_path,
                   tracks, event=None, fps=25):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            ow, oh = 1280, 720

            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (ow, oh))
            if not out.isOpened():
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow, oh))

            cat    = _safe(event.get('category','HIGHLIGHT') if event else 'HIGHLIGHT').upper()
            spd_v  = event.get('speed_kmh', 0) if event else 0
            spd_s  = f"{spd_v:.0f} KM/H" if spd_v > 1 else ""
            is_act = event and event.get('category') in ('Ball Touch','Dribble','Sprint Run')

            # Title card (0.4s fade-in)
            title = self._title_frame(ow, oh, cat, spd_s)
            for k in range(int(fps * 0.4)):
                a = min(1.0, k / max(1, fps*0.2))
                out.write((title * a).astype(np.uint8))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            total = max(end_f - start_f, 1)
            lx = ly = None

            for fi in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret: break
                hf, wf = frame.shape[:2]

                raw = tracks.get(fi)
                cx = float(raw[0]) if raw is not None else wf/2
                cy = float(raw[1]) if raw is not None else hf/2
                if lx is not None:
                    cx = lx*0.85 + cx*0.15
                    cy = ly*0.85 + cy*0.15
                lx, ly = cx, cy

                cs = min(wf, hf, 960)
                x1 = int(max(0, min(wf-cs, cx-cs/2)))
                y1 = int(max(0, min(hf-cs, cy-cs/2)))
                res = cv2.resize(frame[y1:y1+cs, x1:x1+cs], (ow, oh),
                                 interpolation=cv2.INTER_LANCZOS4)

                prog = (fi - start_f) / total
                if prog < 0.1:
                    res = (res * (prog/0.1)).astype(np.uint8)
                elif prog > 0.9:
                    res = (res * ((1-prog)/0.1)).astype(np.uint8)

                # Overlays (ASCII only)
                cv2.putText(res, "TAD AI", (24,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220,40,40), 2, cv2.LINE_AA)
                cv2.putText(res, cat, (24, oh-28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,215,0), 2, cv2.LINE_AA)
                if spd_s:
                    (sw,_),_ = cv2.getTextSize(spd_s, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(res, spd_s, (ow-sw-24, oh-28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                # Slow-mo at event peak (middle 30%)
                if is_act and 0.35 < prog < 0.65:
                    out.write(res)

                out.write(res)

            cap.release(); out.release()
            return os.path.exists(out_path) and os.path.getsize(out_path) > 1000
        except Exception as e:
            print(f"Clip error: {e}")
            import traceback; traceback.print_exc()
            return False

    # ── Master Sizzle Reel ────────────────────────────────────────────────────

    def generate_master_sizzle_reel(self, video_path, events, tracks,
                                     target_id, session_id, player_name="PLAYER"):
        if not events or target_id not in tracks:
            return None
        target_dict = {t['frame']: t['pos_px'] for t in tracks[target_id] if 'pos_px' in t}
        out_name = f"{session_id}_tad_master.mp4"
        out_path = os.path.join(self.highlight_dir, out_name)

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            ow, oh = 1280, 720
            pname = _safe(player_name)

            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (ow, oh))
            if not out.isOpened():
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow, oh))

            # Intro (1.5s)
            intro = self._title_frame(ow, oh, pname, "OFFICIAL HIGHLIGHT FILM")
            for k in range(int(fps*1.5)):
                a = min(1.0, k / max(1, fps*0.5))
                out.write((intro*a).astype(np.uint8))

            pre = int(fps*3); post = int(fps*3)
            lx = ly = None

            for ev in events:
                sf = max(0, int(ev['frame']-pre))
                ef = int(ev['frame']+post)
                cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
                total = max(ef-sf, 1)
                cat   = _safe(ev.get('category','HIGHLIGHT')).upper()
                spd_v = ev.get('speed_kmh', 0)
                spd_s = f"{spd_v:.0f} KM/H" if spd_v > 1 else ""
                is_act = ev.get('category') in ('Ball Touch','Dribble','Sprint Run')

                for fi in range(sf, ef):
                    ret, frame = cap.read()
                    if not ret: break
                    hf, wf = frame.shape[:2]
                    raw = target_dict.get(fi)
                    cx = float(raw[0]) if raw is not None else wf/2
                    cy = float(raw[1]) if raw is not None else hf/2
                    if lx is not None:
                        cx = lx*0.85+cx*0.15; cy = ly*0.85+cy*0.15
                    lx, ly = cx, cy

                    cs = min(wf, hf, 960)
                    x1 = int(max(0, min(wf-cs, cx-cs/2)))
                    y1 = int(max(0, min(hf-cs, cy-cs/2)))
                    res = cv2.resize(frame[y1:y1+cs, x1:x1+cs], (ow, oh),
                                     interpolation=cv2.INTER_LANCZOS4)

                    prog = (fi-sf)/total
                    if prog < 0.08:
                        res = (res*(prog/0.08)).astype(np.uint8)
                    elif prog > 0.92:
                        res = (res*((1-prog)/0.08)).astype(np.uint8)

                    cv2.putText(res, "TAD AI", (24,38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220,40,40), 2, cv2.LINE_AA)
                    cv2.putText(res, cat, (24, oh-28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,215,0), 2, cv2.LINE_AA)
                    (pnw,_),_ = cv2.getTextSize(pname, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(res, pname, (ow-pnw-24, 38),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,215,0), 2, cv2.LINE_AA)
                    if spd_s:
                        (sw,_),_ = cv2.getTextSize(spd_s, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.putText(res, spd_s, (ow-sw-24, oh-28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                    if is_act and 0.35 < prog < 0.65:
                        out.write(res)
                    out.write(res)

                # 0.15s black cut
                black = np.zeros((oh, ow, 3), dtype=np.uint8)
                for _ in range(int(fps*0.15)):
                    out.write(black)

            # Outro (1.5s fade-out)
            outro = self._title_frame(ow, oh, "ANALYSIS COMPLETE", "TAD AI POWERED")
            for k in range(int(fps*1.5)):
                a = max(0.0, 1.0 - k/max(1, fps*0.8))
                out.write((outro*a).astype(np.uint8))

            cap.release(); out.release()
            return f"/static/highlights/{out_name}" if os.path.exists(out_path) else None
        except Exception as e:
            print(f"Master Reel Error: {e}")
            import traceback; traceback.print_exc()
            return None
