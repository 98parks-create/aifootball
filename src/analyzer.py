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
        """Window-based speed: displacement over ±5-frame window.
        Eliminates bbox jitter that inflates frame-to-frame calculations."""
        if len(target_tracks) < 3:
            return {}
        sorted_t = sorted(target_tracks, key=lambda t: t['frame'])
        frame_list = [t['frame'] for t in sorted_t]
        by_frame   = {t['frame']: t['pos'] for t in sorted_t}
        n = len(frame_list)
        HALF = 5
        speed_map = {}
        for idx in range(n):
            lo, hi   = max(0, idx - HALF), min(n - 1, idx + HALF)
            f_lo, f_hi = frame_list[lo], frame_list[hi]
            dt = (f_hi - f_lo) / fps
            if dt <= 0:
                continue
            dx = (by_frame[f_hi][0] - by_frame[f_lo][0]) * SCALE_X
            dy = (by_frame[f_hi][1] - by_frame[f_lo][1]) * SCALE_Y
            spd = min(np.sqrt(dx**2 + dy**2) / dt * 3.6, 38.0)
            speed_map[frame_list[idx]] = spd
        return speed_map

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _smooth_speeds(self, speeds, window=5):
        if len(speeds) < window:
            return speeds
        result = []
        half = window // 2
        for i in range(len(speeds)):
            lo = max(0, i - half)
            hi = min(len(speeds), i + half + 1)
            result.append(float(np.mean(speeds[lo:hi])))
        return result

    def calculate_individual_stats(self, tracks, target_id, position="ST", fps=25):
        empty = {"PAC":50,"PHY":50,"DRI":50,"PAS":50,"SHO":50,
                 "total_distance_km":0,"top_speed_kmh":0,"sprint_count":0}
        if not tracks or target_id not in tracks or not tracks[target_id]:
            return empty

        data = sorted(tracks[target_id], key=lambda t: t['frame'])

        # Window-based speed map (accurate, noise-free)
        speed_map = self._build_speed_map(data, fps)
        speeds = [v for v in speed_map.values()]

        # Distance with per-frame cap to reject teleportation artifacts
        max_d_frame = (38.0 / 3.6 / fps) * 1.5   # 38 km/h × 1.5 headroom
        dist_m = 0.0
        positions = [t['pos'] for t in data]
        for i in range(1, len(positions)):
            dx = (positions[i][0]-positions[i-1][0]) * SCALE_X
            dy = (positions[i][1]-positions[i-1][1]) * SCALE_Y
            d  = np.sqrt(dx**2+dy**2)
            if d <= max_d_frame:
                dist_m += d

        top = min(max(speeds) if speeds else 0, 38.0)
        avg = float(np.mean(speeds)) if speeds else 0
        km  = dist_m / 1000

        # Sprint state machine: on ≥20 km/h, off <15 km/h (hysteresis prevents flicker)
        sprint_count = 0
        in_sprint    = False
        for s in speeds:
            if not in_sprint and s >= 20.0:
                in_sprint = True
                sprint_count += 1
            elif in_sprint and s < 15.0:
                in_sprint = False
        sprints = sprint_count

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

    # ── Position-aware Event Detection ───────────────────────────────────────

    def _detect_events_by_position(self, tracks, target_id, ball_dict,
                                    fps=25, position="ST", target_team=-1):
        pos = position.upper()
        is_fwd = any(p in pos for p in ["ST","CF","LW","RW","SS","FW","CAM","AM"])
        is_mid = (any(p in pos for p in ["MF","CM","DM","CDM"]) and not is_fwd)
        is_def = (any(p in pos for p in ["CB","LB","RB","WB","SW","DF","GK"])
                  or not (is_fwd or is_mid))

        target_tracks = sorted(tracks[target_id], key=lambda t: t['frame'])
        if len(target_tracks) < 2:
            return []

        target_by_frame = {t['frame']: t for t in target_tracks}
        speed_map = self._build_speed_map(target_tracks, fps)

        # Ball velocity map (km/h) using window smoothing
        ball_frames = sorted(ball_dict.keys())
        ball_speed = {}
        for i in range(1, len(ball_frames)):
            f = ball_frames[i]
            p1, p2 = ball_dict[ball_frames[i-1]], ball_dict[f]
            dx = (p2[0]-p1[0]) * SCALE_X
            dy = (p2[1]-p1[1]) * SCALE_Y
            ball_speed[f] = min(np.sqrt(dx**2+dy**2) * fps * 3.6, 150.0)

        # Separate opponents and teammates using team_id
        opponents_by_frame = {}
        teammates_by_frame = {}
        for tid, data in tracks.items():
            if tid == target_id:
                continue
            for t in data:
                if t.get('class') != 0:
                    continue
                f      = t['frame']
                t_team = t.get('team_id', -1)
                is_opp = (target_team == -1 or t_team == -1 or t_team != target_team)
                dest   = opponents_by_frame if is_opp else teammates_by_frame
                if f not in dest:
                    dest[f] = []
                dest[f].append(t['pos'])

        # Legacy alias for code below (contested = any nearby player)
        others_by_frame = {
            f: opponents_by_frame.get(f, []) + teammates_by_frame.get(f, [])
            for f in set(list(opponents_by_frame.keys()) + list(teammates_by_frame.keys()))
        }

        events = []
        min_gap = int(fps * 4)

        def conflict(f):
            return any(abs(f - e['frame']) < min_gap for e in events)

        # ── 1. GOAL (볼이 골 구역 진입) ──────────────────────────────────
        for f in ball_frames:
            bx, by = ball_dict[f]
            if not ((bx < 4 or bx > 96) and 18 < by < 32):
                continue
            # 볼이 빠르게 들어왔는지 확인
            fast = any(ball_speed.get(f-k, 0) > 25 for k in range(1, int(fps*0.5)))
            if not fast: continue
            # 타겟이 2초 내에 볼을 터치했는지 확인
            for k in range(int(fps*2)):
                fb = f - k
                if fb in target_by_frame and fb in ball_dict:
                    d = np.hypot(target_by_frame[fb]['pos'][0]-ball_dict[fb][0],
                                 target_by_frame[fb]['pos'][1]-ball_dict[fb][1])
                    if d < 4 and not conflict(f):
                        events.append({"frame": f, "category": "GOAL", "score": 3000,
                                       "speed_kmh": 0, "section": "attack"})
                        break

        # ── 2. Touch-based events ────────────────────────────────────────
        for t in target_tracks:
            f = t['frame']
            if f not in ball_dict or conflict(f): continue
            tp = t.get('pos', [0,0])
            bp = ball_dict[f]
            dist = np.hypot(tp[0]-bp[0], tp[1]-bp[1])
            if dist > 3.0: continue

            spd   = speed_map.get(f, 0)
            field_x = tp[0]
            in_att  = field_x > 66 or field_x < 34   # 공격/수비 1/3
            in_mid  = 34 <= field_x <= 66

            # 볼이 터치 후 얼마나 이동하는가
            future_ball = [ball_dict[f+k] for k in range(1, int(fps*1.2)) if f+k in ball_dict]
            ball_traveled_m = 0
            if future_ball:
                dx = (future_ball[-1][0]-bp[0]) * SCALE_X
                dy = (future_ball[-1][1]-bp[1]) * SCALE_Y
                ball_traveled_m = np.sqrt(dx**2+dy**2)

            bspd_after = max((ball_speed.get(f+k,0) for k in range(1,int(fps*0.5)) if f+k in ball_speed), default=0)
            bspd_before = max((ball_speed.get(f-k,0) for k in range(1,int(fps*0.4)) if f-k in ball_speed), default=0)

            # 근처에 상대 선수가 있었나 (팀 구분 적용)
            contested = any(
                np.hypot(op[0]-bp[0], op[1]-bp[1]) < 4.0
                for op in opponents_by_frame.get(f, [])
            )
            opp_had_ball = any(
                np.hypot(op[0]-ball_dict.get(f-k,bp)[0], op[1]-ball_dict.get(f-k,bp)[1]) < 2.5
                for k in range(1, int(fps*0.8))
                for op in opponents_by_frame.get(f-k, [])
                if f-k in ball_dict
            )

            # 지속 드리블 여부 (이전 0.5초 동안 볼 유지)
            sustained = sum(
                1 for k in range(1, int(fps*0.6))
                if f-k in ball_dict and f-k in target_by_frame
                and np.hypot(target_by_frame[f-k]['pos'][0]-ball_dict[f-k][0],
                             target_by_frame[f-k]['pos'][1]-ball_dict[f-k][1]) < 3.0
            ) >= int(fps * 0.25)

            # ── SHOT (공격수/미드) ────────────────────────────────────
            if (is_fwd or is_mid) and in_att and bspd_after > 40:
                toward_goal = future_ball and (future_ball[-1][0] < 8 or future_ball[-1][0] > 92)
                if toward_goal or bspd_after > 70:
                    events.append({"frame": f, "category": "SHOT", "score": 1800,
                                   "speed_kmh": round(spd,1), "section": "attack"})
                    continue

            # ── TACKLE (수비수: 상대가 볼 보유 → 탈취) ────────────────
            if is_def and opp_had_ball and spd > 6:
                events.append({"frame": f, "category": "TACKLE", "score": 1500,
                               "speed_kmh": round(spd,1), "section": "defense"})
                continue

            # ── INTERCEPTION (미드/수비: 패스 가로채기) ────────────────
            if (is_def or is_mid) and opp_had_ball and bspd_before > 20 and spd < 10:
                events.append({"frame": f, "category": "INTERCEPTION", "score": 1400,
                               "speed_kmh": round(spd,1), "section": "defense"})
                continue

            # ── CLEARANCE (수비수: 수비 1/3에서 강한 킥) ──────────────
            if is_def and in_att and ball_traveled_m > 18 and bspd_after > 30:
                events.append({"frame": f, "category": "CLEARANCE", "score": 1200,
                               "speed_kmh": round(spd,1), "section": "defense"})
                continue

            # ── DRIBBLE (공격수/미드: 지속 드리블) ────────────────────
            if (is_fwd or is_mid) and sustained and spd > 10:
                events.append({"frame": f, "category": "DRIBBLE", "score": 1000,
                               "speed_kmh": round(spd,1), "section": "attack"})
                continue

            # ── LONG PASS / THROUGH BALL ───────────────────────────────
            if ball_traveled_m > 20 and bspd_after > 30:
                cat = "THROUGH BALL" if is_fwd and in_mid else "LONG PASS"
                events.append({"frame": f, "category": cat, "score": 750,
                               "speed_kmh": round(spd,1), "section": "attack" if is_fwd else "midfield"})
                continue

            # ── SHORT PASS ─────────────────────────────────────────────
            if ball_traveled_m > 5 and bspd_after > 15 and spd < 14:
                events.append({"frame": f, "category": "PASS", "score": 500,
                               "speed_kmh": round(spd,1), "section": "midfield"})
                continue

            # ── BALL CONTROL (볼 받기) ──────────────────────────────────
            if bspd_before > 20 and dist < 2.0:
                events.append({"frame": f, "category": "CONTROL", "score": 350,
                               "speed_kmh": round(spd,1), "section": "midfield"})
                continue

            # ── Generic touch (볼 터치 - 전부 클립 추출) ──────────────
            events.append({"frame": f, "category": "BALL TOUCH", "score": 250,
                           "speed_kmh": round(spd,1),
                           "section": "defense" if is_def else "attack" if is_fwd else "midfield"})

        # ── 3. BALL LOST (볼 손실 감지) ──────────────────────────────────
        for t in target_tracks:
            f = t['frame']
            if f not in ball_dict: continue
            d = np.hypot(t['pos'][0]-ball_dict[f][0], t['pos'][1]-ball_dict[f][1])
            if d > 2.5: continue
            # 이후 다른 선수가 볼을 가져가는지 확인
            for k in range(1, int(fps*1.5)):
                nf = f+k
                if nf not in ball_dict: continue
                if nf in target_by_frame:
                    nd = np.hypot(target_by_frame[nf]['pos'][0]-ball_dict[nf][0],
                                  target_by_frame[nf]['pos'][1]-ball_dict[nf][1])
                    if nd < 3: break   # 볼 유지
                opp_got = any(
                    np.hypot(op[0]-ball_dict[nf][0], op[1]-ball_dict[nf][1]) < 2.5
                    for op in others_by_frame.get(nf, [])
                )
                if opp_got and not conflict(nf):
                    events.append({"frame": nf, "category": "BALL LOST", "score": 100,
                                   "speed_kmh": 0, "section": "bad"})
                    break

        # ── 4. Sprint ────────────────────────────────────────────────────
        last_sp = -int(fps*8)
        for i in range(1, len(target_tracks)):
            f = target_tracks[i]['frame']
            spd = speed_map.get(f, 0)
            if spd > 22 and f - last_sp > int(fps*8) and not conflict(f):
                events.append({"frame": f, "category": "SPRINT", "score": 300,
                               "speed_kmh": round(spd,1), "section": "midfield"})
                last_sp = f

        # ── 5. Fallback ──────────────────────────────────────────────────
        if len(events) < 5:
            win = int(fps*6)
            for i in range(0, len(target_tracks)-win, win):
                f = target_tracks[i+win//2]['frame']
                if conflict(f): continue
                seg = sum(self._speed_kmh(target_tracks[j-1]['pos'], target_tracks[j]['pos'], fps)
                          for j in range(i+1, min(i+win, len(target_tracks)))
                          if self._speed_kmh(target_tracks[j-1]['pos'], target_tracks[j]['pos'], fps) < 40)
                events.append({"frame": f, "category": "ACTIVE PLAY", "score": seg,
                               "speed_kmh": 0, "section": "midfield"})

        events = sorted(events, key=lambda x: x['score'], reverse=True)[:20]
        events = sorted(events, key=lambda x: x['frame'])
        return events

    # ── Clip extraction ───────────────────────────────────────────────────────

    def extract_combined_highlights(self, video_path, tracks, target_id,
                                     fps=25, position="ST", target_team=-1):
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

        events = self._detect_events_by_position(
            tracks, target_id, ball_dict, fps, position, target_team
        )

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
                    "section": ev.get("section", "midfield"),
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
                if raw is not None:
                    cx = float(raw[0])
                    cy = float(raw[1])
                    if lx is not None:
                        cx = lx * 0.7 + cx * 0.3
                        cy = ly * 0.7 + cy * 0.3
                    lx, ly = cx, cy
                else:
                    cx = lx if lx is not None else wf / 2
                    cy = ly if ly is not None else hf / 2

                # Full frame — no crop/zoom
                res = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

                # Draw target ring at player position (scaled to output size)
                sx = int(cx * ow / wf)
                sy = int(cy * oh / hf)
                cv2.circle(res, (sx, sy), 38, (0, 215, 255), 3)
                cv2.circle(res, (sx, sy), 5,  (0, 215, 255), -1)

                prog = (fi - start_f) / total
                if prog < 0.1:
                    res = (res * (prog / 0.1)).astype(np.uint8)
                elif prog > 0.9:
                    res = (res * ((1 - prog) / 0.1)).astype(np.uint8)

                cv2.putText(res, "TAD AI", (24, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (220,40,40), 2, cv2.LINE_AA)
                cv2.putText(res, cat, (24, oh-28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,215,0), 2, cv2.LINE_AA)
                if spd_s:
                    (sw,_),_ = cv2.getTextSize(spd_s, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(res, spd_s, (ow-sw-24, oh-28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                if is_act and 0.35 < prog < 0.65:
                    out.write(res)  # duplicate frame for slow-mo effect
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
                is_act = ev.get('category') in ('BALL TOUCH','DRIBBLE','SPRINT')

                for fi in range(sf, ef):
                    ret, frame = cap.read()
                    if not ret: break
                    hf, wf = frame.shape[:2]

                    raw = target_dict.get(fi)
                    if raw is not None:
                        cx = float(raw[0])
                        cy = float(raw[1])
                        if lx is not None:
                            cx = lx*0.7+cx*0.3; cy = ly*0.7+cy*0.3
                        lx, ly = cx, cy
                    else:
                        cx = lx if lx is not None else wf/2
                        cy = ly if ly is not None else hf/2

                    # Full frame — no crop/zoom
                    res = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

                    # Target ring
                    sx = int(cx * ow / wf)
                    sy = int(cy * oh / hf)
                    cv2.circle(res, (sx, sy), 38, (0, 215, 255), 3)
                    cv2.circle(res, (sx, sy), 5,  (0, 215, 255), -1)

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
