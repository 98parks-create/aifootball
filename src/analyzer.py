import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import cv2
from datetime import datetime

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
    _KO_FONT_PATHS = [
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        r"/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        r"/System/Library/Fonts/AppleGothic.ttf",
    ]
    _font_cache: dict = {}

    def _get_ko_font(size: int):
        if size in _font_cache:
            return _font_cache[size]
        for p in _KO_FONT_PATHS:
            if os.path.exists(p):
                try:
                    f = ImageFont.truetype(p, size)
                    _font_cache[size] = f
                    return f
                except Exception:
                    pass
        f = ImageFont.load_default()
        _font_cache[size] = f
        return f
except ImportError:
    HAS_PIL = False
    def _get_ko_font(size: int):
        return None

SCALE_X = 105.0 / 100.0
SCALE_Y = 68.0  / 50.0

# ── 한국어 카테고리 이름 ────────────────────────────────────────────────────────
CATEGORY_KO = {
    'GOAL':         '⚽ 골',
    'ASSIST':       '🎯 어시스트',
    'SHOT':         '💥 슈팅',
    'DRIBBLE':      '🔥 드리블',
    'THROUGH BALL': '🎯 스루패스',
    'LONG PASS':    '📐 롱패스',
    'PASS':         '📐 패스',
    'BALL TOUCH':   '🦶 볼터치',
    'CONTROL':      '🎱 볼컨트롤',
    'TACKLE':       '⚡ 태클',
    'INTERCEPTION': '✂ 인터셉트',
    'CLEARANCE':    '💪 클리어링',
    'SPRINT':       '🏃 스프린트',
    'BALL LOST':    '❌ 볼 손실',
    'BALL CONTEST': '⚔ 공 다툼',
    'ACTIVE PLAY':  '▶ 활동',
}

# ── 이벤트별 클립 길이 (초): (사전, 사후) ─────────────────────────────────────
CLIP_WINDOWS = {
    'GOAL':         (6, 7),
    'ASSIST':       (5, 6),
    'SHOT':         (4, 5),
    'TACKLE':       (3, 4),
    'INTERCEPTION': (3, 3),
    'CLEARANCE':    (3, 3),
    'DRIBBLE':      (3, 4),
    'THROUGH BALL': (3, 3),
    'LONG PASS':    (2, 3),
    'PASS':         (2, 2),
    'SPRINT':       (2, 4),
    'CONTROL':      (2, 2),
    'BALL TOUCH':   (2, 2),
    'BALL LOST':    (2, 3),
    'BALL CONTEST': (3, 3),
    'ACTIVE PLAY':  (2, 3),
}
DEFAULT_WINDOW = (3, 3)


def _safe(text: str) -> str:
    return ''.join(c if ord(c) < 128 else '' for c in str(text)).strip() or 'PLAYER'


def _put_center(frame, text, y, scale, color, thick):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(frame, text, (max(0, (frame.shape[1] - tw) // 2), y),
                font, scale, color, thick, cv2.LINE_AA)


def _pil_text(frame, texts):
    """texts = list of (x, y, str, font_size, (B,G,R)).
    Converts frame BGR→PIL once, draws all texts, converts back."""
    if not HAS_PIL:
        for x, y, text, size, color in texts:
            cv2.putText(frame, _safe(text), (x, y + size // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, size / 40.0,
                        color, 2, cv2.LINE_AA)
        return
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for x, y, text, size, color in texts:
        font = _get_ko_font(size)
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((x, y), text, font=font, fill=rgb)
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _pil_text_center(frame, text, y, font_size, color):
    """텍스트를 수평 가운데 정렬로 그린다."""
    if not HAS_PIL:
        _put_center(frame, _safe(text), y + font_size // 2,
                    font_size / 40.0, color, 2)
        return
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _get_ko_font(font_size)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x = max(0, (frame.shape[1] - tw) // 2)
    except Exception:
        x = frame.shape[1] // 4
    draw.text((x, y), text, font=font, fill=rgb)
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class FootballAnalyzer:
    def __init__(self):
        base = os.environ.get('TAD_DATA_DIR',
               os.path.join(os.path.expanduser("~"), "aifootball_data"))
        self.highlight_dir = os.path.join(base, "highlights")
        self.data_dir      = os.path.join(base, "analysis")
        os.makedirs(self.highlight_dir, exist_ok=True)
        os.makedirs(self.data_dir,      exist_ok=True)

    # ── Speed helpers ──────────────────────────────────────────────────────────

    def _speed_kmh(self, pos1, pos2, fps):
        dx = (pos2[0] - pos1[0]) * SCALE_X
        dy = (pos2[1] - pos1[1]) * SCALE_Y
        return np.sqrt(dx ** 2 + dy ** 2) * fps * 3.6

    def _build_speed_map(self, target_tracks, fps):
        if len(target_tracks) < 3:
            return {}
        sorted_t   = sorted(target_tracks, key=lambda t: t['frame'])
        frame_list = [t['frame'] for t in sorted_t]
        by_frame   = {t['frame']: t['pos'] for t in sorted_t}
        n, HALF    = len(frame_list), 8
        speed_map  = {}
        for idx in range(n):
            lo, hi   = max(0, idx - HALF), min(n - 1, idx + HALF)
            f_lo, f_hi = frame_list[lo], frame_list[hi]
            dt = (f_hi - f_lo) / fps
            if dt <= 0:
                continue
            dx = (by_frame[f_hi][0] - by_frame[f_lo][0]) * SCALE_X
            dy = (by_frame[f_hi][1] - by_frame[f_lo][1]) * SCALE_Y
            # 아마추어 축구 최대 속도 32 km/h로 캡 (노이즈로 인한 38 이상 수치 방지)
            speed_map[frame_list[idx]] = min(np.sqrt(dx ** 2 + dy ** 2) / dt * 3.6, 32.0)
        return speed_map

    # ── Stats ──────────────────────────────────────────────────────────────────

    def calculate_individual_stats(self, tracks, target_id, position="ST", fps=25):
        empty = {"PAC": 50, "PHY": 50, "DRI": 50, "PAS": 50, "SHO": 50,
                 "total_distance_km": 0, "top_speed_kmh": 0, "sprint_count": 0}
        if not tracks or target_id not in tracks or not tracks[target_id]:
            return empty

        data      = sorted(tracks[target_id], key=lambda t: t['frame'])
        speed_map = self._build_speed_map(data, fps)
        speeds    = list(speed_map.values())

        # 실제 프레임 간격으로 최대 이동거리 계산 (STABLE_SKIP 등 격 프레임 대응)
        dist_m    = 0.0
        frames    = [t['frame'] for t in data]
        positions = [t['pos'] for t in data]
        for i in range(1, len(positions)):
            frame_gap = max(1, frames[i] - frames[i - 1])
            max_d = (32.0 / 3.6 / fps) * frame_gap * 1.5  # 32 km/h 기준 허용 이동거리
            dx = (positions[i][0] - positions[i - 1][0]) * SCALE_X
            dy = (positions[i][1] - positions[i - 1][1]) * SCALE_Y
            d  = np.sqrt(dx ** 2 + dy ** 2)
            if d <= max_d:
                dist_m += d

        top = min(max(speeds) if speeds else 0, 32.0)
        avg = float(np.mean(speeds)) if speeds else 0
        km  = dist_m / 1000

        # 스프린트 기준: 7 m/s (25.2 km/h) 이상, 0.5초(fps*0.5 프레임) 지속
        SPRINT_ENTRY = 7.0 * 3.6   # 25.2 km/h
        SPRINT_EXIT  = 5.5 * 3.6   # 19.8 km/h (이력 기준)
        MIN_SPRINT_F = max(1, int(fps * 0.5))
        sprint_count, in_sprint, sprint_start_f = 0, False, 0
        for f, s in sorted(speed_map.items()):
            if not in_sprint and s >= SPRINT_ENTRY:
                in_sprint = True
                sprint_start_f = f
            elif in_sprint and s < SPRINT_EXIT:
                if f - sprint_start_f >= MIN_SPRINT_F:
                    sprint_count += 1
                in_sprint = False
        if in_sprint:
            last_f = max(speed_map.keys(), default=sprint_start_f)
            if last_f - sprint_start_f >= MIN_SPRINT_F:
                sprint_count += 1

        pos = position.upper()
        # 32 km/h 캡 기준: 28→88, 30→93, 32→99
        pac = min(99, max(55, int(top * 3.1)))
        phy = min(99, max(55, int(km * 12)))

        if any(p in pos for p in ["ST", "FW", "CF", "LW", "RW", "SS"]):
            sho = min(99, max(70, 70 + int(top * 0.4)))
            dri = min(99, max(70, 72 + int(avg * 1.5)))
            pas = min(99, max(65, 65 + int(km * 2)))
        elif any(p in pos for p in ["MF", "CM", "AM", "DM", "CAM", "CDM"]):
            sho = min(99, max(60, 62 + int(top * 0.3)))
            dri = min(99, max(76, 78 + int(avg * 1.2)))
            pas = min(99, max(76, 78 + int(km * 3)))
        else:  # DEF / GK
            sho = min(99, max(45, 48 + int(top * 0.2)))
            dri = min(99, max(60, 62 + int(avg * 0.8)))
            phy = min(99, phy + 10)
            pas = min(99, max(65, 66 + int(km * 2)))

        return {"PAC": pac, "PHY": phy, "DRI": dri, "PAS": pas, "SHO": sho,
                "total_distance_km": round(km, 2),
                "top_speed_kmh": round(top, 1),
                "sprint_count": sprint_count}

    # ── Pitch heatmap ──────────────────────────────────────────────────────────

    def _draw_pitch(self, w=1050, h=680):
        img = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)
        lw, white, m = 3, (255, 255, 255), 15
        cv2.rectangle(img, (m, m), (w - m, h - m), white, lw)
        cx, cy = w // 2, h // 2
        cv2.line(img, (cx, m), (cx, h - m), white, lw)
        cv2.circle(img, (cx, cy), 91, white, lw)
        cv2.circle(img, (cx, cy), 6, white, -1)
        cv2.rectangle(img, (m, 148), (178, h - 148), white, lw)
        cv2.rectangle(img, (w - 178, 148), (w - m, h - 148), white, lw)
        cv2.rectangle(img, (m, 245), (70, h - 245), white, lw)
        cv2.rectangle(img, (w - 70, 245), (w - m, h - 245), white, lw)
        cv2.circle(img, (m + 115, cy), 6, white, -1)
        cv2.circle(img, (w - m - 115, cy), 6, white, -1)
        for px, py, a0 in [(m, m, 0), (w - m, m, 90), (w - m, h - m, 180), (m, h - m, 270)]:
            cv2.ellipse(img, (px, py), (18, 18), 0, a0, a0 + 90, white, lw)
        return img

    def generate_pitch_heatmap(self, tracks, target_id, session_id):
        if not tracks or target_id not in tracks:
            return None
        try:
            positions = np.array([t['pos'] for t in tracks[target_id]])
            pw, ph    = 1050, 680
            pitch     = self._draw_pitch(pw, ph)

            hm, _, _ = np.histogram2d(positions[:, 0], positions[:, 1],
                                      bins=[60, 30], range=[[0, 100], [0, 50]])
            hm        = gaussian_filter(hm.T, sigma=2.5) if HAS_SCIPY else hm.T
            hm_norm   = (hm / (hm.max() + 1e-9) * 255).astype(np.uint8)
            hm_resized = cv2.resize(hm_norm, (pw, ph))
            hm_colored = cv2.applyColorMap(hm_resized, cv2.COLORMAP_HOT)

            mask   = hm_resized > 15
            result = pitch.copy()
            blend  = cv2.addWeighted(pitch, 0.25, hm_colored, 0.75, 0)
            result[mask] = blend[mask]

            filename = f"{session_id}_heatmap.png"
            cv2.imwrite(os.path.join(self.highlight_dir, filename), result)
            return f"/static/highlights/{filename}"
        except Exception as e:
            print(f"Heatmap Error: {e}")
            return None

    # ── AI Comment ─────────────────────────────────────────────────────────────

    def generate_ai_comment(self, stats, position="ST"):
        report = {"pros": [], "cons": [], "scouting_note": ""}
        spd  = stats.get('top_speed_kmh', 0)
        dist = stats.get('total_distance_km', 0)
        spr  = stats.get('sprint_count', 0)
        pos  = position.upper()

        # 강점 분석
        if spd >= 28:
            report['pros'].append(f"엘리트급 최고속도 {spd:.1f}km/h — 리그 상위 5% 스프린트 능력")
        elif spd >= 24:
            report['pros'].append(f"우수한 순간 가속력 ({spd:.1f}km/h) — 수비 배후 공략 가능")
        elif spd >= 20:
            report['pros'].append(f"안정적인 주력 ({spd:.1f}km/h) — 포지션 밸런스 양호")

        if dist >= 9:
            report['pros'].append(f"왕성한 활동량 {dist:.1f}km — 팀의 심장")
        elif dist >= 6:
            report['pros'].append(f"충분한 이동거리 {dist:.1f}km — 포지션 역할 충실")
        elif dist >= 4:
            report['pros'].append(f"기본 활동량 {dist:.1f}km 확보")

        if spr >= 12:
            report['pros'].append(f"폭발적 스프린트 {spr}회 — 전환 플레이에서 탁월")
        elif spr >= 7:
            report['pros'].append(f"스프린트 {spr}회 — 적극적인 전환 가담")

        dri = stats.get('DRI', 0)
        pac = stats.get('PAC', 0)
        if dri >= 82:
            report['pros'].append("볼 킵 능력 우수 — 압박 상황에서도 침착")
        if pac >= 85:
            report['pros'].append("뛰어난 스피드 스탯 — 라인 브레이킹 위협적")

        psr = stats.get('pass_success_rate')
        if psr is not None:
            if psr >= 80:
                report['pros'].append(f"패스 성공률 {psr}% — 정확한 볼 배급, 팀 빌드업 핵심")
            elif psr >= 65:
                report['pros'].append(f"패스 성공률 {psr}% — 준수한 연결 능력")
            elif psr < 50:
                report['cons'].append(f"패스 성공률 {psr}% — 볼 배급 정확도 개선 필요")

        # 포지션별 특이 코멘트
        if any(p in pos for p in ["ST", "CF", "LW", "RW", "SS", "FW"]):
            if spd >= 26:
                report['pros'].append("공격수로서 이상적인 스프린트 — 수비 뒤 공간 침투 가능")
        elif any(p in pos for p in ["CB", "LB", "RB", "WB", "DF"]):
            phy = stats.get('PHY', 0)
            if phy >= 72:
                report['pros'].append("수비수로서 탄탄한 체력 — 90분 커버리지 가능")

        # 보완점 분석
        if spd < 18:
            report['cons'].append("스프린트 최고속도 향상 필요 (목표: 22km/h 이상)")
        if dist < 3.5:
            report['cons'].append("이동거리 부족 — 전·후방 적극 이동 훈련 권장")
        elif dist < 5 and any(p in pos for p in ["MF", "CM", "AM", "DM", "CAM", "CDM"]):
            report['cons'].append("미드필더 기준 이동거리 추가 확보 필요 (목표: 8km 이상)")
        if spr < 3:
            report['cons'].append("스프린트 횟수 부족 — 폭발적 전환 훈련 필요")
        phy = stats.get('PHY', 0)
        if phy < 62:
            report['cons'].append("체력 지수 개선 — 후반 압박 지속력 강화 훈련 권장")

        # 스카우팅 노트
        pro_cnt = len(report['pros'])
        if pro_cnt >= 3:
            report['scouting_note'] = (
                f"최고속도 {spd:.1f}km/h, 이동거리 {dist:.1f}km. "
                "복수의 특출난 지표가 확인됩니다. 즉전감으로 팀에 기여 가능하며, "
                "지속적인 데이터 관리 시 상위 리그 진출도 기대됩니다."
            )
        elif pro_cnt >= 1:
            report['scouting_note'] = (
                "기본기 탄탄. 핵심 강점을 더 발전시키면 상위 레벨 경쟁 충분합니다."
            )
        else:
            report['scouting_note'] = (
                "경기 데이터를 지속 축적해 성장 궤적을 추적하겠습니다. "
                "다음 경기에서 더 많은 데이터 확보를 권장합니다."
            )
        return report

    # ── Minimap overlay ────────────────────────────────────────────────────────

    def _draw_minimap(self, frame, field_pos, size_w=130):
        """오른쪽 상단에 미니 경기장 + 선수 위치 표시."""
        size_h = size_w // 2
        h, w   = frame.shape[:2]
        margin = 12

        # 미니 필드 생성
        mini = np.full((size_h, size_w, 3), (28, 100, 36), dtype=np.uint8)
        # 경계선
        cv2.rectangle(mini, (1, 1), (size_w - 2, size_h - 2), (255, 255, 255), 1)
        # 중앙선
        cv2.line(mini, (size_w // 2, 1), (size_w // 2, size_h - 2), (255, 255, 255), 1)
        # 중앙 원
        cv2.circle(mini, (size_w // 2, size_h // 2), size_h // 4, (255, 255, 255), 1)
        # 페널티 박스
        pa_w = max(2, int(size_w * 0.14))
        pa_h = int(size_h * 0.55)
        pa_t = (size_h - pa_h) // 2
        cv2.rectangle(mini, (1, pa_t), (pa_w, pa_t + pa_h), (255, 255, 255), 1)
        cv2.rectangle(mini, (size_w - pa_w - 1, pa_t), (size_w - 2, pa_t + pa_h), (255, 255, 255), 1)

        # 선수 위치 도트
        if field_pos is not None:
            px = int(np.clip(field_pos[0] / 100 * size_w, 3, size_w - 4))
            py = int(np.clip(field_pos[1] / 50  * size_h, 3, size_h - 4))
            cv2.circle(mini, (px, py), 4, (0, 215, 255), -1)
            cv2.circle(mini, (px, py), 5, (0, 0, 0), 1)

        # 프레임에 반투명 합성
        x1 = w - size_w - margin
        y1 = margin
        roi     = frame[y1:y1 + size_h, x1:x1 + size_w]
        blended = cv2.addWeighted(roi, 0.25, mini, 0.75, 0)
        frame[y1:y1 + size_h, x1:x1 + size_w] = blended
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x1 + size_w, y1 + size_h), (200, 200, 200), 1)

    # ── Frame annotation (텍스트 + 미니맵 한 번에) ───────────────────────────────

    def _annotate_frame(self, frame, px_pos, norm_pos, cat_ko, speed_str, prog, player_name="", indicator_alpha=1.0):
        ow, oh = frame.shape[1], frame.shape[0]

        # 페이드
        if prog < 0.1:
            frame = (frame * (prog / 0.1)).astype(np.uint8)
        elif prog > 0.9:
            frame = (frame * ((1.0 - prog) / 0.1)).astype(np.uint8)

        # 타겟 링 (알파 페이드 — 클립 시작 후 점점 사라짐)
        if px_pos is not None and indicator_alpha > 0.02:
            sx = max(45, min(ow - 45, int(px_pos[0])))
            sy = max(45, min(oh - 45, int(px_pos[1])))
            overlay = frame.copy()
            cv2.circle(overlay, (sx, sy), 40, (0, 215, 255), 3)
            cv2.circle(overlay, (sx, sy), 6, (0, 215, 255), -1)
            arm = 18
            for dx2, dy2 in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                ex, ey = sx + dx2 * (40 + arm), sy + dy2 * (40 + arm)
                cv2.line(overlay, (sx + dx2 * 40, ey), (ex, ey), (0, 215, 255), 2)
                cv2.line(overlay, (ex, sy + dy2 * 40), (ex, ey), (0, 215, 255), 2)
            cv2.addWeighted(overlay, float(indicator_alpha), frame, float(1.0 - indicator_alpha), 0, frame)

        # 하단 바 (반투명)
        bar_h = 54
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, oh - bar_h), (ow, oh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # PIL 텍스트
        texts = [
            (18, 12, "TAD AI", 30, (220, 40, 40)),
            (18, oh - bar_h + 11, cat_ko, 30, (255, 215, 0)),
        ]
        if player_name:
            # 우측 상단 선수명
            if HAS_PIL:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil)
                font_nm = _get_ko_font(26)
                try:
                    bbox = draw.textbbox((0, 0), player_name, font=font_nm)
                    pw = bbox[2] - bbox[0]
                except Exception:
                    pw = len(player_name) * 14
                draw.text((ow - pw - 18, 14), player_name, font=font_nm, fill=(255, 215, 0))
                for x, y, text, size, color in texts:
                    font = _get_ko_font(size)
                    rgb  = (int(color[2]), int(color[1]), int(color[0]))
                    draw.text((x, y), text, font=font, fill=rgb)
                if speed_str:
                    font_s = _get_ko_font(28)
                    try:
                        bbox = draw.textbbox((0, 0), speed_str, font=font_s)
                        sw = bbox[2] - bbox[0]
                    except Exception:
                        sw = len(speed_str) * 14
                    draw.text((ow - sw - 18, oh - bar_h + 13), speed_str, font=font_s, fill=(255, 255, 255))
                frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                return frame
        # player_name 없는 경우 또는 PIL 없는 경우
        if speed_str:
            texts.append((ow - len(speed_str) * 16 - 18, oh - bar_h + 11, speed_str, 28, (255, 255, 255)))
        _pil_text(frame, texts)
        return frame

    # ── Title card ─────────────────────────────────────────────────────────────

    def _title_frame(self, w, h, line1, line2=""):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for r in range(h):
            t = r / h
            img[r, :] = [int(8 + 22 * (1 - t)), 0, 0]

        if HAS_PIL:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)

            # TAD AI 로고
            font_logo = _get_ko_font(80)
            logo_text = "TAD AI"
            try:
                bbox = draw.textbbox((0, 0), logo_text, font=font_logo)
                tw   = bbox[2] - bbox[0]
                x    = (w - tw) // 2
            except Exception:
                x = w // 4
            draw.text((x, h // 2 - 90), logo_text, font=font_logo, fill=(220, 40, 40))

            # 이벤트 카테고리
            font_mid = _get_ko_font(44)
            try:
                bbox = draw.textbbox((0, 0), line1, font=font_mid)
                tw   = bbox[2] - bbox[0]
                x    = (w - tw) // 2
            except Exception:
                x = w // 4
            draw.text((x, h // 2 + 14), line1, font=font_mid, fill=(255, 215, 0))

            # 속도 등 부가 정보
            if line2:
                font_sm = _get_ko_font(30)
                try:
                    bbox = draw.textbbox((0, 0), line2, font=font_sm)
                    tw   = bbox[2] - bbox[0]
                    x    = (w - tw) // 2
                except Exception:
                    x = w // 4
                draw.text((x, h // 2 + 76), line2, font=font_sm, fill=(180, 180, 180))

            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            _put_center(img, "TAD AI", h // 2 - 40, 2.2, (220, 40, 40), 4)
            _put_center(img, _safe(line1).upper(), h // 2 + 20, 1.0, (255, 215, 0), 2)
            if line2:
                _put_center(img, _safe(line2).upper(), h // 2 + 65, 0.65, (180, 180, 180), 1)
        return img

    # ── Pass result helper ─────────────────────────────────────────────────────

    def _check_pass_result(self, f, ball_dict, teammates_by_frame, fps):
        """패스 결과: 1.5초 내 팀원이 볼 받으면 'success', 아니면 'fail', 팀 데이터 없으면 'unknown'"""
        if not teammates_by_frame:
            return "unknown"
        for k in range(1, int(fps * 1.5) + 1):
            nf = f + k
            if nf not in ball_dict:
                continue
            bpos = ball_dict[nf]
            for tpos in teammates_by_frame.get(nf, []):
                if np.hypot(tpos[0] - bpos[0], tpos[1] - bpos[1]) < 3.5:
                    return "success"
        return "fail"

    # ── Event detection ────────────────────────────────────────────────────────

    def _detect_events_by_position(self, tracks, target_id, ball_dict,
                                    fps=25, position="ST", target_team=-1):
        target_tracks = sorted(tracks[target_id], key=lambda t: t['frame'])
        if len(target_tracks) < 2:
            return []

        target_by_frame = {t['frame']: t for t in target_tracks}
        speed_map       = self._build_speed_map(target_tracks, fps)

        # 볼 데이터 희소성 체크 — 희소하면 임계값 완화
        ball_frames  = sorted(ball_dict.keys())
        ball_sparse  = len(ball_dict) < max(len(target_tracks) * 0.08, 1)
        dist_thr     = 4.5 if ball_sparse else 3.2

        ball_speed: dict = {}
        for i in range(1, len(ball_frames)):
            f      = ball_frames[i]
            p1, p2 = ball_dict[ball_frames[i - 1]], ball_dict[f]
            dx     = (p2[0] - p1[0]) * SCALE_X
            dy     = (p2[1] - p1[1]) * SCALE_Y
            ball_speed[f] = min(np.sqrt(dx ** 2 + dy ** 2) * fps * 3.6, 150.0)

        # 팀별 선수 위치 분류
        opponents_by_frame: dict = {}
        teammates_by_frame: dict = {}
        for tid, data in tracks.items():
            if tid == target_id:
                continue
            for t in data:
                if t.get('class') != 0:
                    continue
                f      = t['frame']
                t_team = t.get('team_id', -1)
                is_opp = (target_team == -1 or t_team == -1 or t_team != target_team)
                (opponents_by_frame if is_opp else teammates_by_frame).setdefault(f, []).append(t['pos'])

        events:  list = []
        min_gap: int  = int(fps * 3.5)

        def conflict(f):
            return any(abs(f - e['frame']) < min_gap for e in events)

        # ── 헬퍼: 골 구역 진입 확인 ─────────────────────────────────────────
        def in_goal_zone(bx, by, margin=2.0):
            return (bx < 6 + margin or bx > 94 - margin) and 15 < by < 35

        def ball_stayed_in_goal(f_enter, min_frames=2):
            return sum(1 for k in range(1, int(fps * 0.5))
                       if f_enter + k in ball_dict
                       and in_goal_zone(*ball_dict[f_enter + k])) >= min_frames

        # ── 1. GOAL (본인 슈팅 → 볼이 골 구역 진입 후 머무름) ───────────────
        for f in ball_frames:
            if conflict(f):
                continue
            bx, by = ball_dict[f]
            if not in_goal_zone(bx, by):
                continue
            if not ball_stayed_in_goal(f):
                continue
            # 최근 2초 내 볼 움직임
            if not any(ball_speed.get(f - k, 0) > 12 for k in range(1, int(fps * 2) + 1)):
                continue
            # 타겟이 4초 내 볼 근처에 있었는지
            for k in range(int(fps * 4)):
                fb = f - k
                if fb in target_by_frame and fb in ball_dict:
                    d = np.hypot(target_by_frame[fb]['pos'][0] - ball_dict[fb][0],
                                 target_by_frame[fb]['pos'][1] - ball_dict[fb][1])
                    if d < 6:
                        spd_v = max((ball_speed.get(f - j, 0) for j in range(1, 8)), default=0)
                        events.append({"frame": f, "category": "GOAL", "score": 3000,
                                       "speed_kmh": round(spd_v, 1), "section": "attack"})
                        break

        # ── 2. ASSIST (타겟이 패스 → 이후 볼이 골 구역 진입) ───────────────
        for t in target_tracks:
            f = t['frame']
            if f not in ball_dict or conflict(f):
                continue
            tp   = t['pos']
            bp   = ball_dict[f]
            dist = np.hypot(tp[0] - bp[0], tp[1] - bp[1])
            if dist > dist_thr:
                continue
            # 이미 GOAL로 잡힌 프레임 근처는 ASSIST로 중복 등록 안 함
            if any(e['category'] == 'GOAL' and abs(e['frame'] - f) < fps * 4 for e in events):
                continue
            # 볼이 타겟에서 빠르게 이탈 (패스)
            pass_frame = None
            for k in range(1, int(fps * 0.6)):
                nf = f + k
                if nf not in ball_dict:
                    continue
                nf_target_pos = target_by_frame[nf]['pos'] if nf in target_by_frame else tp
                nd = np.hypot(nf_target_pos[0] - ball_dict[nf][0],
                              nf_target_pos[1] - ball_dict[nf][1])
                if nd > 5 and ball_speed.get(nf, 0) > 18:
                    pass_frame = nf
                    break
            if pass_frame is None:
                continue
            # 패스 후 3초 내 볼이 골 구역 진입
            for gk in range(1, int(fps * 3) + 1):
                gf = pass_frame + gk
                if gf not in ball_dict:
                    continue
                if in_goal_zone(*ball_dict[gf]) and ball_stayed_in_goal(gf, min_frames=1):
                    events.append({"frame": f, "category": "ASSIST", "score": 2500,
                                   "speed_kmh": round(ball_speed.get(pass_frame, 0), 1),
                                   "section": "attack"})
                    break

        # ── 3. 볼 터치 기반 이벤트 (포지션 제한 없음) ───────────────────────
        for t in target_tracks:
            f = t['frame']
            if f not in ball_dict or conflict(f):
                continue
            tp   = t.get('pos', [0, 0])
            bp   = ball_dict[f]
            dist = np.hypot(tp[0] - bp[0], tp[1] - bp[1])
            if dist > dist_thr:
                continue

            spd     = speed_map.get(f, 0)
            field_x = tp[0]
            in_att  = field_x > 63 or field_x < 37
            in_def  = field_x < 28 or field_x > 72

            future_ball = [ball_dict[f + k] for k in range(1, int(fps * 1.2)) if f + k in ball_dict]
            ball_traveled_m = 0.0
            if future_ball:
                dx = (future_ball[-1][0] - bp[0]) * SCALE_X
                dy = (future_ball[-1][1] - bp[1]) * SCALE_Y
                ball_traveled_m = np.sqrt(dx ** 2 + dy ** 2)

            bspd_after  = max((ball_speed.get(f + k, 0) for k in range(1, int(fps * 0.5))
                               if f + k in ball_speed), default=0)
            bspd_before = max((ball_speed.get(f - k, 0) for k in range(1, int(fps * 0.4))
                               if f - k in ball_speed), default=0)

            opp_had_ball = any(
                np.hypot(op[0] - ball_dict.get(f - k, bp)[0],
                         op[1] - ball_dict.get(f - k, bp)[1]) < 3.0
                for k in range(1, int(fps * 0.8))
                for op in opponents_by_frame.get(f - k, [])
                if f - k in ball_dict
            )
            if not ball_sparse:
                # 볼이 감지된 경우: 볼 근접 지속으로 드리블 판단
                sustained = sum(
                    1 for k in range(1, int(fps * 0.6))
                    if f - k in ball_dict and f - k in target_by_frame
                    and np.hypot(target_by_frame[f - k]['pos'][0] - ball_dict[f - k][0],
                                 target_by_frame[f - k]['pos'][1] - ball_dict[f - k][1]) < dist_thr
                ) >= int(fps * 0.25)
            else:
                # 볼 희소: 10~24 km/h 속도 구간 지속으로 드리블 판단 (핸드폰 영상 대응)
                sustained = sum(
                    1 for k in range(1, int(fps * 1.2))
                    if 10.0 < speed_map.get(f - k, 0) < 24.0
                ) >= int(fps * 0.7)

            # ── SHOT: 포지션 무관, 공격 지역 + 볼이 빠르게 골 방향으로 ──────
            if in_att and bspd_after > 28:
                toward_goal = future_ball and (future_ball[-1][0] < 10 or future_ball[-1][0] > 90)
                if toward_goal or bspd_after > 55:
                    events.append({"frame": f, "category": "SHOT", "score": 1800,
                                   "speed_kmh": round(spd, 1), "section": "attack"})
                    continue

            # ── TACKLE: 포지션 무관, 상대가 볼 가졌다가 뺏김 ─────────────────
            if opp_had_ball and spd > 4:
                events.append({"frame": f, "category": "TACKLE", "score": 1500,
                               "speed_kmh": round(spd, 1), "section": "defense"})
                continue

            # ── INTERCEPTION: 상대 패스 중간 차단 ────────────────────────────
            if opp_had_ball and bspd_before > 16 and spd < 14:
                events.append({"frame": f, "category": "INTERCEPTION", "score": 1400,
                               "speed_kmh": round(spd, 1), "section": "defense"})
                continue

            # ── CLEARANCE: 수비 지역에서 강하게 걷어냄 ───────────────────────
            if in_def and ball_traveled_m > 14 and bspd_after > 22:
                events.append({"frame": f, "category": "CLEARANCE", "score": 1200,
                               "speed_kmh": round(spd, 1), "section": "defense"})
                continue

            # ── DRIBBLE: 볼 유지하며 이동 (조깅~고속 범위, 스프린트 제외) ────────
            if sustained and 10 < spd < 26:
                events.append({"frame": f, "category": "DRIBBLE", "score": 1000,
                               "speed_kmh": round(spd, 1), "section": "attack"})
                continue

            # ── LONG PASS / THROUGH BALL ───────────────────────────────────
            if ball_traveled_m > 16 and bspd_after > 22:
                toward_att = future_ball and future_ball[-1][0] > field_x + 12
                cat = "THROUGH BALL" if (in_att and toward_att) else "LONG PASS"
                pr = self._check_pass_result(f, ball_dict, teammates_by_frame, fps)
                events.append({"frame": f, "category": cat, "score": 750,
                               "speed_kmh": round(spd, 1), "section": "attack",
                               "pass_result": pr})
                continue

            # ── SHORT PASS ────────────────────────────────────────────────
            if ball_traveled_m > 4.0 and bspd_after > 12 and spd < 15:
                pr = self._check_pass_result(f, ball_dict, teammates_by_frame, fps)
                events.append({"frame": f, "category": "PASS", "score": 500,
                               "speed_kmh": round(spd, 1), "section": "midfield",
                               "pass_result": pr})
                continue

            # ── BALL CONTROL: 빠른 볼 받아서 안정적으로 제어 ─────────────────
            if bspd_before > 16 and dist < 2.0:
                events.append({"frame": f, "category": "CONTROL", "score": 350,
                               "speed_kmh": round(spd, 1), "section": "midfield"})
                continue

            # ── BALL TOUCH (fallback) ─────────────────────────────────────
            events.append({"frame": f, "category": "BALL TOUCH", "score": 250,
                           "speed_kmh": round(spd, 1), "section": "midfield"})

        # ── 4. BALL LOST ─────────────────────────────────────────────────────
        for t in target_tracks:
            f = t['frame']
            if f not in ball_dict:
                continue
            d = np.hypot(t['pos'][0] - ball_dict[f][0], t['pos'][1] - ball_dict[f][1])
            if d > 2.5:
                continue
            for k in range(1, int(fps * 1.5)):
                nf = f + k
                if nf not in ball_dict:
                    continue
                if nf in target_by_frame:
                    nd = np.hypot(target_by_frame[nf]['pos'][0] - ball_dict[nf][0],
                                  target_by_frame[nf]['pos'][1] - ball_dict[nf][1])
                    if nd < 3:
                        break
                opp_got = any(np.hypot(op[0] - ball_dict[nf][0], op[1] - ball_dict[nf][1]) < 2.5
                              for op in opponents_by_frame.get(nf, []))
                if opp_got and not conflict(nf):
                    events.append({"frame": nf, "category": "BALL LOST", "score": 100,
                                   "speed_kmh": 0, "section": "bad"})
                    break

        # ── 5. SPRINT ────────────────────────────────────────────────────────
        last_sp = -int(fps * 8)
        for i in range(1, len(target_tracks)):
            f   = target_tracks[i]['frame']
            spd = speed_map.get(f, 0)
            if spd > 22 and f - last_sp > int(fps * 8) and not conflict(f):
                events.append({"frame": f, "category": "SPRINT", "score": 300,
                               "speed_kmh": round(spd, 1), "section": "midfield"})
                last_sp = f

        # ── 6. MOTION SHOT (볼 미감지 시 타겟 모션 기반 슈팅 감지) ─────────────
        # 공격 지역에서 뛰다 급감속 + 골 방향 이동 = 슈팅 동작으로 판정.
        # 볼 추적이 희소한 핸드폰 영상에서도 슈팅 하이라이트를 잡기 위함.
        _MS_WIN_F     = int(fps * 0.7)   # 전후 0.7초 프레임 수
        _MS_MIN_SPD   = 11.0             # 슈팅 직전 최소 속도 (km/h)
        _MS_DECEL_THR = 0.55             # 이 비율 아래로 감속되어야 슈팅 판정

        for si, st in enumerate(target_tracks):
            sf2 = st['frame']
            tp2 = st['pos']

            if conflict(sf2):
                continue
            # 이미 SHOT/GOAL 이벤트가 근처에 있으면 중복 스킵
            if any(e['category'] in ('SHOT', 'GOAL') and abs(e['frame'] - sf2) < int(fps * 3)
                   for e in events):
                continue
            # 공격 지역에서만 감지 (필드 중앙 기준 60% 이상 or 40% 이하)
            if not (tp2[0] > 58 or tp2[0] < 42):
                continue

            # 슈팅 전 0.7초 평균 속도
            pre_spds = [v for fk, v in speed_map.items() if sf2 - _MS_WIN_F <= fk < sf2]
            avg_pre  = float(np.mean(pre_spds)) if pre_spds else 0
            if avg_pre < _MS_MIN_SPD:
                continue

            # 슈팅 후 0.7초 평균 속도 (급감속 확인)
            post_spds = [v for fk, v in speed_map.items() if sf2 < fk <= sf2 + _MS_WIN_F]
            avg_post  = float(np.mean(post_spds)) if post_spds else avg_pre
            if avg_post > avg_pre * _MS_DECEL_THR:
                continue  # 감속 불충분

            # 골 방향 이동 확인 (0.7초 전 위치 → 현재 위치 벡터)
            prev_pos = None
            for dlt in range(0, _MS_WIN_F + 1):
                if (sf2 - dlt) in target_by_frame:
                    prev_pos = target_by_frame[sf2 - dlt]['pos']
                    break
            if prev_pos is None:
                continue
            dx = tp2[0] - prev_pos[0]
            toward_goal = (tp2[0] > 50 and dx > 1.0) or (tp2[0] < 50 and dx < -1.0)
            if not toward_goal:
                continue

            events.append({
                "frame": sf2, "category": "SHOT", "score": 1500,
                "speed_kmh": round(avg_pre, 1), "section": "attack",
            })

        # ── 7. BALL CONTEST (공 다툼: 상대 밀집 + 저속 지속) ─────────────────
        # 타겟 주변 3.5m 내 상대 선수 2명 이상이 0.8초 이상 머무를 때 공 다툼 판정.
        _BC_OPP_RADIUS = 3.5           # 상대 감지 반경 (정규화 좌표)
        _BC_MIN_OPP    = 2             # 최소 상대 수
        _BC_SPD_MAX    = 13.0          # 다툼 중 타겟 최대 속도
        _BC_SUSTAIN_F  = int(fps * 0.8)  # 지속 판정 프레임 수
        last_contest   = -int(fps * 8)

        for ct in target_tracks:
            cf   = ct['frame']
            ctp  = ct['pos']
            cspd = speed_map.get(cf, 0)

            if conflict(cf) or cf - last_contest < int(fps * 8):
                continue
            if cspd > _BC_SPD_MAX:
                continue

            # 타겟 주변 상대 선수 수
            opp_now = sum(
                1 for op in opponents_by_frame.get(cf, [])
                if np.hypot(op[0] - ctp[0], op[1] - ctp[1]) < _BC_OPP_RADIUS
            )
            if opp_now < _BC_MIN_OPP:
                continue

            # 볼이 감지된 경우 볼 근접도 추가 확인 (너무 멀면 스킵)
            if not ball_sparse and cf in ball_dict:
                bd = np.hypot(ctp[0] - ball_dict[cf][0], ctp[1] - ball_dict[cf][1])
                if bd > dist_thr * 2.5:
                    continue

            # 상대가 0.8초 이상 근처에 있어야 (순간적 교차 제외)
            sustained_opp = sum(
                1 for k in range(1, _BC_SUSTAIN_F + 1)
                if any(
                    np.hypot(op[0] - ctp[0], op[1] - ctp[1]) < _BC_OPP_RADIUS * 1.4
                    for op in opponents_by_frame.get(cf - k, [])
                )
            ) >= int(_BC_SUSTAIN_F * 0.5)
            if not sustained_opp:
                continue

            events.append({
                "frame": cf, "category": "BALL CONTEST", "score": 1100,
                "speed_kmh": round(cspd, 1), "section": "defense",
            })
            last_contest = cf

        # ── 8. ACTIVE PLAY Fallback (볼 희소 or 이벤트 부족 시) ───────────────
        if ball_sparse or len(events) < 5:
            win = int(fps * 6)
            for i in range(0, len(target_tracks) - win, win):
                f = target_tracks[i + win // 2]['frame']
                if conflict(f):
                    continue
                speeds_seg = [
                    self._speed_kmh(target_tracks[j - 1]['pos'], target_tracks[j]['pos'], fps)
                    for j in range(i + 1, min(i + win, len(target_tracks)))
                ]
                speeds_seg = [s for s in speeds_seg if s < 40]
                avg_spd = float(np.mean(speeds_seg)) if speeds_seg else 0
                if avg_spd < 5:
                    continue
                events.append({"frame": f, "category": "ACTIVE PLAY",
                               "score": avg_spd * len(speeds_seg),
                               "speed_kmh": round(avg_spd, 1), "section": "midfield"})

        # ── 데드볼 필터 ──────────────────────────────────────────────────────
        def is_dead_ball(ev):
            f = ev['frame']
            pre_s, _ = CLIP_WINDOWS.get(ev['category'], DEFAULT_WINDOW)
            sf = max(0, int(f - fps * pre_s))
            ef = int(f + fps * 1.0)
            if ball_dict and not ball_sparse:
                ball_spds = [ball_speed.get(bf, 0) for bf in range(sf, ef) if bf in ball_speed]
                if ball_spds and max(ball_spds) < 5:
                    return True
            spd_in = [speed_map.get(tf, 0) for tf in range(sf, ef) if tf in speed_map]
            if spd_in and max(spd_in) < 3:
                return True
            return False

        events = [ev for ev in events if not is_dead_ball(ev)]
        events = sorted(events, key=lambda x: x['score'], reverse=True)[:20]
        events = sorted(events, key=lambda x: x['frame'])
        return events

    # ── Clip extraction ────────────────────────────────────────────────────────

    def extract_combined_highlights(self, video_path, tracks, target_id,
                                     fps=25, position="ST", target_team=-1,
                                     player_name="PLAYER"):
        if target_id not in tracks:
            return [], [], {"pass_total": 0, "pass_success": None, "pass_success_rate": None}
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            cap.release()
        except Exception:
            pass

        px_dict   = {t['frame']: t['pos_px'] for t in tracks[target_id] if 'pos_px' in t}
        norm_dict = {t['frame']: t['pos']    for t in tracks[target_id] if 'pos'    in t}

        ball_tracks = [t for tid, data in tracks.items()
                       if data and data[0].get('class') == 32 for t in data]
        ball_dict   = {b['frame']: b['pos'] for b in ball_tracks}

        events = self._detect_events_by_position(
            tracks, target_id, ball_dict, fps, position, target_team
        )

        # 타겟 선수가 실제로 해당 장면에 있는지 검증
        validated = []
        for ev in events:
            f = ev['frame']
            # 이벤트 시점 ±2초 내에 타겟 추적 데이터가 있어야 함
            nearby = any(abs(pf - f) <= fps * 2 for pf in px_dict)
            if not nearby:
                continue
            # 클립 구간에서 타겟이 20% 이상 추적되어야 함
            pre_s, post_s = CLIP_WINDOWS.get(ev['category'], DEFAULT_WINDOW)
            sf_v = max(0, int(f - fps * pre_s))
            ef_v = int(f + fps * post_s)
            clip_len = max(ef_v - sf_v, 1)
            tracked_in = sum(1 for pf in px_dict if sf_v <= pf <= ef_v)
            if tracked_in / clip_len < 0.20:
                continue
            validated.append(ev)
        events = validated

        highlights = []
        for i, ev in enumerate(events):
            pre_s, post_s = CLIP_WINDOWS.get(ev['category'], DEFAULT_WINDOW)
            start_f = max(0, int(ev['frame'] - fps * pre_s))
            end_f   = int(ev['frame'] + fps * post_s)

            clip_name = f"tad_hl_{target_id}_{i}.mp4"
            clip_path = os.path.join(self.highlight_dir, clip_name)

            px_seg   = {f: px_dict[f]   for f in range(start_f, end_f + 1) if f in px_dict}
            norm_seg = {f: norm_dict[f]  for f in range(start_f, end_f + 1) if f in norm_dict}

            if self._save_clip(video_path, start_f, end_f, clip_path,
                               px_seg, norm_seg, ev, fps, player_name):
                highlights.append({
                    "url":       f"/static/highlights/{clip_name}",
                    "category":  ev['category'],
                    "section":   ev.get("section", "midfield"),
                    "frame":     int(ev['frame']),
                    "speed_kmh": ev.get("speed_kmh", 0),
                })

        # ── 패스 성공률 집계 ──────────────────────────────────────────────────
        pass_cats = {'PASS', 'LONG PASS', 'THROUGH BALL'}
        pass_events = [e for e in events if e['category'] in pass_cats]
        assist_count = sum(1 for e in events if e['category'] == 'ASSIST')
        total = len(pass_events) + assist_count
        if any(e.get('pass_result') == 'unknown' for e in pass_events):
            pass_stats = {"pass_total": total, "pass_success": None, "pass_success_rate": None}
        else:
            success = sum(1 for e in pass_events if e.get('pass_result') == 'success') + assist_count
            rate = round(success / total * 100) if total > 0 else None
            pass_stats = {"pass_total": total, "pass_success": success, "pass_success_rate": rate}

        # ── 볼 터치 횟수 집계 ─────────────────────────────────────────────────
        # 볼이 충분히 감지된 경우에만 카운트 (희소 시 None)
        ball_sparse_here = len(ball_dict) < max(len(tracks.get(target_id, [])) * 0.08, 1)
        if not ball_sparse_here and ball_dict:
            target_data = tracks.get(target_id, [])
            _dist_thr = 4.5 if ball_sparse_here else 3.2
            btc = sum(
                1 for t in target_data
                if t['frame'] in ball_dict
                and np.hypot(t['pos'][0] - ball_dict[t['frame']][0],
                             t['pos'][1] - ball_dict[t['frame']][1]) < _dist_thr
            )
            pass_stats['ball_touch_count'] = btc
        else:
            pass_stats['ball_touch_count'] = None

        return highlights, events, pass_stats

    # ── Single clip writer ─────────────────────────────────────────────────────

    def _save_clip(self, video_path, start_f, end_f, out_path,
                   px_tracks, norm_tracks=None, event=None, fps=25, player_name=""):
        norm_tracks = norm_tracks or {}
        try:
            cap  = cv2.VideoCapture(video_path)
            fps  = cap.get(cv2.CAP_PROP_FPS) or fps
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 최대 1920×1080 유지, 비율 보존
            max_w, max_h = 1280, 720
            ratio = min(max_w / max(src_w, 1), max_h / max(src_h, 1), 1.0)
            ow, oh = int(src_w * ratio), int(src_h * ratio)
            # 짝수 보장 (H.264 요구사항)
            ow = ow - (ow % 2)
            oh = oh - (oh % 2)

            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (ow, oh))
            if not out.isOpened():
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow, oh))

            cat      = event.get('category', 'HIGHLIGHT') if event else 'HIGHLIGHT'
            cat_ko   = CATEGORY_KO.get(cat, cat)
            spd_v    = event.get('speed_kmh', 0) if event else 0
            spd_str  = f"{spd_v:.0f} km/h" if spd_v > 1 else ""
            is_slow  = cat in ('GOAL', 'SHOT')

            # 타이틀 카드 (0.4초)
            title = self._title_frame(ow, oh, cat_ko, spd_str)
            for k in range(int(fps * 0.4)):
                a = min(1.0, k / max(1, fps * 0.2))
                out.write((title * a).astype(np.uint8))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            total = max(end_f - start_f, 1)
            lx = ly = None

            for fi in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret:
                    break
                hf, wf = frame.shape[:2]

                raw = px_tracks.get(fi)
                if raw is not None:
                    cx, cy = float(raw[0]), float(raw[1])
                    if lx is not None:
                        cx = lx * 0.72 + cx * 0.28
                        cy = ly * 0.72 + cy * 0.28
                    lx, ly = cx, cy
                else:
                    cx = lx if lx is not None else wf / 2
                    cy = ly if ly is not None else hf / 2

                res      = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
                prog     = (fi - start_f) / total
                norm_pos = norm_tracks.get(fi)

                # 픽셀 좌표 스케일 변환
                px_scaled = [cx * ow / wf, cy * oh / hf] if cx is not None else None

                # 타겟 인디케이터: 1.5초 표시 후 0.7초 페이드 아웃
                clip_t = (fi - start_f) / fps
                if clip_t < 1.5:
                    ind_a = min(1.0, clip_t / 0.25)
                elif clip_t < 2.2:
                    ind_a = 1.0 - (clip_t - 1.5) / 0.7
                else:
                    ind_a = 0.0

                self._annotate_frame(res, px_scaled, norm_pos, cat_ko, spd_str, prog, player_name, ind_a)

                # 슬로우모션 (핵심 구간 프레임 복제)
                if is_slow and 0.35 < prog < 0.65:
                    out.write(res)
                out.write(res)

            cap.release()
            out.release()
            return os.path.exists(out_path) and os.path.getsize(out_path) > 1000
        except Exception as e:
            print(f"Clip error: {e}")
            import traceback; traceback.print_exc()
            return False

    # ── Master sizzle reel ─────────────────────────────────────────────────────

    def generate_master_sizzle_reel(self, video_path, events, tracks,
                                     target_id, session_id, player_name="PLAYER"):
        if not events or target_id not in tracks:
            return None

        px_dict   = {t['frame']: t['pos_px'] for t in tracks[target_id] if 'pos_px' in t}
        norm_dict = {t['frame']: t['pos']    for t in tracks[target_id] if 'pos'    in t}
        out_name  = f"{session_id}_tad_master.mp4"
        out_path  = os.path.join(self.highlight_dir, out_name)

        try:
            cap  = cv2.VideoCapture(video_path)
            fps  = cap.get(cv2.CAP_PROP_FPS) or 25
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ratio = min(1280 / max(src_w, 1), 720 / max(src_h, 1), 1.0)
            ow = int(src_w * ratio) - (int(src_w * ratio) % 2)
            oh = int(src_h * ratio) - (int(src_h * ratio) % 2)

            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (ow, oh))
            if not out.isOpened():
                out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ow, oh))

            # ── 인트로 (1.5초) ────────────────────────────────────────────
            intro = self._title_frame(ow, oh, player_name, "OFFICIAL HIGHLIGHT FILM")
            for k in range(int(fps * 1.5)):
                a = min(1.0, k / max(1, fps * 0.5))
                out.write((intro * a).astype(np.uint8))

            lx = ly = None
            black = np.zeros((oh, ow, 3), dtype=np.uint8)

            for ev in events:
                pre_s, post_s = CLIP_WINDOWS.get(ev['category'], DEFAULT_WINDOW)
                sf    = max(0, int(ev['frame'] - fps * pre_s))
                ef    = int(ev['frame'] + fps * post_s)
                cat   = ev.get('category', 'HIGHLIGHT')
                cat_ko = CATEGORY_KO.get(cat, cat)
                spd_v  = ev.get('speed_kmh', 0)
                spd_str = f"{spd_v:.0f} km/h" if spd_v > 1 else ""
                is_slow = cat in ('BALL TOUCH', 'DRIBBLE', 'SPRINT', 'GOAL', 'SHOT')
                total   = max(ef - sf, 1)

                cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
                for fi in range(sf, ef):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    hf, wf = frame.shape[:2]

                    raw = px_dict.get(fi)
                    if raw is not None:
                        cx, cy = float(raw[0]), float(raw[1])
                        if lx is not None:
                            cx = lx * 0.72 + cx * 0.28
                            cy = ly * 0.72 + cy * 0.28
                        lx, ly = cx, cy
                    else:
                        cx = lx if lx is not None else wf / 2
                        cy = ly if ly is not None else hf / 2

                    res      = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
                    prog     = (fi - sf) / total
                    norm_pos = norm_dict.get(fi)
                    px_scaled = [cx * ow / wf, cy * oh / hf] if cx is not None else None

                    # 타겟 인디케이터: 1.5초 표시 후 0.7초 페이드 아웃
                    clip_t = (fi - sf) / fps
                    if clip_t < 1.5:
                        ind_a = min(1.0, clip_t / 0.25)
                    elif clip_t < 2.2:
                        ind_a = 1.0 - (clip_t - 1.5) / 0.7
                    else:
                        ind_a = 0.0

                    self._annotate_frame(res, px_scaled, norm_pos, cat_ko, spd_str, prog, player_name, ind_a)

                    if is_slow and 0.35 < prog < 0.65:
                        out.write(res)
                    out.write(res)

                # 0.15초 블랙 컷
                for _ in range(int(fps * 0.15)):
                    out.write(black)

            # ── 아웃트로 (1.5초) ──────────────────────────────────────────
            outro = self._title_frame(ow, oh, "분석 완료", "TAD AI POWERED")
            for k in range(int(fps * 1.5)):
                a = max(0.0, 1.0 - k / max(1, fps * 0.8))
                out.write((outro * a).astype(np.uint8))

            cap.release()
            out.release()
            return f"/static/highlights/{out_name}" if os.path.exists(out_path) else None

        except Exception as e:
            print(f"Master Reel Error: {e}")
            import traceback; traceback.print_exc()
            return None

    # ── FIFA 스타일 선수 카드 이미지 생성 ──────────────────────────────────────

    def generate_player_card(self, stats, player_name, position, session_id,
                             photo_path=None):
        """340×500px FIFA 스타일 선수 카드 PNG 생성."""
        if not HAS_PIL:
            return None
        try:
            W, H = 340, 500
            # 진한 남색 배경
            bg = np.zeros((H, W, 3), dtype=np.uint8)
            for y in range(H):
                t = y / H
                bg[y, :] = [int(8 + 18 * t), 0, int(28 + 22 * (1 - t))]
            pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)

            # 상단/하단 강조 바
            draw.rectangle([(0, 0), (W, 7)],   fill=(220, 26, 26))
            draw.rectangle([(0, H - 7), (W, H)], fill=(220, 26, 26))

            # 스탯
            pac  = int(stats.get('PAC', 60))
            sho  = int(stats.get('SHO', 60))
            pas_ = int(stats.get('PAS', 60))
            dri  = int(stats.get('DRI', 60))
            phy  = int(stats.get('PHY', 60))
            overall = (pac + sho + pas_ + dri + phy) // 5
            dist = float(stats.get('total_distance_km', 0))
            spd  = float(stats.get('top_speed_kmh', 0))
            spr  = int(stats.get('sprint_count', 0))

            f_big  = _get_ko_font(64)
            f_pos  = _get_ko_font(24)
            f_name = _get_ko_font(32)
            f_lbl  = _get_ko_font(19)
            f_val  = _get_ko_font(21)
            f_mini = _get_ko_font(17)
            f_brand = _get_ko_font(18)

            # 종합 평점 + 포지션 (좌상단)
            draw.text((14, 14),  str(overall),         font=f_big,  fill=(255, 215, 0))
            draw.text((16, 86),  position.upper()[:3], font=f_pos,  fill=(255, 215, 0))

            # 선수 사진 or 실루엣
            ph_cx, ph_y, ph_r = W // 2, 48, 64
            if photo_path and os.path.exists(photo_path):
                try:
                    ph = Image.open(photo_path).convert('RGB').resize(
                        (ph_r * 2, ph_r * 2), Image.LANCZOS)
                    mask_img = Image.new('L', (ph_r * 2, ph_r * 2), 0)
                    ImageDraw.Draw(mask_img).ellipse(
                        (0, 0, ph_r * 2, ph_r * 2), fill=255)
                    pil.paste(ph, (ph_cx - ph_r, ph_y), mask_img)
                except Exception:
                    pass
            else:
                # 기본 실루엣 원
                draw.ellipse(
                    [(ph_cx - ph_r, ph_y), (ph_cx + ph_r, ph_y + ph_r * 2)],
                    fill=(35, 35, 55), outline=(255, 215, 0), width=2)
                # 머리
                hr = 22
                draw.ellipse(
                    [(ph_cx - hr, ph_y + ph_r - hr * 2 - 18),
                     (ph_cx + hr, ph_y + ph_r - 18)],
                    fill=(80, 80, 100))
                # 몸통
                draw.rectangle(
                    [(ph_cx - hr - 4, ph_y + ph_r - 16),
                     (ph_cx + hr + 4, ph_y + ph_r * 2 - 10)],
                    fill=(80, 80, 100))

            # 선수 이름
            name_y = ph_y + ph_r * 2 + 14
            try:
                bbox = draw.textbbox((0, 0), player_name, font=f_name)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = len(player_name) * 15
            draw.text(((W - tw) // 2, name_y), player_name,
                      font=f_name, fill=(255, 255, 255))

            # 구분선
            sep_y = name_y + 44
            draw.line([(28, sep_y), (W - 28, sep_y)], fill=(55, 55, 75), width=1)

            # 스탯 바 5개 (2열)
            stat_list = [('PAC', pac), ('SHO', sho),
                         ('PAS', pas_), ('DRI', dri),
                         ('PHY', phy)]
            bx_cols = [28, 184]
            bar_w   = 118
            bar_sy  = sep_y + 16
            bar_row_h = 42

            for i, (lbl, val) in enumerate(stat_list):
                col = i % 2
                row = i // 2
                bx  = bx_cols[col]
                by  = bar_sy + row * bar_row_h

                col_c = (50, 220, 80) if val >= 80 else \
                        (255, 170, 0) if val >= 65 else (220, 60, 60)

                draw.text((bx, by), lbl, font=f_lbl, fill=(150, 150, 175))
                # 배경 바
                draw.rectangle([(bx, by + 24), (bx + bar_w, by + 33)],
                                fill=(28, 28, 45))
                # 채운 바
                filled = max(1, int(bar_w * val / 99))
                draw.rectangle([(bx, by + 24), (bx + filled, by + 33)],
                                fill=col_c)
                # 수치
                val_s = str(val)
                try:
                    vb = draw.textbbox((0, 0), val_s, font=f_val)
                    vw = vb[2] - vb[0]
                except Exception:
                    vw = len(val_s) * 10
                draw.text((bx + bar_w - vw, by), val_s,
                          font=f_val, fill=(255, 255, 255))

            # 퍼포먼스 3줄
            perf_y = bar_sy + 3 * bar_row_h + 10
            draw.line([(28, perf_y), (W - 28, perf_y)], fill=(55, 55, 75), width=1)
            perf_y += 10
            perf_items = [(f"{dist:.1f}km", "이동"),
                          (f"{spd:.0f}km/h", "최고속"),
                          (f"{spr}회", "스프")]
            px_list = [30, 140, 250]
            for xi, (v_txt, l_txt) in zip(px_list, perf_items):
                draw.text((xi, perf_y),      l_txt, font=f_mini, fill=(120, 120, 145))
                draw.text((xi, perf_y + 18), v_txt, font=f_mini, fill=(210, 210, 230))

            # TAD AI 워터마크
            draw.text((W - 70, H - 26), "TAD AI", font=f_brand, fill=(220, 26, 26))

            filename = f"{session_id}_card.png"
            filepath = os.path.join(self.highlight_dir, filename)
            cv2.imwrite(filepath, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
            return f"/static/highlights/{filename}"
        except Exception as e:
            print(f"[TAD] 카드 생성 오류: {e}")
            return None

    # ── PDF 성장 리포트 생성 (Pillow 기반, 다중 페이지) ────────────────────────

    def generate_pdf_report(self, stats, ai_comment, player_name, position,
                             history, session_id):
        """
        Pillow로 A4 크기 이미지를 만들어 PDF로 저장.
        history: [{"date":str, "stats":dict}, ...] 최근 경기 목록.
        """
        if not HAS_PIL:
            return None
        try:
            W, H = 794, 1123   # A4 @ 96dpi

            def new_page():
                bg = Image.new('RGB', (W, H), (10, 10, 18))
                d  = ImageDraw.Draw(bg)
                d.rectangle([(0, 0), (W, 10)], fill=(220, 26, 26))
                d.rectangle([(0, H - 10), (W, H)], fill=(220, 26, 26))
                return bg, d

            f_h1   = _get_ko_font(52)
            f_h2   = _get_ko_font(30)
            f_body = _get_ko_font(24)
            f_sm   = _get_ko_font(20)
            f_xs   = _get_ko_font(17)

            pages = []

            # ── 1페이지: 스탯 + AI 리포트 ─────────────────────────────────────
            p1, d1 = new_page()
            d1.text((60, 28),  "TAD AI 성장 리포트",  font=f_h1,   fill=(255, 215, 0))
            d1.text((60, 96),  player_name,            font=f_h2,   fill=(255, 255, 255))
            d1.text((60, 138), f"{position.upper()}  |  "
                               f"{datetime.now().strftime('%Y년 %m월 %d일')}",
                    font=f_sm, fill=(140, 140, 165))
            d1.line([(60, 180), (W - 60, 180)], fill=(55, 55, 75), width=1)

            # 종합 평점
            pac  = int(stats.get('PAC', 60))
            sho  = int(stats.get('SHO', 60))
            pas_ = int(stats.get('PAS', 60))
            dri  = int(stats.get('DRI', 60))
            phy  = int(stats.get('PHY', 60))
            overall = (pac + sho + pas_ + dri + phy) // 5
            dist = float(stats.get('total_distance_km', 0))
            spd  = float(stats.get('top_speed_kmh', 0))
            spr  = int(stats.get('sprint_count', 0))
            psr  = stats.get('pass_success_rate')
            btc  = stats.get('ball_touch_count')

            f_rate = _get_ko_font(80)
            d1.text((60, 195), str(overall), font=f_rate, fill=(255, 215, 0))
            d1.text((60, 284), "종합 평점",  font=f_sm,   fill=(160, 160, 185))

            # 스탯 바
            stat_list = [('PAC 스피드', pac), ('SHO 슈팅', sho),
                         ('PAS 패스', pas_),  ('DRI 드리블', dri),
                         ('PHY 체력', phy)]
            bar_x, bar_w = 280, 320
            for i, (lbl, val) in enumerate(stat_list):
                by = 310 + i * 52
                c  = (50, 220, 80) if val >= 80 else \
                     (255, 170, 0) if val >= 65 else (220, 60, 60)
                d1.text((60,  by), lbl, font=f_body, fill=(180, 180, 200))
                d1.rectangle([(bar_x, by + 6), (bar_x + bar_w, by + 28)],
                             fill=(28, 28, 45))
                filled = max(1, int(bar_w * val / 99))
                d1.rectangle([(bar_x, by + 6), (bar_x + filled, by + 28)],
                             fill=c)
                d1.text((bar_x + bar_w + 14, by), str(val),
                        font=f_body, fill=(255, 255, 255))

            # 퍼포먼스 수치 행
            perf_y = 310 + 5 * 52 + 16
            d1.line([(60, perf_y), (W - 60, perf_y)], fill=(55, 55, 75), width=1)
            perf_y += 14
            items = [("이동거리", f"{dist:.2f} km"),
                     ("최고속도", f"{spd:.1f} km/h"),
                     ("스프린트", f"{spr}회")]
            if psr is not None:
                items.append(("패스성공률", f"{psr}%"))
            if btc is not None:
                items.append(("볼 터치", f"{btc}회"))
            cw = (W - 120) // max(len(items), 1)
            for j, (lbl2, val2) in enumerate(items):
                px = 60 + j * cw
                d1.text((px, perf_y),      lbl2, font=f_xs, fill=(130, 130, 155))
                d1.text((px, perf_y + 22), val2, font=f_body, fill=(255, 255, 255))

            # AI 스카우팅 노트
            note_y = perf_y + 76
            d1.line([(60, note_y), (W - 60, note_y)], fill=(55, 55, 75), width=1)
            d1.text((60, note_y + 14), "AI 스카우팅 리포트", font=f_h2,
                    fill=(255, 215, 0))
            note = ai_comment.get('scouting_note', '')
            # 단순 줄바꿈 (30자 기준)
            lines_n = [note[i:i+30] for i in range(0, min(len(note), 180), 30)]
            for li, ln in enumerate(lines_n):
                d1.text((60, note_y + 56 + li * 28), ln,
                        font=f_sm, fill=(195, 195, 215))

            pros_y = note_y + 56 + len(lines_n) * 28 + 16
            d1.text((60, pros_y), "강점", font=f_body, fill=(50, 220, 80))
            for ki, pro in enumerate(ai_comment.get('pros', [])[:4]):
                d1.text((60, pros_y + 28 + ki * 26), f"• {pro[:36]}",
                        font=f_xs, fill=(170, 200, 170))
            cx2 = W // 2 + 20
            d1.text((cx2, pros_y), "보완점", font=f_body, fill=(220, 80, 60))
            for ki, con in enumerate(ai_comment.get('cons', [])[:4]):
                d1.text((cx2, pros_y + 28 + ki * 26), f"• {con[:30]}",
                        font=f_xs, fill=(200, 170, 170))

            d1.text((60, H - 36), "Powered by TAD AI  |  AI Football Analysis Platform",
                    font=f_xs, fill=(75, 75, 95))
            pages.append(p1)

            # ── 2페이지: 성장 기록 테이블 ─────────────────────────────────────
            if history:
                p2, d2 = new_page()
                d2.text((60, 28),  "성장 기록",      font=f_h1, fill=(255, 215, 0))
                d2.text((60, 96),  "최근 경기별 스탯 추이", font=f_h2,
                        fill=(180, 180, 200))
                d2.line([(60, 140), (W - 60, 140)], fill=(55, 55, 75), width=1)

                headers = ["날짜", "PAC", "SHO", "PAS", "DRI", "PHY", "거리(km)", "최고속(km/h)"]
                col_ws  = [160, 60, 60, 60, 60, 60, 100, 120]
                cs = [60]
                for cw2 in col_ws[:-1]:
                    cs.append(cs[-1] + cw2)
                cell_h = 38
                ty = 160

                # 헤더
                d2.rectangle([(60, ty), (W - 60, ty + cell_h)], fill=(28, 28, 48))
                for ci, hdr in enumerate(headers):
                    d2.text((cs[ci] + 4, ty + 10), hdr, font=f_xs, fill=(200, 200, 220))

                for ri, hr2 in enumerate(list(reversed(history))[:8]):
                    ry = ty + cell_h * (ri + 1)
                    bg_c = (16, 16, 26) if ri % 2 == 0 else (20, 20, 32)
                    d2.rectangle([(60, ry), (W - 60, ry + cell_h)], fill=bg_c)
                    hs = hr2.get('stats', {})
                    row_v = [
                        str(hr2.get('date', ''))[:10],
                        str(hs.get('PAC', '-')),
                        str(hs.get('SHO', '-')),
                        str(hs.get('PAS', '-')),
                        str(hs.get('DRI', '-')),
                        str(hs.get('PHY', '-')),
                        f"{float(hs.get('total_distance_km', 0)):.1f}",
                        f"{float(hs.get('top_speed_kmh', 0)):.1f}",
                    ]
                    for ci2, vv in enumerate(row_v):
                        d2.text((cs[ci2] + 4, ry + 10), vv,
                                font=f_xs, fill=(195, 195, 215))

                d2.text((60, H - 36),
                        "Powered by TAD AI  |  AI Football Analysis Platform",
                        font=f_xs, fill=(75, 75, 95))
                pages.append(p2)

            filename = f"{session_id}_report.pdf"
            filepath = os.path.join(self.highlight_dir, filename)
            pages[0].save(filepath, 'PDF', save_all=True,
                          append_images=pages[1:], resolution=96)
            return f"/static/highlights/{filename}"

        except Exception as e:
            print(f"[TAD] PDF 리포트 오류: {e}")
            import traceback; traceback.print_exc()
            return None
