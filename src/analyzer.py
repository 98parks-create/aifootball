import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
import cv2

class FootballAnalyzer:
    def __init__(self):
        self.base_storage = r"D:\aifootball_data"
        self.data_dir = os.path.join(self.base_storage, "analysis")
        self.highlight_dir = os.path.join(self.base_storage, "highlights")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.highlight_dir, exist_ok=True)

    def calculate_individual_stats(self, tracks, target_id, position="ST"):
        """포지션 가중치가 적용된 현실적 스탯 계산"""
        if not tracks or target_id not in tracks or not tracks[target_id]:
            return {"PAC": 50, "PHY": 50, "DRI": 50, "PAS": 50, "SHO": 50, "total_distance": 0}
            
        target_data = tracks[target_id]
        target_positions = [t['pos'] for t in target_data]
        distance = 0
        speeds = []
        for i in range(1, len(target_positions)):
            p1, p2 = target_positions[i-1], target_positions[i]
            d = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            distance += d
            speeds.append(d)
        
        top_speed = max(speeds) if speeds else 0
        pos = position.upper()
        pac = min(99, max(60, int(top_speed * 12)))
        phy = min(99, max(55, int(distance * 0.5)))
        
        if "ST" in pos or "FW" in pos:
            sho = min(99, max(75, 78 + (top_speed/2)))
            dri = min(99, max(75, 75 + (pac/10)))
            pas = 70
        elif "MF" in pos:
            sho = 72
            dri = min(99, max(80, 82 + (distance/200)))
            pas = min(99, max(82, 84 + (distance/150)))
        else: # DF
            sho = 62
            dri = 68
            phy = min(99, phy + 12)
            pas = 74

        return {"PAC": pac, "PHY": phy, "DRI": dri, "PAS": pas, "SHO": sho, "total_distance": round(distance, 2)}

    def generate_pitch_heatmap(self, tracks, target_id, session_id):
        """타겟 플레이어의 히트맵 생성 (TAD AI)"""
        if not tracks or target_id not in tracks: return None
        try:
            positions = np.array([t['pos'] for t in tracks[target_id]])
            heatmap, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=[50, 25], range=[[0, 100], [0, 50]])
            fig = Figure(figsize=(10, 5))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            # 투명 배경 히트맵 (mix-blend-mode 호환성 극대화)
            ax.imshow(heatmap.T, origin='lower', extent=[0, 100, 0, 50], cmap='YlOrRd', interpolation='gaussian')
            ax.axis('off')
            filename = f"{session_id}_heatmap.png"
            heatmap_path = os.path.join(self.highlight_dir, filename)
            fig.savefig(heatmap_path, bbox_inches='tight', transparent=True, dpi=150)
            return f"/static/highlights/{filename}"
        except Exception as e:
            print(f"Heatmap Error: {e}"); return None

    def generate_ai_comment(self, stats):
        """전문 스카우팅 리포트 (TAD AI)"""
        report = {"pros": [], "cons": [], "scouting_note": ""}
        if stats['PAC'] > 85: report['pros'].append("폭발적인 순간 가속력과 스프린트")
        if stats['DRI'] > 80: report['pros'].append("안정적인 볼 키핑 및 드리블 전진 능력")
        if stats['PHY'] > 80: report['pros'].append("경기 전체를 소화하는 왕성한 활동량")
        if stats['PAC'] < 72: report['cons'].append("역습 시 순간적인 속도 경쟁력 보완 필요")
        if stats['SHO'] < 75: report['cons'].append("결정적인 상황에서의 과감한 슈팅 시도 부족")
        
        if len(report['pros']) >= 1: 
            report['scouting_note'] = "TAD AI 분석 결과, 팀의 공격 전개에 있어 대체 불가능한 에이스로서의 잠재력이 확인되었습니다."
        else: 
            report['scouting_note'] = "TAD AI는 당신의 기본기와 안정적인 운영을 높게 평가합니다. 다만 더 높은 무대를 위해 자신만의 확실한 무기가 필요합니다."
        return report

    def extract_combined_highlights(self, video_path, tracks, target_id):
        """하이라이트 추출 (3초 전후, 선명한 화질 보장)"""
        if target_id not in tracks: return [], []
        
        target_tracks = tracks[target_id]
        target_dict = {t['frame']: t['pos_px'] for t in target_tracks if 'pos_px' in t}
        ball_tracks = [t for tid, data in tracks.items() if data and data[0].get('class') == 32 for t in data]
        ball_dict = {b['frame']: b['pos'] for b in ball_tracks}
        
        events = []
        last_event_frame = -175
        frames_list = sorted(list(target_dict.keys()))
        
        # 볼 터치 기반 하이라이트 (전후 3초 보장)
        for i, f in enumerate(frames_list):
            if f - last_event_frame < 150: continue # 6초 간격 (25fps 기준)
            if f in ball_dict:
                p_pos = target_tracks[i].get('pos', [0,0])
                dist = np.sqrt((p_pos[0]-ball_dict[f][0])**2 + (p_pos[1]-ball_dict[f][1])**2)
                if dist < 2.5: 
                    events.append({"frame": f, "category": "Touch Focus", "score": 100})
                    last_event_frame = f

        # 활동량 기반 하이라이트 (보완)
        if len(events) < 5:
            window_size = 150 
            for start_idx in range(0, len(frames_list) - window_size, window_size):
                f = frames_list[start_idx + window_size//2]
                if any(abs(f - e['frame']) < 180 for e in events): continue
                seg_dist = sum([np.sqrt((target_tracks[j]['pos'][0]-target_tracks[j-1]['pos'][0])**2 + 
                                       (target_tracks[j]['pos'][1]-target_tracks[j-1]['pos'][1])**2) 
                               for j in range(start_idx+1, start_idx+window_size)])
                events.append({"frame": f, "category": "Active Play", "score": seg_dist})
        
        events = sorted(events, key=lambda x: x['score'], reverse=True)[:10]
        events = sorted(events, key=lambda x: x['frame'])
        
        highlights = []
        for i, ev in enumerate(events):
            start_f = max(0, int(ev['frame'] - 75)) # 3초 전
            end_f = int(ev['frame'] + 75)   # 3초 후
            clip_name = f"tad_hl_{target_id}_{i}.mp4"
            clip_path = os.path.join(self.highlight_dir, clip_name)
            
            segment_tracks = {f: target_dict[f] for f in range(start_f, end_f) if f in target_dict}
            if self._save_cinematic_clip(video_path, start_f, end_f, clip_path, segment_tracks):
                highlights.append({"url": f"/static/highlights/{clip_name}", "category": ev['category'], "frame": int(ev['frame'])})
        
        return highlights, events

    def _save_cinematic_clip(self, video_path, start_f, end_f, out_path, tracks):
        """TAD AI 전용 선명한 액션 리얼 (줌 최적화)"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            out_w, out_h = 1280, 720
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (out_w, out_h))
            if not out.isOpened(): out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            lx, ly = None, None
            for f_idx in range(start_f, end_f):
                ret, frame = cap.read()
                if not ret: break
                h, w = frame.shape[:2]
                cx, cy = tracks.get(f_idx, (w/2, h/2))
                
                if lx is not None: cx, cy = lx*0.8 + cx*0.2, ly*0.8 + cy*0.2
                lx, ly = cx, cy
                
                # [개선] 줌 범위를 1000px로 넓혀 해상도 손실 최소화 (화질 보호)
                cs = 1000
                x1, y1 = int(max(0, min(w-cs, cx-cs/2))), int(max(0, min(h-cs, cy-cs/2)))
                crop = frame[y1:y1+cs, x1:x1+cs]
                res = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                
                # TAD AI 태그
                cv2.putText(res, "TAD AI | ACTION FOCUS", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 215, 0), 3)
                out.write(res)
            cap.release(); out.release()
            return os.path.exists(out_path) and os.path.getsize(out_path) > 100
        except Exception as e:
            print(f"Clip Error: {e}"); return False

    def generate_master_sizzle_reel(self, video_path, events, tracks, target_id, session_id):
        """TAD AI 통합 하이라이트 영화 제작"""
        if not events or target_id not in tracks: return None
        target_dict = {t['frame']: t['pos_px'] for t in tracks[target_id] if 'pos_px' in t}
        output_name = f"{session_id}_tad_master.mp4"
        output_path = os.path.join(self.highlight_dir, output_name)
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            out_w, out_h = 1280, 720
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (out_w, out_h))
            if not out.isOpened(): out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
            
            lx, ly = None, None
            for ev in events:
                start_f, end_f = max(0, int(ev['frame'] - 75)), int(ev['frame'] + 75)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                for f_idx in range(start_f, end_f):
                    ret, frame = cap.read()
                    if not ret: break
                    h, w = frame.shape[:2]
                    cx, cy = target_dict.get(f_idx, (w/2, h/2))
                    if lx is not None: cx, cy = lx*0.8 + cx*0.2, ly*0.8 + cy*0.2
                    lx, ly = cx, cy
                    cs = 1000
                    x1, y1 = int(max(0, min(w-cs, cx-cs/2))), int(max(0, min(h-cs, cy-cs/2)))
                    crop = frame[y1:y1+cs, x1:x1+cs]
                    res = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                    cv2.putText(res, "TAD AI | OFFICIAL FILM", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3)
                    out.write(res)
                # Cut transition
                for _ in range(5): out.write(np.zeros((out_h, out_w, 3), dtype=np.uint8))
            
            cap.release(); out.release()
            return f"/static/highlights/{output_name}" if os.path.exists(output_path) else None
        except Exception as e:
            print(f"Master Reel Error: {e}"); return None
