from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import cv2
import json
import uuid
import numpy as np
from flask_cors import CORS
from datetime import datetime
from src.detector import FootballDetector

# NumPy 데이터 타입 호환성을 위한 커스텀 JSON 인코더
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float64)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json.cls = NumpyEncoder 
CORS(app)

# --- 저장소 설정 (D: 드라이브 고정) ---
BASE_STORAGE = r"D:\aifootball_data"
UPLOAD_FOLDER = os.path.join(BASE_STORAGE, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_STORAGE, 'processed')
CALIBRATION_FOLDER = os.path.join(BASE_STORAGE, 'calibration')
HIGHLIGHT_FOLDER = os.path.join(BASE_STORAGE, 'highlights')
HISTORY_FILE = os.path.join(BASE_STORAGE, 'analysis_history.json')

# 필요한 모든 디렉토리 강제 생성
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, CALIBRATION_FOLDER, HIGHLIGHT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 인메모리 세션 관리
analysis_sessions = {}
processing_progress = {}

# --- 외부 저장소 파일 서빙 전용 라우트 (중복 금지) ---
@app.route('/data/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/static/highlights/<path:filename>')
def serve_highlights(filename):
    return send_from_directory(HIGHLIGHT_FOLDER, filename)

@app.route('/static/calibration/<path:filename>')
def serve_calibration(filename):
    return send_from_directory(CALIBRATION_FOLDER, filename)

# --- 유틸리티 및 API 라우트 ---
def update_history(new_record):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except: history = []
    history.append(new_record)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    return history

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def get_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try: return jsonify(json.load(f))
            except: return jsonify([])
    return jsonify([])

@app.route('/progress/<session_id>')
def get_progress(session_id):
    return jsonify(processing_progress.get(session_id, {"percent": 0, "status": "waiting"}))

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files: return jsonify({"error": "No video file"}), 400
    video = request.files['video']
    session_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.mp4")
    video.save(video_path)
    
    processing_progress[session_id] = {"percent": 0, "status": "initializing"}
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(os.path.join(CALIBRATION_FOLDER, f"{session_id}_first.jpg"), frame)
    cap.release()
    
    # Selection용 감지 (YOLOv8s 사용)
    detector = FootballDetector('models/yolov8s.pt')
    detections = detector.detect_players_for_selection(video_path)
    
    # 포지션 정보 받아오기 (생략 시 ST 기본값)
    position = request.form.get('position', 'ST')
    
    analysis_sessions[session_id] = {
        "video_path": video_path,
        "position": position
    }
    return jsonify({"session_id": session_id, "frame_url": f"/static/calibration/{session_id}_first.jpg", "players": detections})

@app.route('/analyze', methods=['POST'])
def analyze():
    session_id = request.json.get('session_id')
    calibration_points = request.json.get('points') 
    target_player_id = request.json.get('target_id')
    session = analysis_sessions.get(session_id)
    if not session: return jsonify({"error": "Invalid session"}), 404
    
    try:
        def progress_callback(p):
            processing_progress[session_id] = {"percent": int(p), "status": "analyzing"}

        detector = FootballDetector('models/yolov8s.pt')
        output_video = os.path.join(PROCESSED_FOLDER, f"{session_id}_analyzed.mp4")
        results = detector.process_video_v2(session['video_path'], output_video, calibration_points, target_player_id, progress_callback=progress_callback)
        
        # 포지션 동적 반영
        player_pos = session.get("position", "ST")
        stats = detector.analyzer.calculate_individual_stats(detector.player_tracks, results['target_track_id'], position=player_pos)
        
        # 하이라이트 및 마스터 릴 생성
        highlights, events = detector.analyzer.extract_combined_highlights(session['video_path'], detector.player_tracks, results['target_track_id'])
        master_reel_url = detector.analyzer.generate_master_sizzle_reel(session['video_path'], events, detector.player_tracks, results['target_track_id'], session_id)
        
        heatmap_url = detector.analyzer.generate_pitch_heatmap(detector.player_tracks, results['target_track_id'], session_id)
        ai_comment = detector.analyzer.generate_ai_comment(stats)
        
        update_history({"session_id": session_id, "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "stats": stats, "target_id": results['target_track_id']})
        processing_progress[session_id] = {"percent": 100, "status": "completed"}
        
        # [CRITICAL] 풀영상이 아닌, 오직 마스터 하이라이트 영상만 반환
        if not master_reel_url:
            return jsonify({"error": "No highlight moments detected even after fallback. Try selecting a different player or longer video."}), 400
            
        return jsonify({
            "status": "success", 
            "video_url": master_reel_url, # 오직 하이라이트 영화만 메인으로
            "stats": stats, 
            "highlights": highlights,
            "heatmap_url": heatmap_url, 
            "ai_comment": ai_comment
        })
    except Exception as e:
        processing_progress[session_id] = {"status": "error", "message": str(e)}
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
