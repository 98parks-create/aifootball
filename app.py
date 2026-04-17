from flask import Flask, render_template, request, send_from_directory, jsonify
import os
import cv2
import json
import uuid
import threading
import numpy as np
from flask_cors import CORS
from datetime import datetime
from src.detector import FootballDetector

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float64)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json.cls = NumpyEncoder
CORS(app)

BASE_STORAGE = r"D:\aifootball_data"
UPLOAD_FOLDER = os.path.join(BASE_STORAGE, 'uploads')
PROCESSED_FOLDER = os.path.join(BASE_STORAGE, 'processed')
CALIBRATION_FOLDER = os.path.join(BASE_STORAGE, 'calibration')
HIGHLIGHT_FOLDER = os.path.join(BASE_STORAGE, 'highlights')
HISTORY_FILE = os.path.join(BASE_STORAGE, 'analysis_history.json')

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, CALIBRATION_FOLDER, HIGHLIGHT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

analysis_sessions = {}
processing_progress = {}
analysis_results = {}   # stores completed results per session_id


# --- Static file routes ---
@app.route('/data/processed/<path:filename>')
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/static/highlights/<path:filename>')
def serve_highlights(filename):
    return send_from_directory(HIGHLIGHT_FOLDER, filename)

@app.route('/static/calibration/<path:filename>')
def serve_calibration(filename):
    return send_from_directory(CALIBRATION_FOLDER, filename)


# --- Utilities ---
def update_history(record):
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(record)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    return history


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def get_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify([])

@app.route('/progress/<session_id>')
def get_progress(session_id):
    return jsonify(processing_progress.get(session_id, {"percent": 0, "status": "waiting"}))

@app.route('/results/<session_id>')
def get_results(session_id):
    result = analysis_results.get(session_id)
    if result:
        return jsonify(result)
    prog = processing_progress.get(session_id, {})
    if prog.get('status') == 'error':
        return jsonify({"error": prog.get('message', '분석 중 오류가 발생했습니다.')}), 500
    return jsonify({"status": "processing"}), 202

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "영상 파일이 없습니다."}), 400

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

    detector = FootballDetector('models/yolov8s.pt')
    detections = detector.detect_players_for_selection(video_path)

    position = request.form.get('position', 'ST')
    player_name = request.form.get('player_name', 'PLAYER')

    analysis_sessions[session_id] = {
        "video_path": video_path,
        "position": position,
        "player_name": player_name,
    }
    return jsonify({
        "session_id": session_id,
        "frame_url": f"/static/calibration/{session_id}_first.jpg",
        "players": detections,
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    session_id = data.get('session_id')
    calibration_points = data.get('points')
    target_player_id = data.get('target_id')
    session = analysis_sessions.get(session_id)
    if not session:
        return jsonify({"error": "세션이 존재하지 않습니다."}), 404

    def run_analysis():
        try:
            def progress_cb(p):
                processing_progress[session_id] = {"percent": int(p), "status": "analyzing"}

            detector = FootballDetector('models/yolov8s.pt')
            output_video = os.path.join(PROCESSED_FOLDER, f"{session_id}_analyzed.mp4")
            results = detector.process_video_v2(
                session['video_path'], output_video,
                calibration_points, target_player_id,
                progress_callback=progress_cb
            )

            player_pos = session.get("position", "ST")
            player_name = session.get("player_name", "PLAYER")

            # Get FPS for accurate stats
            cap = cv2.VideoCapture(session['video_path'])
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            cap.release()

            stats = detector.analyzer.calculate_individual_stats(
                detector.player_tracks, results['target_track_id'],
                position=player_pos, fps=fps
            )

            highlights, events = detector.analyzer.extract_combined_highlights(
                session['video_path'], detector.player_tracks,
                results['target_track_id'], fps=fps
            )

            master_reel_url = detector.analyzer.generate_master_sizzle_reel(
                session['video_path'], events, detector.player_tracks,
                results['target_track_id'], session_id, player_name=player_name
            )

            heatmap_url = detector.analyzer.generate_pitch_heatmap(
                detector.player_tracks, results['target_track_id'], session_id
            )
            ai_comment = detector.analyzer.generate_ai_comment(stats)

            if not master_reel_url:
                processing_progress[session_id] = {
                    "status": "error",
                    "message": "하이라이트 장면이 감지되지 않았습니다. 다른 선수를 선택하거나 더 긴 영상을 시도해 주세요."
                }
                return

            update_history({
                "session_id": session_id,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "stats": stats,
                "target_id": results['target_track_id'],
                "player_name": player_name,
            })

            analysis_results[session_id] = {
                "status": "success",
                "video_url": master_reel_url,
                "stats": stats,
                "highlights": highlights,
                "heatmap_url": heatmap_url,
                "ai_comment": ai_comment,
            }
            processing_progress[session_id] = {"percent": 100, "status": "completed"}

        except Exception as e:
            import traceback; traceback.print_exc()
            processing_progress[session_id] = {"status": "error", "message": str(e)}

    processing_progress[session_id] = {"percent": 0, "status": "analyzing"}
    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()
    return jsonify({"status": "started", "session_id": session_id})


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
