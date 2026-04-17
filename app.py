from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, send_from_directory, jsonify, session
import os, cv2, json, uuid, threading, sqlite3, smtplib
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from datetime import datetime
from src.detector import FootballDetector

# ── App setup ──────────────────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):  return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray):              return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json.cls = NumpyEncoder
app.secret_key = os.environ.get('SECRET_KEY', 'tad-ai-secret-2026-football-platform')
CORS(app)

# ── Storage paths ───────────────────────────────────────────────────────────────
BASE        = r"D:\aifootball_data"
UPLOAD_DIR  = os.path.join(BASE, 'uploads')
PROC_DIR    = os.path.join(BASE, 'processed')
CALIB_DIR   = os.path.join(BASE, 'calibration')
HL_DIR      = os.path.join(BASE, 'highlights')
DB_PATH     = os.path.join(BASE, 'tad.db')

for d in [UPLOAD_DIR, PROC_DIR, CALIB_DIR, HL_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Email config (환경변수로 설정) ────────────────────────────────────────────
MAIL_USER = os.environ.get('MAIL_USER', '')   # Gmail 주소
MAIL_PASS = os.environ.get('MAIL_PASS', '')   # Gmail 앱 비밀번호
MAIL_FROM = os.environ.get('MAIL_FROM', MAIL_USER)
BASE_URL  = os.environ.get('BASE_URL', 'http://localhost:5000')

# ── SQLite init ──────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT    UNIQUE NOT NULL,
            email        TEXT    UNIQUE NOT NULL,
            password_hash TEXT   NOT NULL,
            created_at   TEXT    DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS analysis_history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            session_id   TEXT,
            date         TEXT,
            player_name  TEXT,
            position     TEXT,
            stats_json   TEXT,
            target_track_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS analysis_jobs (
            session_id   TEXT    PRIMARY KEY,
            user_id      INTEGER,
            player_name  TEXT,
            status       TEXT    DEFAULT 'pending',
            result_json  TEXT,
            created_at   TEXT,
            completed_at TEXT
        );
        """)

init_db()

# ── In-memory session stores ─────────────────────────────────────────────────────
analysis_sessions  = {}
processing_progress = {}

# ── Email helper ─────────────────────────────────────────────────────────────────
def send_result_email(to_email: str, player_name: str, session_id: str):
    if not MAIL_USER or not MAIL_PASS:
        print("[TAD] 이메일 미설정 — 발송 건너뜀")
        return
    result_url = f"{BASE_URL}/?sid={session_id}"
    html = f"""
    <div style="font-family:sans-serif;max-width:560px;margin:0 auto;background:#0a0a0a;color:#fff;padding:40px;border-radius:16px">
      <h1 style="color:#ff1a1a;font-size:2rem;margin-bottom:4px">TAD AI</h1>
      <p style="opacity:.5;margin-bottom:32px;font-size:.9rem">AI 축구 분석 플랫폼</p>
      <h2 style="font-size:1.4rem;margin-bottom:12px">⚽ 분석이 완료됐습니다!</h2>
      <p style="opacity:.8;line-height:1.7;margin-bottom:28px">
        <strong style="color:#ffd700">{player_name}</strong> 선수의 하이라이트 영상과<br>
        퍼포먼스 스탯이 준비됐어요. 아래 버튼을 눌러 확인하세요.
      </p>
      <a href="{result_url}"
         style="display:inline-block;background:linear-gradient(45deg,#ff1a1a,#dc143c);
                color:#fff;text-decoration:none;padding:16px 36px;border-radius:100px;
                font-weight:800;font-size:1rem">
        결과 확인하기 →
      </a>
      <p style="opacity:.35;font-size:.78rem;margin-top:32px">
        링크가 열리지 않으면 복사하세요: {result_url}
      </p>
    </div>
    """
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[TAD AI] {player_name} 하이라이트 분석 완료!"
        msg['From']    = MAIL_FROM
        msg['To']      = to_email
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10) as s:
            s.login(MAIL_USER, MAIL_PASS)
            s.send_message(msg)
        print(f"[TAD] 이메일 발송 완료 → {to_email}")
    except Exception as e:
        print(f"[TAD] 이메일 발송 실패: {e}")

# ── Static file routes ───────────────────────────────────────────────────────────
@app.route('/data/processed/<path:filename>')
def serve_processed(filename): return send_from_directory(PROC_DIR, filename)

@app.route('/static/highlights/<path:filename>')
def serve_highlights(filename): return send_from_directory(HL_DIR, filename)

@app.route('/static/calibration/<path:filename>')
def serve_calibration(filename): return send_from_directory(CALIB_DIR, filename)

# ── Auth helpers ─────────────────────────────────────────────────────────────────
def current_user():
    uid = session.get('user_id')
    if not uid: return None
    with get_db() as db:
        row = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None

# ── Auth routes ──────────────────────────────────────────────────────────────────
@app.route('/api/me')
def api_me():
    u = current_user()
    if not u: return jsonify({"logged_in": False}), 200
    return jsonify({"logged_in": True, "id": u['id'], "username": u['username'], "email": u['email']})

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json or {}
    username = (data.get('username') or '').strip()
    email    = (data.get('email')    or '').strip()
    password = (data.get('password') or '').strip()
    if not username or not email or not password:
        return jsonify({"error": "모든 필드를 입력해 주세요."}), 400
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?,?,?)",
                (username, email, generate_password_hash(password))
            )
        with get_db() as db:
            u = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        session['user_id'] = u['id']
        return jsonify({"ok": True, "username": username})
    except sqlite3.IntegrityError:
        return jsonify({"error": "이미 사용 중인 이메일 또는 닉네임입니다."}), 409

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json or {}
    email    = (data.get('email')    or '').strip()
    password = (data.get('password') or '').strip()
    with get_db() as db:
        u = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not u or not check_password_hash(u['password_hash'], password):
        return jsonify({"error": "이메일 또는 비밀번호가 틀렸습니다."}), 401
    session['user_id'] = u['id']
    return jsonify({"ok": True, "username": u['username']})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({"ok": True})

# ── Main routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/history')
def get_history():
    u = current_user()
    if not u: return jsonify([])
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM analysis_history WHERE user_id=? ORDER BY id DESC LIMIT 50",
            (u['id'],)
        ).fetchall()
    result = []
    for r in rows:
        try:
            stats = json.loads(r['stats_json'])
        except Exception:
            stats = {}
        result.append({"date": r['date'], "player_name": r['player_name'],
                        "stats": stats, "target_id": r['target_track_id'],
                        "session_id": r['session_id']})
    return jsonify(result)

@app.route('/progress/<session_id>')
def get_progress(session_id):
    return jsonify(processing_progress.get(session_id, {"percent": 0, "status": "waiting"}))

@app.route('/results/<session_id>')
def get_results(session_id):
    # DB에서 먼저 확인 (서버 재시작 후에도 복원 가능)
    with get_db() as db:
        row = db.execute("SELECT * FROM analysis_jobs WHERE session_id=?", (session_id,)).fetchone()
    if row and row['status'] == 'completed' and row['result_json']:
        return jsonify(json.loads(row['result_json']))
    if row and row['status'] == 'error':
        return jsonify({"error": "분석 오류가 발생했습니다."}), 500

    prog = processing_progress.get(session_id, {})
    if prog.get('status') == 'error':
        return jsonify({"error": prog.get('message', '분석 오류')}), 500
    return jsonify({"status": "processing"}), 202

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "영상 파일이 없습니다."}), 400
    video = request.files['video']
    sid   = str(uuid.uuid4())
    vpath = os.path.join(UPLOAD_DIR, f"{sid}.mp4")
    video.save(vpath)

    processing_progress[sid] = {"percent": 0, "status": "initializing", "stage_msg": "영상 업로드 완료. 선수 감지 중..."}

    cap = cv2.VideoCapture(vpath)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_min = total_f / vid_fps / 60
    if duration_min > 25:
        cap.release()
        os.remove(vpath)
        return jsonify({"error": f"영상 길이가 {duration_min:.0f}분입니다. 최대 25분까지 지원합니다. 구간을 잘라서 다시 올려주세요."}), 400
    ok, frame = cap.read()
    if ok:
        cv2.imwrite(os.path.join(CALIB_DIR, f"{sid}_first.jpg"), frame)
    cap.release()

    det  = FootballDetector('models/yolov8s.pt', scan_model_path='models/yolov8n.pt')
    dets = det.detect_players_for_selection(vpath)

    analysis_sessions[sid] = {
        "video_path":  vpath,
        "position":    request.form.get('position', 'ST'),
        "player_name": request.form.get('player_name', 'PLAYER'),
        "user_id":     session.get('user_id'),
    }
    return jsonify({
        "session_id": sid,
        "frame_url":  f"/static/calibration/{sid}_first.jpg",
        "players":    dets,
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    data   = request.json
    sid    = data.get('session_id')
    pts    = data.get('points')
    target = data.get('target_id')
    sess   = analysis_sessions.get(sid)
    if not sess: return jsonify({"error": "세션이 없습니다."}), 404

    uid         = sess.get('user_id')
    player_name = sess.get('player_name', 'PLAYER')

    # 분석 작업 DB 등록
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO analysis_jobs (session_id,user_id,player_name,status,created_at) VALUES (?,?,?,?,?)",
            (sid, uid, player_name, 'pending', datetime.now().strftime("%Y-%m-%d %H:%M"))
        )

    def run():
        try:
            def cb_scan(p):
                processing_progress[sid] = {
                    "percent":   int(p * 0.65),
                    "status":    "analyzing",
                    "stage_msg": f"1단계 — 480p 빠른 스캔 중 ({int(p)}%)",
                }

            det = FootballDetector('models/yolov8s.pt', scan_model_path='models/yolov8n.pt')
            res = det.process_video_v2(
                sess['video_path'], None, pts, target, progress_callback=cb_scan)

            fps = res.get('fps', 25)

            processing_progress[sid] = {
                "percent": 65, "status": "analyzing",
                "stage_msg": "2단계 — 이벤트 감지 중...",
            }
            stats = det.analyzer.calculate_individual_stats(
                det.player_tracks, res['target_track_id'],
                position=sess.get('position', 'ST'), fps=fps)

            processing_progress[sid] = {
                "percent": 68, "status": "analyzing",
                "stage_msg": "2단계 — 하이라이트 클립 추출 중...",
            }
            highlights, events = det.analyzer.extract_combined_highlights(
                sess['video_path'], det.player_tracks,
                res['target_track_id'], fps=fps,
                position=sess.get('position', 'ST'),
                target_team=res.get('target_team', -1),
                player_name=player_name)

            processing_progress[sid] = {
                "percent": 82, "status": "analyzing",
                "stage_msg": "2단계 — 마스터 하이라이트 영상 생성 중...",
            }
            master_url = det.analyzer.generate_master_sizzle_reel(
                sess['video_path'], events, det.player_tracks,
                res['target_track_id'], sid, player_name=player_name)

            processing_progress[sid] = {
                "percent": 95, "status": "analyzing",
                "stage_msg": "2단계 — 히트맵 & 리포트 생성 중...",
            }
            heatmap_url = det.analyzer.generate_pitch_heatmap(
                det.player_tracks, res['target_track_id'], sid)
            ai_comment  = det.analyzer.generate_ai_comment(stats)

            if not master_url:
                with get_db() as db:
                    db.execute("UPDATE analysis_jobs SET status=? WHERE session_id=?", ('error', sid))
                processing_progress[sid] = {
                    "status": "error",
                    "message": "하이라이트 감지 실패. 더 긴 영상이나 다른 선수를 선택해 주세요."
                }
                return

            result = {
                "status":      "success",
                "video_url":   master_url,
                "stats":       stats,
                "highlights":  highlights,
                "heatmap_url": heatmap_url,
                "ai_comment":  ai_comment,
            }

            # analysis_history 저장
            if uid:
                with get_db() as db:
                    db.execute(
                        "INSERT INTO analysis_history (user_id,session_id,date,player_name,position,stats_json,target_track_id) VALUES (?,?,?,?,?,?,?)",
                        (uid, sid, datetime.now().strftime("%Y-%m-%d %H:%M"),
                         player_name, sess.get('position', 'ST'),
                         json.dumps(stats, cls=NumpyEncoder),
                         res['target_track_id'])
                    )

            # 결과 JSON DB 저장 (서버 재시작 후에도 접근 가능)
            with get_db() as db:
                db.execute(
                    "UPDATE analysis_jobs SET status=?,result_json=?,completed_at=? WHERE session_id=?",
                    ('completed', json.dumps(result, cls=NumpyEncoder),
                     datetime.now().strftime("%Y-%m-%d %H:%M"), sid)
                )

            processing_progress[sid] = {"percent": 100, "status": "completed"}

            # 원본 영상 삭제 (디스크 절약 — 클립/마스터릴은 이미 별도 저장됨)
            try:
                if os.path.exists(sess['video_path']):
                    os.remove(sess['video_path'])
                    print(f"[TAD] 원본 영상 삭제: {sess['video_path']}")
            except Exception as e:
                print(f"[TAD] 원본 영상 삭제 실패: {e}")

            # 이메일 발송
            if uid:
                with get_db() as db:
                    user_row = db.execute("SELECT email FROM users WHERE id=?", (uid,)).fetchone()
                if user_row:
                    send_result_email(user_row['email'], player_name, sid)

        except Exception as e:
            import traceback; traceback.print_exc()
            with get_db() as db:
                db.execute("UPDATE analysis_jobs SET status=? WHERE session_id=?", ('error', sid))
            processing_progress[sid] = {"status": "error", "message": str(e)}

    processing_progress[sid] = {"percent": 0, "status": "analyzing", "stage_msg": "AI 모델 로딩 중..."}
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started", "session_id": sid})


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
