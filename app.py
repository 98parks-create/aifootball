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
BASE        = os.path.join(os.path.expanduser("~"), "aifootball_data")
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
        -- ── Stage 2: 팀 피드 ──────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS teams (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            invite_code TEXT    UNIQUE NOT NULL,
            creator_id  INTEGER,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS team_members (
            team_id   INTEGER NOT NULL,
            user_id   INTEGER NOT NULL,
            joined_at TEXT    DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS feed_posts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id       INTEGER NOT NULL,
            user_id       INTEGER NOT NULL,
            player_name   TEXT,
            session_id    TEXT,
            highlight_url TEXT,
            message       TEXT,
            created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
        );
        -- ── Stage 2: 리그 순위표 ──────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS leagues (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            invite_code TEXT    UNIQUE NOT NULL,
            creator_id  INTEGER,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS league_members (
            league_id   INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            player_name TEXT,
            position    TEXT,
            PRIMARY KEY (league_id, user_id)
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

# ── 자동 캘리브레이션 ─────────────────────────────────────────────────────────────

def _auto_detect_field_corners(frame):
    """
    첫 프레임에서 녹색 잔디 영역을 감지하고 경기장 4모서리를 반환.
    반환 순서: [좌상, 우상, 우하, 좌하]
    """
    h, w = frame.shape[:2]

    # HSV 녹색 마스크
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([28, 35, 25]), np.array([92, 255, 255]))

    # 잡음 제거
    k    = np.ones((18, 18), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < h * w * 0.07:
        return None

    # 볼록 껍질 → 4각형 근사
    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    pts  = None
    for eps in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(-1, 2).astype(float)
            break

    if pts is None:
        # 최소 외접 사각형으로 fallback
        rect = cv2.minAreaRect(largest)
        pts  = cv2.boxPoints(rect).astype(float)

    # 좌상→우상→우하→좌하 정렬
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl   = pts[np.argmin(s)]
    br   = pts[np.argmax(s)]
    tr   = pts[np.argmin(diff)]
    bl   = pts[np.argmax(diff)]

    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]


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

@app.route('/api/auto_calibrate/<session_id>')
def auto_calibrate(session_id):
    frame_path = os.path.join(CALIB_DIR, f"{session_id}_first.jpg")
    if not os.path.exists(frame_path):
        return jsonify({"error": "캘리브레이션 프레임이 없습니다."}), 404
    frame = cv2.imread(frame_path)
    if frame is None:
        return jsonify({"error": "이미지 로드 실패"}), 500
    corners = _auto_detect_field_corners(frame)
    if not corners:
        return jsonify({"error": "경기장 자동 감지 실패. 수동으로 클릭해 주세요."}), 422
    return jsonify({"corners": corners, "width": int(frame.shape[1]), "height": int(frame.shape[0])})

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
    try:
        video.save(vpath)

        processing_progress[sid] = {"percent": 0, "status": "initializing", "stage_msg": "영상 업로드 완료. 선수 감지 중..."}

        cap = cv2.VideoCapture(vpath)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        duration_min = total_f / vid_fps / 60
        if duration_min > 25:
            cap.release()
            os.remove(vpath)
            return jsonify({"error": f"영상 길이가 {duration_min:.0f}분입니다. 최대 25분까지 지원합니다."}), 400
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
    except Exception as e:
        import traceback; traceback.print_exc()
        if os.path.exists(vpath):
            try: os.remove(vpath)
            except: pass
        return jsonify({"error": f"업로드 처리 중 오류: {str(e)}"}), 500

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
            highlights, events, pass_stats = det.analyzer.extract_combined_highlights(
                sess['video_path'], det.player_tracks,
                res['target_track_id'], fps=fps,
                position=sess.get('position', 'ST'),
                target_team=res.get('target_team', -1),
                player_name=player_name)

            # 패스 성공률 스탯에 병합
            stats.update({k: v for k, v in pass_stats.items() if v is not None})

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


# ── Stage 2: FIFA 카드 다운로드 ─────────────────────────────────────────────────
@app.route('/card/<session_id>')
def get_card(session_id):
    """분석 결과에서 FIFA 스타일 카드 PNG를 생성해 반환."""
    with get_db() as db:
        row = db.execute(
            "SELECT result_json, player_name FROM analysis_jobs WHERE session_id=?",
            (session_id,)).fetchone()
    if not row or not row['result_json']:
        return jsonify({"error": "분석 데이터 없음"}), 404
    try:
        result = json.loads(row['result_json'])
        stats  = result.get('stats', {})
        player_name = row['player_name'] or 'PLAYER'
        position = 'ST'
        # history에서 포지션 가져오기
        with get_db() as db:
            h = db.execute(
                "SELECT position FROM analysis_history WHERE session_id=?",
                (session_id,)).fetchone()
        if h:
            position = h['position'] or 'ST'
        from src.analyzer import FootballAnalyzer
        ana = FootballAnalyzer()
        card_url = ana.generate_player_card(stats, player_name, position, session_id)
        if not card_url:
            return jsonify({"error": "카드 생성 실패"}), 500
        return jsonify({"card_url": card_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Stage 3: PDF 성장 리포트 다운로드 ────────────────────────────────────────────
@app.route('/report/<session_id>')
def get_report(session_id):
    """PDF 성장 리포트 생성 후 다운로드."""
    with get_db() as db:
        job = db.execute(
            "SELECT result_json, player_name, user_id FROM analysis_jobs WHERE session_id=?",
            (session_id,)).fetchone()
    if not job or not job['result_json']:
        return jsonify({"error": "분석 데이터 없음"}), 404
    try:
        result      = json.loads(job['result_json'])
        stats       = result.get('stats', {})
        ai_comment  = result.get('ai_comment', {})
        player_name = job['player_name'] or 'PLAYER'
        position    = 'ST'
        history     = []
        uid = job['user_id']
        with get_db() as db:
            h = db.execute(
                "SELECT position FROM analysis_history WHERE session_id=?",
                (session_id,)).fetchone()
            if h:
                position = h['position'] or 'ST'
            if uid:
                rows = db.execute(
                    "SELECT date, stats_json FROM analysis_history "
                    "WHERE user_id=? ORDER BY id DESC LIMIT 8", (uid,)).fetchall()
                for r in rows:
                    try:
                        history.append({"date": r['date'],
                                        "stats": json.loads(r['stats_json'])})
                    except Exception:
                        pass
        from src.analyzer import FootballAnalyzer
        ana = FootballAnalyzer()
        pdf_url = ana.generate_pdf_report(
            stats, ai_comment, player_name, position, history, session_id)
        if not pdf_url:
            return jsonify({"error": "PDF 생성 실패"}), 500
        filename = f"{session_id}_report.pdf"
        pdf_path = os.path.join(HL_DIR, filename)
        return send_from_directory(HL_DIR, filename, as_attachment=True,
                                   download_name=f"TAD_리포트_{player_name}.pdf")
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Stage 3: 코치에게 리포트 이메일 발송 ─────────────────────────────────────────
@app.route('/api/report/share', methods=['POST'])
def share_report():
    """PDF 리포트를 코치 이메일로 발송."""
    u = current_user()
    if not u:
        return jsonify({"error": "로그인 필요"}), 401
    data       = request.json or {}
    session_id = data.get('session_id', '').strip()
    coach_email = data.get('coach_email', '').strip()
    message    = data.get('message', '').strip()
    if not session_id or not coach_email:
        return jsonify({"error": "session_id, coach_email 필수"}), 400

    # PDF 파일 경로
    pdf_filename = f"{session_id}_report.pdf"
    pdf_path = os.path.join(HL_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        # 없으면 먼저 생성
        with get_db() as db:
            job = db.execute(
                "SELECT result_json, player_name FROM analysis_jobs WHERE session_id=?",
                (session_id,)).fetchone()
        if not job or not job['result_json']:
            return jsonify({"error": "분석 없음"}), 404
        result     = json.loads(job['result_json'])
        player_name = job['player_name'] or 'PLAYER'
        history = []
        with get_db() as db:
            h = db.execute(
                "SELECT position FROM analysis_history WHERE session_id=?",
                (session_id,)).fetchone()
            position = h['position'] if h else 'ST'
            rows = db.execute(
                "SELECT date, stats_json FROM analysis_history "
                "WHERE user_id=? ORDER BY id DESC LIMIT 8", (u['id'],)).fetchall()
            for r in rows:
                try:
                    history.append({"date": r['date'],
                                    "stats": json.loads(r['stats_json'])})
                except Exception:
                    pass
        from src.analyzer import FootballAnalyzer
        ana = FootballAnalyzer()
        ana.generate_pdf_report(
            result.get('stats', {}), result.get('ai_comment', {}),
            player_name, position, history, session_id)
    else:
        with get_db() as db:
            job = db.execute(
                "SELECT player_name FROM analysis_jobs WHERE session_id=?",
                (session_id,)).fetchone()
        player_name = job['player_name'] if job else 'PLAYER'

    if not os.path.exists(pdf_path):
        return jsonify({"error": "PDF 생성 실패"}), 500

    if not MAIL_USER or not MAIL_PASS:
        return jsonify({"error": "이메일 서버 미설정"}), 503
    try:
        import smtplib
        from email.mime.base import MIMEBase
        from email import encoders
        msg = MIMEMultipart()
        msg['Subject'] = f"[TAD AI] {player_name} 선수 성장 리포트"
        msg['From']    = MAIL_FROM
        msg['To']      = coach_email
        body = message or f"{player_name} 선수의 TAD AI 성장 리포트를 공유합니다."
        msg.attach(MIMEText(body, 'plain'))
        with open(pdf_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        f'attachment; filename="TAD_리포트_{player_name}.pdf"')
        msg.attach(part)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15) as s:
            s.login(MAIL_USER, MAIL_PASS)
            s.send_message(msg)
        return jsonify({"ok": True, "to": coach_email})
    except Exception as e:
        return jsonify({"error": f"발송 실패: {e}"}), 500


# ── Stage 2: 팀 피드 API ─────────────────────────────────────────────────────────

def _gen_invite_code():
    import random, string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

@app.route('/api/teams/create', methods=['POST'])
def team_create():
    u = current_user()
    if not u:
        return jsonify({"error": "로그인 필요"}), 401
    name = (request.json or {}).get('name', '').strip()
    if not name:
        return jsonify({"error": "팀 이름 필요"}), 400
    code = _gen_invite_code()
    with get_db() as db:
        db.execute("INSERT INTO teams (name, invite_code, creator_id) VALUES (?,?,?)",
                   (name, code, u['id']))
        t = db.execute("SELECT id FROM teams WHERE invite_code=?", (code,)).fetchone()
        db.execute("INSERT OR IGNORE INTO team_members (team_id, user_id) VALUES (?,?)",
                   (t['id'], u['id']))
    return jsonify({"ok": True, "invite_code": code, "team_id": t['id']})

@app.route('/api/teams/join/<code>', methods=['POST'])
def team_join(code):
    u = current_user()
    if not u:
        return jsonify({"error": "로그인 필요"}), 401
    with get_db() as db:
        t = db.execute("SELECT * FROM teams WHERE invite_code=?", (code,)).fetchone()
        if not t:
            return jsonify({"error": "초대 코드가 없습니다"}), 404
        db.execute("INSERT OR IGNORE INTO team_members (team_id, user_id) VALUES (?,?)",
                   (t['id'], u['id']))
    return jsonify({"ok": True, "team_id": t['id'], "team_name": t['name']})

@app.route('/api/teams/my')
def team_my():
    u = current_user()
    if not u:
        return jsonify([])
    with get_db() as db:
        rows = db.execute(
            "SELECT t.id, t.name, t.invite_code "
            "FROM teams t JOIN team_members m ON t.id=m.team_id "
            "WHERE m.user_id=? ORDER BY t.id DESC", (u['id'],)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/teams/<int:team_id>/feed')
def team_feed(team_id):
    u = current_user()
    if not u:
        return jsonify({"error": "로그인 필요"}), 401
    # 멤버 확인
    with get_db() as db:
        m = db.execute("SELECT 1 FROM team_members WHERE team_id=? AND user_id=?",
                       (team_id, u['id'])).fetchone()
        if not m:
            return jsonify({"error": "팀 멤버만 조회 가능"}), 403
        posts = db.execute(
            "SELECT p.*, u.username FROM feed_posts p "
            "JOIN users u ON p.user_id=u.id "
            "WHERE p.team_id=? ORDER BY p.id DESC LIMIT 30",
            (team_id,)).fetchall()
    return jsonify([dict(r) for r in posts])

@app.route('/api/teams/<int:team_id>/post', methods=['POST'])
def team_post(team_id):
    u = current_user()
    if not u:
        return jsonify({"error": "로그인 필요"}), 401
    data = request.json or {}
    session_id    = data.get('session_id', '')
    message       = data.get('message', '').strip()
    highlight_url = data.get('highlight_url', '')
    player_name   = data.get('player_name', u['username'])
    with get_db() as db:
        m = db.execute("SELECT 1 FROM team_members WHERE team_id=? AND user_id=?",
                       (team_id, u['id'])).fetchone()
        if not m:
            return jsonify({"error": "팀 멤버만 게시 가능"}), 403
        db.execute(
            "INSERT INTO feed_posts (team_id,user_id,player_name,session_id,"
            "highlight_url,message,created_at) VALUES (?,?,?,?,?,?,?)",
            (team_id, u['id'], player_name, session_id,
             highlight_url, message, datetime.now().strftime("%Y-%m-%d %H:%M")))
    return jsonify({"ok": True})


# ── Stage 2: 리그 순위표 API ─────────────────────────────────────────────────────

@app.route('/api/leagues/create', methods=['POST'])
def league_create():
    u = current_user()
    if not u:
        return jsonify({"error": "로그인 필요"}), 401
    data = request.json or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({"error": "리그 이름 필요"}), 400
    code = _gen_invite_code()
    player_name = data.get('player_name', u['username'])
    position    = data.get('position', 'MF')
    with get_db() as db:
        db.execute("INSERT INTO leagues (name, invite_code, creator_id) VALUES (?,?,?)",
                   (name, code, u['id']))
        lg = db.execute("SELECT id FROM leagues WHERE invite_code=?", (code,)).fetchone()
        db.execute(
            "INSERT OR IGNORE INTO league_members (league_id,user_id,player_name,position) "
            "VALUES (?,?,?,?)", (lg['id'], u['id'], player_name, position))
    return jsonify({"ok": True, "invite_code": code, "league_id": lg['id']})

@app.route('/api/leagues/join/<code>', methods=['POST'])
def league_join(code):
    u = current_user()
    if not u:
        return jsonify({"error": "로그인 필요"}), 401
    data = request.json or {}
    player_name = data.get('player_name', u['username'])
    position    = data.get('position', 'MF')
    with get_db() as db:
        lg = db.execute("SELECT * FROM leagues WHERE invite_code=?", (code,)).fetchone()
        if not lg:
            return jsonify({"error": "초대 코드가 없습니다"}), 404
        db.execute(
            "INSERT OR IGNORE INTO league_members (league_id,user_id,player_name,position) "
            "VALUES (?,?,?,?)", (lg['id'], u['id'], player_name, position))
    return jsonify({"ok": True, "league_id": lg['id'], "league_name": lg['name']})

@app.route('/api/leagues/my')
def league_my():
    u = current_user()
    if not u:
        return jsonify([])
    with get_db() as db:
        rows = db.execute(
            "SELECT l.id, l.name, l.invite_code "
            "FROM leagues l JOIN league_members m ON l.id=m.league_id "
            "WHERE m.user_id=? ORDER BY l.id DESC", (u['id'],)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/leagues/<int:league_id>/standings')
def league_standings(league_id):
    """
    리그 멤버별 최근 stats 합산 → 종합 점수 순위.
    종합점수 = (PAC+SHO+PAS+DRI+PHY)/5 * 경기수 + 이동거리 + 스프린트*2
    """
    u = current_user()
    if not u:
        return jsonify({"error": "로그인 필요"}), 401
    with get_db() as db:
        m = db.execute("SELECT 1 FROM league_members WHERE league_id=? AND user_id=?",
                       (league_id, u['id'])).fetchone()
        if not m:
            return jsonify({"error": "리그 멤버만 조회 가능"}), 403
        members = db.execute(
            "SELECT m.user_id, m.player_name, m.position, u.username "
            "FROM league_members m JOIN users u ON m.user_id=u.id "
            "WHERE m.league_id=?", (league_id,)).fetchall()

    standings = []
    for mem in members:
        uid = mem['user_id']
        with get_db() as db:
            rows = db.execute(
                "SELECT stats_json FROM analysis_history "
                "WHERE user_id=? ORDER BY id DESC LIMIT 5", (uid,)).fetchall()
        if not rows:
            continue
        pac_l, sho_l, pas_l, dri_l, phy_l = [], [], [], [], []
        dist_t, spr_t, spd_best, games = 0.0, 0, 0.0, 0
        for r in rows:
            try:
                s = json.loads(r['stats_json'])
                pac_l.append(s.get('PAC', 60))
                sho_l.append(s.get('SHO', 60))
                pas_l.append(s.get('PAS', 60))
                dri_l.append(s.get('DRI', 60))
                phy_l.append(s.get('PHY', 60))
                dist_t += float(s.get('total_distance_km', 0))
                spr_t  += int(s.get('sprint_count', 0))
                spd_best = max(spd_best, float(s.get('top_speed_kmh', 0)))
                games  += 1
            except Exception:
                pass
        if not games:
            continue
        avg_overall = (
            np.mean(pac_l) + np.mean(sho_l) + np.mean(pas_l) +
            np.mean(dri_l) + np.mean(phy_l)
        ) / 5
        score = round(avg_overall * games + dist_t + spr_t * 2, 1)
        standings.append({
            "player_name":       mem['player_name'] or mem['username'],
            "position":          mem['position'],
            "games":             games,
            "avg_overall":       round(avg_overall, 1),
            "total_distance_km": round(dist_t, 2),
            "top_speed_kmh":     round(spd_best, 1),
            "sprint_total":      spr_t,
            "score":             score,
        })
    standings.sort(key=lambda x: x['score'], reverse=True)
    for i, s in enumerate(standings):
        s['rank'] = i + 1
    return jsonify(standings)


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
