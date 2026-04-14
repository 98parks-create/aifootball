from flask import Flask, render_template, request, send_from_directory
import os
import json
from src.detector import FootballDetector

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('data', 'uploads')
PROCESSED_FOLDER = os.path.join('data', 'processed')
STATIC_IMAGES = os.path.join('static', 'images')
STATIC_VIDEOS = os.path.join('static', 'videos')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(STATIC_IMAGES, exist_ok=True)
os.makedirs(STATIC_VIDEOS, exist_ok=True)

@app.route('/')
def index():
    # 데이터가 있으면 대시보드 표시, 없으면 업로드 페이지
    has_results = os.path.exists(os.path.join('data', 'analysis', 'final_heatmap.png'))
    return render_template('index.html', has_results=has_results)

@app.route('/analyze', methods=['POST'])
def analyze():
    # 실제로는 업로드된 파일을 처리해야 하나, MVP에서는 샘플 영상을 다시 돌리는 버튼으로 구현
    detector = FootballDetector('yolov8n.pt')
    input_video = os.path.join(UPLOAD_FOLDER, 'sample.mp4')
    output_video = os.path.join(STATIC_VIDEOS, 'detected_video.mp4')
    
    stats = detector.process_video(input_video, output_video)
    
    # 히트맵 파일을 static으로 복사
    heatmap_src = os.path.join('data', 'analysis', 'final_heatmap.png')
    heatmap_dst = os.path.join(STATIC_IMAGES, 'heatmap.png')
    if os.path.exists(heatmap_src):
        import shutil
        shutil.copy(heatmap_src, heatmap_dst)
        
    return {"status": "success", "stats": stats}

if __name__ == '__main__':
    app.run(debug=True, port=5000)
