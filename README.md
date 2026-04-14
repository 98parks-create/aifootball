# ⚽ Football AI Analysis Platform (MVP)

AI 기반 축구 경기 영상 분석 플랫폼입니다. YOLOv8과 원근 변환(Perspective Transformation) 기술을 사용하여 선수들의 움직임을 정밀하게 분석하고 히트맵과 스탯을 생성합니다.

## 🚀 주요 기능
- **Player & Ball Detection**: YOLOv8(Large) 모델을 사용한 고정밀 객체 탐지
- **Object Tracking**: ByteTrack을 활용한 끊김 없는 선수 추적
- **Tactical Heatmap**: 원근 변환을 통한 2D 평면 기반의 전술적 히트맵 생성
- **Highlight Extraction**: 득점 찬스 및 주요장면 자동 추출
- **Interactive Dashboard**: 분석 결과를 한눈에 확인할 수 있는 프리미엄 웹 대시보드

## 🛠 Tech Stack
- **AI**: YOLOv8 (Ultralytics), Supervision, ByteTrack
- **Backend**: Flask (Python)
- **Frontend**: HTML5, Vanilla CSS (Modern Dark Mode), JS
- **Processing**: OpenCV, NumPy, Pandas, Matplotlib

## 📂 프로젝트 구조
```
aifootball/
├── data/
│   ├── uploads/      # 원본 영상 저장소
│   ├── processed/    # 탐지 처리된 영상 저장소
│   └── analysis/     # 히트맵 및 통계 데이터
├── src/
│   ├── detector.py   # AI 탐지 및 추적 로직
│   ├── transformer.py# 원근 변환 좌표 계산
│   └── analyzer.py   # 히트맵 및 스탯 생성
├── app.py            # 웹 서버 매니저
└── README.md
```

## 📈 진행 상황 (MVP)
- [x] 프로젝트 초기화 및 폴더 구조 세팅
- [x] Git 저장소 초기화
- [x] 가상환경 및 종속성 설치 (ultralytics, supervision, flask 등)
- [x] 샘플 영상 확보 완료 (`data/uploads/sample.mp4`)
- [x] AI 탐지 및 추적 테스트 완료 (`data/processed/detected_sample.mp4`)
- [x] 원근 변환(Perspective Transformation) 엔진 구현 완료
- [x] 히트맵 및 스탯 생성 로직 통합 완료
- [/] 프리미엄 웹 대시보드 (Flask + Modern UI) 개발 및 배포 중

---
*Developed by Antigravity (Advanced Agentic Coding AI)*
