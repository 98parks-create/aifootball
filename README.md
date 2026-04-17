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

## 📈 진행 상황

### v1 — MVP
- [x] 프로젝트 초기화 및 폴더 구조 세팅
- [x] YOLO + ByteTrack 탐지/추적 파이프라인
- [x] 원근 변환(Perspective Transformation) 엔진
- [x] 히트맵 및 스탯 생성 로직
- [x] 프리미엄 웹 대시보드 (Flask + Modern UI)

### v2 — 품질 업그레이드 (2026-04-17)
- [x] 비동기 분석 (threading + 진행률 폴링)
- [x] 실제 km/h 속도 계산 + 스프린트 감지
- [x] 타이틀카드/페이드/슬로우모션 하이라이트
- [x] FIFA 카드 UI + 섹션 탭 필터
- [x] 로그인/회원가입 + 히스토리

### v3 — 추적 정확도 & UX 개선 (2026-04-17)
- [x] **타겟 추적 강화**: 속도 벡터 예측 + 유니폼 색상 유사도 + 팀 컬러 필터 결합
  - 위치 기반 재탐색 시 상대팀 선수에 0.15 패널티 적용
  - 예상 위치 = 마지막 위치 + 속도 × 경과 프레임으로 보정
- [x] **타겟 인디케이터 페이드**: 클립 시작 1.5초 표시 후 0.7초에 걸쳐 페이드 아웃
- [x] **미니맵 제거**: 우측 상단 검은 박스 제거, 클린 UI
- [x] **속도 계산 개선**: ±8 프레임 윈도우로 지터 감소 (기존 ±5)
- [x] **클립 유효성 검사**: 타겟 선수가 클립의 20% 미만으로 추적된 클립 제외
- [x] **팀 인식 재탐색**: 외모 기반 재식별 시 팀 컬러 활용

---
*TAD AI — AI Football Analysis Platform for Youth & Amateur Players*
