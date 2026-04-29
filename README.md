# ⚽ TAD AI — Football Analysis Platform

AI 기반 축구 경기 영상 분석 플랫폼. 스마트폰으로 찍은 영상을 업로드하면 선수를 자동 추적하고 하이라이트·스탯·히트맵을 생성합니다.

## 🚀 주요 기능

| 기능 | 설명 |
|------|------|
| **Player Tracking** | YOLOv8s + ByteTrack, HSV 유니폼 지문 기반 Re-ID |
| **Heatmap** | 원근 변환(4점 캘리브레이션) → 100×50 정규화 필드 히트맵 |
| **Highlights** | 이벤트별 자동 클립 추출 (드리블·슈팅·패스·태클·공 다툼 등) |
| **FIFA 카드** | PIL 생성 PNG + 인라인 HTML 카드 (Overall, PAC/SHO/PAS/DRI/PHY) |
| **성장 리포트 PDF** | A4 2페이지 PDF — 스탯·AI 분석 노트·경기 히스토리 그래프 |
| **코치 공유** | Gmail SMTP로 PDF 리포트 이메일 첨부 발송 |
| **팀 피드** | 초대코드 기반 팀 생성/참가, 하이라이트 클립 공유 |
| **리그 순위표** | 초대코드 기반 리그, 경기 수·평점·이동거리·스프린트 기반 점수 랭킹 |

## 🛠 Tech Stack

- **AI**: YOLOv8s (Ultralytics), Supervision, ByteTrack
- **Backend**: Flask (Python), SQLite
- **Frontend**: HTML5, Vanilla JS, Chart.js
- **Image/Video**: OpenCV, Pillow, NumPy

## 📂 폴더 구조

```
aifootball-main/
├── src/
│   ├── detector.py      # YOLOv8 + ByteTrack 추적, HSV Re-ID
│   ├── analyzer.py      # 스탯·하이라이트·히트맵·카드·PDF 생성
│   └── transformer.py   # 4점 → 정규화 필드 원근 변환
├── templates/
│   └── index.html       # 단일 페이지 앱 (로그인~대시보드~팀/리그)
├── models/              # yolov8s.pt (git 제외, 수동 다운로드)
├── bytetrack.yaml       # ByteTrack 추적 파라미터
├── app.py               # Flask 서버 (포트 5000)
└── requirements.txt
```

데이터 저장: `~/aifootball_data/` (repo 외부)

## ⚡ 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정 (.env)
MAIL_USER=your@gmail.com
MAIL_PASS=앱비밀번호
BASE_URL=http://localhost:5000

# 3. 모델 다운로드 (models/ 폴더에 배치)
# https://github.com/ultralytics/assets/releases → yolov8s.pt

# 4. 서버 실행
python app.py
# → http://localhost:5000
```

## 📈 개발 이력

### v1 — MVP
- [x] YOLO + ByteTrack 탐지/추적 파이프라인
- [x] 원근 변환 엔진 (4점 캘리브레이션)
- [x] 히트맵 및 스탯 생성
- [x] Flask 웹 대시보드

### v2 — 품질 업그레이드 (2026-04-17)
- [x] 비동기 분석 + 진행률 폴링
- [x] 실제 km/h 속도 계산 + 스프린트 감지
- [x] 타이틀카드·페이드·슬로우모션 하이라이트 영상
- [x] FIFA 카드 UI + 섹션 탭 필터
- [x] 로그인/회원가입 + 분석 히스토리

### v3 — 추적 정확도 개선 (2026-04-17)
- [x] 속도 벡터 예측 + 유니폼 색상 유사도 결합 재탐색
- [x] 상체+반바지 2-region HSV 지문
- [x] ID 스위치 즉시 감지 (코사인 유사도 0.30 이하 → 재탐색)
- [x] 소실 길이에 따라 탐색 반경 동적 확대
- [x] 클러스터(선수 밀집) 구간 지문 업데이트 동결

### v4 — 하이라이트 & 스탯 강화 (2026-04-30)
- [x] **볼 없는 슈팅 감지**: 속도 + 감속 패턴 기반 MOTION SHOT
- [x] **볼 희소 드리블 감지**: 볼 미감지 구간에서 속도 패턴 폴백
- [x] **공 다툼(BALL CONTEST)**: 2명 이상 대결 0.8초 이상 지속
- [x] **볼터치 횟수** 스탯 배지 추가
- [x] ByteTrack 파라미터 최적화 (track_buffer 90, match_thresh 0.85)

### v5 — 소셜 레이어 (2026-04-30)
- [x] **FIFA 카드 PNG API**: `/card/<session_id>` → Pillow 생성 다운로드
- [x] **성장 리포트 PDF**: `/report/<session_id>` → A4 2페이지 (스탯+히스토리)
- [x] **코치 공유**: PDF 이메일 첨부 발송 (`/api/report/share`)
- [x] **팀 시스템**: 초대코드 생성/참가, 팀 피드, 하이라이트 공유
- [x] **리그 순위표**: 초대코드 기반 리그, 복합 점수 랭킹

---
*TAD AI — AI Football Analysis Platform for Youth & Amateur Players*
