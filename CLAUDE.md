# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 모델 사용 규칙

| 용도 | 모델 |
|------|------|
| 단순 작업 (파일 읽기/쓰기, 간단한 수정) | Haiku |
| 기본 개발 (기능 구현, 버그 수정, 코드 리뷰) | Sonnet |
| 아키텍처 설계 (시스템 설계, 대형 리팩터링 결정) | Opus |
| Planner (계획 수립, 설계 검토) | Opus |
| Worker (계획 실행, 반복 작업) | Haiku |

## 브리프 규칙
1. 요청이 불완전하면 작업 전에 반드시 질문한다
2. 폴더가 없으면 자동 생성 (`mkdir -p`)
3. 파일명은 날짜_내용 형식으로 (예: `20260417_feature.md`)
4. 모든 말은 한국어로 한다

## 개발자
- 이름: 인서 (Inseo Park) — 주니어 개발자, Python 기초 수준
- OS: Windows
- GitHub: https://github.com/98parks-create/aifootball

## Commands

```bash
# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
python app.py
# http://localhost:5000
```

## 프로젝트 개요 — TAD AI

AI 기반 축구 영상 분석 플랫폼. 스마트폰으로 찍은 영상을 업로드하면 선수를 추적하고 하이라이트/스탯/히트맵을 자동 생성.

**3단계 로드맵:**
- 1단계 (현재): 영상 업로드 → AI 분석 → 하이라이트 클립 + 스탯 + 히트맵
- 2단계: 소셜 레이어 — FIFA 카드 프로필, 스탯 누적, 팀 공유 피드
- 3단계: 학부모/유소년 특화 — 월간 성장 리포트 PDF, AI 코멘트, 코치 공유

## 폴더 구조

```
aifootball/
├── ABOUT-ME/       ← 개발자 소개
├── PROJECT/        ← 로드맵(roadmap.md), 기능 명세
├── DOCS/           ← 문서 템플릿
├── RESULT/         ← 결과물, 스크린샷
├── models/         ← yolov8s.pt, yolov8n.pt (gitignore됨)
├── src/            ← AI 핵심 로직
├── templates/      ← Flask HTML (index.html)
├── static/         ← 이미지, 영상, 하이라이트 (gitignore됨)
├── data/           ← 로컬 분석 데이터 (gitignore됨)
└── app.py          ← Flask 서버 (포트 5000)
```

## Architecture

**API 흐름:**
1. `POST /upload` — 영상 + 포지션 수신 → 첫 프레임 YOLO 감지 → 선수 목록 반환 (사용자가 타겟 선택)
2. `POST /analyze` — session_id + 4점 캘리브레이션 좌표 + 타겟 선수 수신 → 전체 영상 분석 → 스탯/하이라이트/히트맵 반환
3. `GET /progress/<session_id>` — 분석 진행률 폴링

**핵심 모듈 (`src/`):**
- `detector.py` (`FootballDetector`) — YOLOv8s + Supervision ByteTrack. 첫 프레임 선수 감지 및 전체 영상 추적. 타겟 락온 + 150px 반경 재탐색 로직 포함
- `analyzer.py` (`FootballAnalyzer`) — 포지션별 가중치 FIFA 스탯(PAC/PHY/DRI/PAS/SHO) 계산, 히트맵 PNG 생성, 하이라이트 클립 추출, 마스터 하이라이트 릴 생성
- `transformer.py` (`ViewTransformer`) — 사용자 지정 4점 → `cv2.getPerspectiveTransform`으로 픽셀 좌표를 100×50 정규화 필드로 변환

**저장소:** 분석 데이터는 `D:\aifootball_data\`에 저장 (repo 외부). Flask가 `/data/processed/`, `/static/highlights/`, `/static/calibration/` 라우트로 서빙.

**모델:** `models/yolov8s.pt` 필요 (class 0=사람, 32=공). git에서 제외됨 — 수동으로 다운로드 필요.

## 알려진 기술 부채
- `analyzer.py` 저장 경로 하드코딩 (`D:\aifootball_data`) → 환경변수로 분리 필요
- 인메모리 세션 (`analysis_sessions`) → 서버 재시작 시 소실, 추후 DB 교체 필요
- 이벤트 감지가 볼 터치 근접도 기반 → 골/어시스트 실제 구분 없음

## 작업 이력
- 2026-04-17: 초기 구조 정리 — ABOUT-ME/PROJECT/DOCS/RESULT 폴더 생성, models/ 폴더 분리, .gitignore 추가, 모델 경로 `models/yolov8s.pt`로 통일
- 2026-04-17: 품질 대형 업그레이드 — analyzer.py 전체 재작성 (실제 km/h 계산, 스프린트 감지, 타이틀카드/페이드/슬로우모션 영상), app.py 비동기 처리(threading + /results 엔드포인트), index.html 폴링 방식 전환, flask-cors 의존성 추가
- 2026-04-17: 추적 정확도 & UX 개선 — detector.py: 속도 벡터 예측 + 유니폼 색상 유사도 + 팀 컬러 필터 결합 위치 재탐색 (상대팀 0.15 패널티, 최소 score 0.22 임계값), 팀 인식 appearance re-ID 추가, 초기 락온 반경 180px. analyzer.py: 미니맵 제거(우측 상단 검은 박스), 타겟 인디케이터 1.5초 표시 후 0.7초 페이드 아웃, 속도 계산 윈도우 ±8 프레임, 클립 유효성 검사(타겟 20% 미만 추적 클립 제외)
