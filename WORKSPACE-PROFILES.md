# Research Workspace Profiles

이 프로젝트는 코드/기능/프롬프트 골격을 한 벌만 유지하면서 연구 메모리와 전문성 컨텍스트를 Workspace 단위로 분리합니다.

## Workspace

### 1. General Research Fellow
- URL: `?workspace=general`
- Local DB: `data/research_fellow.db`
- Cache: `.cache/research-fellow/`
- 목적: 전체 관심사와 장기 연구 메모리
- 기존 legacy JSONL이 있으면 이 Workspace에서만 1회 마이그레이션합니다.

### 2. Agent Development Research Fellow
- URL: `?workspace=agent_development`
- Local DB: `data/research_fellow_agent_development.db`
- Cache: `.cache/research-fellow-agent-development/`
- 목적: 현재 논문과 AI 에이전트 개발 전문 연구
- 별도 DB로 시작하며 General Workspace의 지식을 자동 복사하지 않습니다.
- Prompt Policy에 에이전트 개발 전문성 컨텍스트가 추가됩니다.

## 동시에 두 페이지 사용

같은 Streamlit 앱을 띄운 뒤 브라우저 탭을 두 개 열어 각각 아래 주소를 사용합니다.

- `http://localhost:8501/?workspace=general`
- `http://localhost:8501/?workspace=agent_development`

사이드바의 `Research workspace` 선택으로도 전환할 수 있습니다.

## Prompt 구조

Task Prompt는 한 벌입니다.

`Task Prompt + Common Research Fellow Policy + Workspace Expertise Injection + Output Language Injection`

따라서 Sensemaking/M2/Ontology/Paper Reading 등을 Workspace별로 복제하지 않습니다.

## 서버 동기화

같은 서버 디렉터리를 지정해도 파일명은 분리됩니다.

- General: `research_fellow.db`
- Agent Development: `research_fellow_agent_development.db`

필요하면 Workspace별 환경변수를 사용할 수 있습니다.

- `RESEARCH_FELLOW_SERVER_DIR_GENERAL`
- `RESEARCH_FELLOW_SERVER_DIR_AGENT_DEVELOPMENT`
- `RESEARCH_FELLOW_DB_FILENAME_GENERAL`
- `RESEARCH_FELLOW_DB_FILENAME_AGENT_DEVELOPMENT`
- `RESEARCH_FELLOW_CACHE_DIR_GENERAL`
- `RESEARCH_FELLOW_CACHE_DIR_AGENT_DEVELOPMENT`

기존 `RESEARCH_FELLOW_SERVER_DIR`, `RESEARCH_FELLOW_DB_FILENAME`, `RESEARCH_FELLOW_CACHE_DIR`은 General Workspace에만 하위 호환으로 적용됩니다.
