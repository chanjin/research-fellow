# 공유현상 중심 연구동반자: 다음 개발 세션 브리프

## 1. 목적과 현재 판단

이 코드베이스는 기존의 기능별 화면·큐·알림을 계속 덧붙이는 방식 대신, 잭슨의 문제프레임에서 말하는 **공유현상(shared phenomenon)** 을 협력의 중심으로 둔다.

이 방향은 맞다. M1 Curator, M2 Advisor, 연구자, 외부 자문 요청자는 각각 독립된 도메인이고, 그 사이에서 관찰·제어되는 요청, 보고, 의뢰, 승인, 통지가 명시돼야 한다. 다만 현재 코드는 **구조를 검증하는 기준 구현**이다. 바로 기능을 크게 추가하기보다, 아래의 불변 규칙과 수직 시나리오를 먼저 견고하게 만드는 것이 중요하다.

### 현재 구현 범위

- Streamlit UI: 연구위원 홈, M1 자료·탐색, M2 연구 방향, 외부 자문
- SQLite: `cases`, `phenomena`, `decisions`
- JSONL: 연구자 승인 후의 `KnowledgeCard`만 저장
- M1: PDF/TXT/MD 텍스트 추출 → 후보 지식 카드 → 연구자 승인 → M2 통지
- M2: 연구 질문 검토 → M1 탐색 Intent 제안 → 연구자 승인 → M1 실행함 전달
- 외부 자문: 요청 → M2 전문성 기반 해석·범위 확인 → 답변 초안
- Ollama: 초안 생성만 담당하며 상태 전이는 LLM 구조화 JSON에 의존하지 않음

### 아직 의도적으로 단순화한 부분

- PDF에서 다수의 근거 카드, 페이지, 인용 문맥을 정확히 추출하지 않는다.
- 임베딩·LlamaIndex·관계 그래프는 아직 이 기준 구현에 통합하지 않았다.
- M1 Intent 실행은 사람이 결과 요약을 입력하는 형태다.
- 외부 자문 해석의 요청자 확인·보정과 최종 발송 이력이 완결돼 있지 않다.
- 사례 단위의 상태 전이 규칙, 재시도, 감사 로그, 테스트가 최소 수준이다.

## 2. 핵심 요구사항과 불변 규칙

### 역할

- **M1 Curator**: 외부 문헌과 연구 노트를 근거 가능한 지식 후보로 구조화하고, 승인 지식의 변화·상충·공백을 M2에 알린다.
- **M2 Advisor**: 연구자의 연구 상태와 M1의 지식을 해석하여 연구 방향, 비판, 후속 탐색 Intent, 외부 자문 답변을 만든다.
- **연구자**: 지식·연구 방향·후속 실행의 확정 권한을 가진다. 긴 폼 작성보다 다중 선택 승인과 짧은 보충 의견이 기본이어야 한다.
- **외부 자문 요청자**: M2에 질문·맥락을 제시하고, M2의 전문성 기반 요청 해석과 자문 답변을 받는다. 연구 방향을 승인하는 주체는 아니다.

### 공유현상 유형

| 유형 | 제어 → 관찰 | 의미 |
| --- | --- | --- |
| `research_update` | 연구자 → M2 | 연구 질문, 진척, 제약, 새 아이디어의 제출·수정 |
| `advice_report` | M2 → 연구자 | 연구 상태 해석, 비판, 권고, 보고 |
| `decision_request` | M1/M2 → 연구자 | 연구자 승인이 필요한 지식·관계·Intent·자문 초안 |
| `decision` | 연구자 → M1/M2 | 승인, 보류, 반려, 의견 |
| `curation_intent` | M2 → M1 | 탐색·검증·계보 생성 의뢰 |
| `knowledge_update` | M1 → M2/연구자 | 신규 지식, 상충, 공백, 탐색 결과 통지 |
| `advisory_exchange` | 외부 요청자 ↔ M2 | 요청, 전문성 기반 해석 확인, 답변, 후속 질문 |

### 불변 규칙

1. **공유현상과 저장물은 다르다.** JSONL 지식 카드, SQLite 레코드, 화면의 알림은 공유현상을 기록하거나 투영한 결과이지 공유현상 자체가 아니다.
2. **승인 전 후보 지식은 의미 기억에 저장하지 않는다.** `KnowledgeCard`의 JSONL 저장은 연구자 승인 뒤에만 발생한다.
3. **근거 없는 단정 금지.** M2의 보고·외부 자문은 근거 카드, 적용 조건, 한계를 분리해서 표시한다.
4. **LLM은 상태 전이를 결정하지 않는다.** LLM은 초안을 만들 수 있으나, ID·승인 상태·근거 참조·큐 이동은 결정론적 코드가 검증한다.
5. **연구자 결정을 재사용한다.** 승인 한 번은 후속 공유현상(`knowledge_update`, `curation_intent`)으로 변환되고, 같은 내용을 다시 입력하게 하지 않는다.
6. **외부 자문에는 전문성 경계가 있다.** 요청 원문을 바로 답하지 않고, M2가 해석한 문제·답변 범위·범위 밖 항목·추가 확인 사항을 먼저 제시한다.

## 3. 권장 모듈 구조

```text
app.py                              # 화면 조립과 라우팅만 담당
src/research_fellow/
  domain/
    phenomena.py                    # 유형별 payload dataclass/Pydantic 모델
    cases.py                         # ResearchCase, ExternalAdvisoryCase
    knowledge.py                     # KnowledgeCard, Evidence, KnowledgeRelation
  application/
    decisions.py                     # 승인·보류·반려 및 후속 현상 생성
    curation.py                      # M1 문서 지식화·Intent 실행
    advising.py                      # M2 상태 파악·가설/설계 비판·자문 해석
    projections.py                   # 홈, 승인함, M1/M2 입력함 조회
  infrastructure/
    sqlite_ledger.py                 # cases/phenomena/decisions 저장소
    jsonl_memory.py                  # 승인 지식 카드 canonical store
    document_reader.py               # PDF/TXT/MD 추출 및 페이지 보존
    retrieval.py                     # 초기 lexical → 이후 LlamaIndex adapter
    ollama_client.py                 # 비구조화 초안 생성과 재시도
  prompts/
    m1_curation.md
    m2_research_review.md
    m2_external_interpretation.md
    m2_external_response.md
  ui/
    home.py
    inbox.py
    m1_workspace.py
    m2_workspace.py
    external_advisory.py
tests/
```

처음부터 이 구조로 파일을 나누되, **도메인 모델과 저장소를 UI에서 직접 호출하지 않는다.** UI는 application service만 호출한다.

## 4. 모듈별 개발 계획

### P0. 기반 안정화 — 먼저 수행

**목표:** 공유현상 원장을 신뢰할 수 있게 만들고 이후 UI·LLM 변경의 영향을 격리한다.

- `Phenomenon` payload를 Pydantic 모델로 유형별 검증
- 상태 전이표 명시: `proposed → approved/deferred/rejected`, `ready → completed/failed`
- `decision_request`의 대상 ID, 생산자, 수신자, 근거 참조가 필수인지 정의
- 중복 결정 방지 및 결정의 멱등성(idempotency) 보장
- `case_id`별 타임라인 조회와 감사 로그
- SQLite migration 또는 schema version 도입
- 단위 테스트: 승인한 지식만 JSONL에 저장되는지, 승인 Intent만 M1에 나타나는지

**완료 기준:** LLM 없이도 M2 Intent → 연구자 승인 → M1 실행 → M2 통지의 전체 상태 전이가 자동 테스트로 통과한다.

### P1. M1 문서 지식화 — 첫 번째 수직 기능

**목표:** 논문 PDF·연구 노트에서 연구자가 검토 가능한 다수의 후보 지식 카드를 만든다.

- PDF 페이지별 텍스트·쪽수·섹션 보존
- 연구자 자료의 성격 태그: `external_paper`, `researcher_published_work`, `researcher_idea_note`
- 후보 카드에 `claim`, `evidence_excerpt`, `evidence_pages`, `citation_markers`, `labels`, `conditions`, `limits` 포함
- 한 문서에서 1개가 아닌 여러 카드 제안
- LLM 출력은 엄격한 스키마 검증 대신, **텍스트 초안 → 결정론적 정규화·검증 → 부족 필드 보완 요청** 방식으로 처리
- 연구자 다중 카드 선택 승인/반려

**완료 기준:** PDF 한 편에서 최소 3개 후보 카드와 페이지 근거를 만들고, 선택 승인 카드만 JSONL에 저장된다.

### P2. 지식 관계·계보 — M1의 두 번째 수직 기능

**목표:** 문헌 노드가 아니라 승인된 **지식 간 관계**를 제안·검토·저장한다.

- `KnowledgeRelation`: `supports`, `extends`, `contradicts`, `qualifies`, `uses_method`, `addresses_gap`
- 관계마다 출발/도착 카드, 관계 근거, 조건, 신뢰 수준 포함
- 후보 관계도 `decision_request`로 연구자에게 제시하고 다중 승인
- 계보 맵은 최대 10개 내외의 핵심 지식 노드만 보이고, 상세 설명은 텍스트 요약으로 제공
- 그래프 렌더링은 저장소가 아니라 projection; Graphify는 구현 산출물 관리 용도와 연구 지식 그래프를 혼동하지 않음

**완료 기준:** 승인 카드 10개 이하에서 관계 승인 후, 개념/방법 계보와 텍스트 요약이 일관되게 보인다.

### P3. 검색·의미 기억 — M1과 M2의 공통 기반

**목표:** M2가 전체 JSONL을 무작정 프롬프트에 넣지 않고 관련 승인 지식을 근거로 사용한다.

- JSONL을 canonical store로 유지
- metadata/label/키워드 lexical 검색을 baseline으로 유지
- LlamaIndex는 ingestion·chunking·retrieval adapter로 도입
- Ollama embedding model(예: `nomic-embed-text`)을 선택적으로 사용
- 임베딩, 인덱스, JSONL 원본의 버전·재생성 정책 명시
- 검색 결과에 card ID, 출처, 유사도/선정 이유 제공

**완료 기준:** 같은 질문에서 lexical fallback과 semantic retrieval이 모두 작동하고, M2 보고는 카드 ID와 출처를 인용한다.

### P4. M2 연구 파악·방향 — 두 번째 핵심 수직 기능

**목표:** 연구자의 질문·진척과 M1 업데이트를 연구 상태로 종합하고, 연구자가 처리할 판단만 간결히 제시한다.

- `ResearchState`: 질문, 현재 가설, 제약, 미결 사항, 최근 근거 변화, 확신 수준
- 연구 검토: 지지 근거, 반대 근거, 숨은 가정, 공백, 다음 행동을 분리
- M1 `knowledge_update`를 읽어 M2의 새 정보 입력함·요약 보고 생성
- `curation_intent`를 구조화: 목적, 질문, 레이블, 우선순위, 기대 근거, 완료 조건
- M2 화면의 상세 입력은 기본값/접기 처리; 연구자 홈의 승인함에서 다중 처리

**완료 기준:** 연구자가 한 개의 연구 질문만 입력해도 M2가 근거 기반 검토와 최대 3개의 선택 가능한 후속 Intent를 제시한다.

### P5. 외부 자문 사례 — 전문성 경계가 보이는 기능

**목표:** 외부 요청을 M2의 전문성으로 재해석하고, 내부 연구 축적과 외부 답변을 구분한다.

- `ExternalAdvisoryCase` 상태: `received → interpreted → scope_confirmed → answering → sent → closed`
- 요청 해석 payload: 실제 문제, 답할 수 있는 하위 질문, 범위 밖/전제, 확인 질문, 답변 전략
- 요청자의 해석 확인·보정 기록
- 답변: 주장/권고, 근거 카드, 조건, 한계, 추가 탐색 필요 여부
- 근거 부족 시 M2가 연구자에게 `decision_request`를 내고, 승인 시 M1 Intent 생성
- 발송 답변과 내부 연구자 보고를 서로 다른 projection으로 표시

**완료 기준:** 외부 자문 1건이 해석 확인을 거친 뒤, 근거가 충분하면 답변으로 닫히고 부족하면 M1 탐색으로 이어진다.

### P6. 연구자 경험과 운영성

**목표:** 연구자에게는 복잡한 작업실이 아니라, 보고·판단·후속 행동이 보이는 운영 화면을 제공한다.

- 홈: 현재 연구 상태, 새 지식, 대기 승인, 최근 보고, 외부 자문 상태
- 승인함: 유형별 필터, 체크박스 일괄 승인/보류/반려, 짧은 공통 의견
- 로그: 사례별 타임라인, 생산자/수신자/상태/근거 필터
- M1/M2 고급 작업실은 운영자용으로 접거나 분리
- 실행 중 버튼 비활성화, 진행 상태, 실패 시 재시도와 원인 표시

**완료 기준:** 연구자는 홈에서 시작해 승인함에서 판단하고, 별도 화면 탐색 없이 후속 작업이 자동으로 이어짐을 확인한다.

## 5. 권장 구현 순서

`P0 → P1 → P4 → P6 → P3 → P2 → P5`

- P1과 P4를 먼저 연결하면 “문서 지식화 → 승인 → M2 연구 검토”라는 가장 중요한 폐루프를 빠르게 검증할 수 있다.
- 벡터 검색(P3)은 P1에서 충분한 승인 카드가 쌓인 뒤 도입하는 편이 평가가 가능하다.
- 관계·계보(P2)는 좋은 카드와 검색 기반이 없는 상태에서 먼저 만들면 보기 좋은 그래프만 남을 위험이 있다.
- 외부 자문(P5)은 M2의 전문성·근거·한계 표현이 안정된 뒤 확장한다.

## 6. 다음 세션의 첫 작업 권고

다음 세션에서는 전체 기능을 다시 만들지 말고 **P0 + P1의 첫 절반**만 구현한다.

1. 현재 ZIP을 새 Git 저장소의 기준 커밋으로 둔다.
2. `domain/phenomena.py`와 `application/decisions.py`로 상태 전이와 승인을 분리한다.
3. `tests/test_decision_flow.py`를 먼저 작성한다.
4. M1 PDF 추출 결과를 페이지 단위 데이터로 바꾼다.
5. 후보 카드 다중 생성·다중 승인을 구현한다.
6. 실제 PDF 2~3편과 연구 노트 1개로 수동 acceptance test를 수행한다.

성공한 뒤에만 P4를 연결한다. 이 방식이 기능을 빠르게 늘리는 것보다 프롬프트·지식 스키마·승인 기준을 안정시키는 데 유리하다.

## 7. 다음 세션 시작 프롬프트

아래를 그대로 새 세션에 입력한다.

```text
나는 로컬 Mac에서 `shared-phenomena-research-fellow` 코드베이스를 개발하려고 한다.
이 프로젝트는 잭슨의 문제프레임에서 말하는 공유현상(shared phenomenon)을 중심으로 M1 Curator, M2 Advisor, 연구자, 외부 자문 요청자 간 협력을 구현한다.

현재 코드베이스를 먼저 읽고, 기존 기능을 무시하거나 별도 앱을 새로 만들지 말고 점진적으로 개선해줘. Python 3.11+, Streamlit, SQLite, JSONL, 로컬 Ollama를 사용한다. LLM은 초안 생성에만 사용하고, 공유현상의 상태 전이·ID·승인·근거 참조는 반드시 결정론적 코드와 검증으로 처리해야 한다.

핵심 불변 규칙:
1. JSONL 의미 기억에는 연구자 승인 후의 KnowledgeCard만 저장한다.
2. SQLite는 사례, 공유현상, 연구자 결정을 기록한다.
3. 공유현상과 저장물/화면 투영을 혼동하지 않는다.
4. 연구자는 다중 선택 승인·보류·반려와 짧은 의견 입력을 기본으로 사용한다.
5. M2의 M1 탐색 Intent는 연구자 승인 뒤에만 M1 실행함에 나타난다.
6. 외부 자문은 요청 원문을 곧바로 답하지 않고 M2 전문성 기반 요청 해석·범위 확인을 먼저 한다.
7. 모든 M2 보고와 외부 자문은 근거, 조건, 한계를 구분한다.

현재 구현에는 `cases`, `phenomena`, `decisions` SQLite 테이블과 승인 지식 JSONL이 있다. 공유현상 유형은 research_update, advice_report, decision_request, decision, curation_intent, knowledge_update, advisory_exchange 이다.

이번 세션의 범위는 P0과 P1의 첫 절반이다:
- phenomenon payload를 유형별 Pydantic 모델로 정의한다.
- 상태 전이와 승인 후속 동작을 UI에서 분리한 application service로 옮긴다.
- 단위 테스트를 추가한다: 승인 전 지식은 JSONL에 없고, 승인된 M2 Intent만 M1 실행함에 나타나야 한다.
- PDF 추출을 페이지 단위로 보존할 수 있게 리팩터링한다.
- 아직 LlamaIndex, 관계 그래프, 외부 자문 확장, 대규모 UI 재설계는 하지 않는다.

먼저 코드 구조와 변경 계획을 짧게 제시하고, 이어서 파일 단위로 구현해줘. 기존 사용자 데이터와 기존 DB를 파괴하지 말고 migration 또는 호환 경로를 제공해줘. 각 단계 뒤에는 실행 방법과 테스트 결과를 알려줘.
```
