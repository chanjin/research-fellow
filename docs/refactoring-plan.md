
좋습니다. 그 순서가 가장 맞습니다. 지금부터는 **DSL을 먼저 강제하지 않고, 현재 구현에서 반복되는 실행 메커니즘을 정리한 뒤 그 결과를 바탕으로 DSL을 정의하고, 마지막에 DSL에 맞춰 Python 구조를 다시 정렬**하는 방식으로 가면 됩니다.

전체 작업 순서는 이렇게 고정하는 게 좋겠습니다.

1. **선 리팩토링: 현재 Python의 공통 실행 메커니즘 추출**
   - LLM 호출 / retry / parse / validation / fallback
   - prompt 실행 인터페이스
   - manual recovery / external chat
   - workflow 상태 전이
   - ledger / trace / logging
   - 공통 입력·출력 정규화
   - 중복 helper 정리
   - 단, 업무 의미가 다른 로직은 억지로 합치지 않음

2. **DSL 정의: AJD Workflow DSL v0.1**
   - 코드 구조가 아니라 직무 관점에서 vocabulary 정의
   - `purpose`
   - `trigger`
   - `inputs`
   - `steps`
   - `decision`
   - `authority`
   - `memory`
   - `phenomenon`
   - `outputs`
   - `completion`
   - runtime concern과 업무 concern의 경계 명확화

3. **DSL 기준 Python 재리팩토링**
   - DSL의 각 의미 단위와 Python capability가 1:1 또는 명확한 N:1 관계를 갖도록 정리
   - workflow orchestration은 YAML이 소유
   - Python은 capability implementation 역할로 제한
   - handler 이름도 구현 함수명이 아니라 업무 capability 이름 중심으로 재구성

4. **기존 DSL 재작성**
   지금 만든
   - `research_cycle.yaml`
   - `auto_literature_review.yaml`
   - `paper_reading_summary.yaml`
   
   을 최종 DSL 규칙에 맞게 다시 단순화합니다.

5. **나머지 Workflow 확장**
   - Evidence Investigation
   - Knowledge Curation
   - Research State Update
   - Advisory
   - Coauthor
   - Ontology 관련 workflow

핵심적으로는 아래 방향입니다.

```text
현재
Workflow 의미 + 실행 메커니즘 + 저장 + LLM 제어
                ↓
            Python 파일


1차 리팩토링 후
Domain/Application Capability
        +
Common Runtime Services


DSL 정의 후
AJD
 ↓
Workflow DSL
 ↓
Capability
 ↓
Common Runtime
 ↓
LLM / Tool / Storage
```

그리고 리팩토링 원칙 하나는 명확히 잡는 게 좋습니다.

> **먼저 반복되는 실행 메커니즘을 제거하고, 이후 DSL에서는 업무 의미만 남긴다.**

예를 들어 아래는 공통화 대상입니다.

```text
prompt render
→ LLM call
→ retry
→ output parse
→ validation
→ fallback
→ recovery
→ trace
```

반대로 다음은 별개의 capability로 유지합니다.

```text
Research Question 우선순위 판단
Paper relevance 판단
Knowledge conflict 판단
Ontology relation 판단
```

겉으로는 모두 LLM 평가이지만 **직무 의미가 다르기 때문**입니다.

따라서 다음 실제 작업은 **현재 전체 `application/*.py`를 기준으로 중복 패턴을 조사해서**

- 반복 패턴
- 등장 파일
- 공통화 후보
- 추출할 모듈
- 리팩토링 우선순위
- 그대로 남길 업무 고유 코드

를 정리한 **Python Pre-DSL Refactoring Map**을 만드는 것으로 시작하는 게 좋겠습니다.

그 맵을 먼저 보고 구조를 확정한 다음 코드 리팩토링에 들어가면 됩니다.




맞습니다. 지금부터는 “어떤 Python 파일을 다음에 YAML로 옮길까”보다 먼저, **DSL을 무엇으로 정의할지와 전체 전환 계획을 고정**하는 게 좋습니다.

제가 보기에는 이 프로젝트에서 DSL의 목적은 단순한 workflow serialization이 아니라 다음입니다.

> **AJD에 정의된 직무를 사람이 읽고 검토할 수 있는 수준으로 추상화하면서, 동시에 실행 가능한 명세로 만드는 것**

즉 DSL은 코드의 복사본이 아니라 **코드보다 한 단계 높은 설계 언어**여야 합니다.

---

## 1. DSL의 추상화 원칙부터 고정하는 것이 좋습니다

DSL에는 “무엇을 수행하는가”가 보여야 하고, “어떻게 코딩했는가”는 최대한 감춰야 합니다.

예를 들어 좋지 않은 DSL은 이런 것입니다.

```yaml
- handler: parse_rq_priority_assessment
- handler: update_research_question
- handler: create_curation_intent
- handler: append_ledger
```

이건 Python 코드를 YAML로 옮긴 것에 가깝습니다.

우리가 원하는 것은:

```yaml
- id: prioritize_research_questions
  performer: m2
  purpose: >
    현재 연구 상태에서 추가 조사가 필요한
    연구질문의 우선순위를 결정한다.

- id: request_literature_review
  performer: m2
  emits:
    phenomenon: curation_intent
    to: m1
```

입니다.

실제 `parse`, DB 저장, ledger append 같은 것은 DSL 아래 runtime/capability에 숨어 있어야 합니다.

---

# 2. 전체 구조는 4단계로 보는 것이 가장 깔끔합니다

```text
APF
세상과 Agent의 경계
공유현상
        ↓
AJD
Agent의 직무·책임·권한·기억
        ↓
Workflow DSL
직무 수행 절차
        ↓
Capability Implementation
Python / LLM / Tool / Storage
```

각 단계가 답하는 질문도 다릅니다.

| 계층 | 핵심 질문 |
|---|---|
| APF | Agent는 어떤 세상과 상호작용하는가 |
| AJD | Agent가 맡은 직무와 책임은 무엇인가 |
| Workflow DSL | 그 책임을 어떻게 수행하는가 |
| Python/LLM | 각 동작을 실제로 어떻게 구현하는가 |

DSL은 정확히 **AJD와 구현 사이**에 있어야 합니다.

---

# 3. Workflow DSL에서 보여야 할 것은 7가지 정도면 충분합니다

저는 지금보다 DSL vocabulary를 오히려 줄이는 것을 추천합니다.

### ① 목적

```yaml
purpose:
```

이 workflow가 왜 존재하는가.

---

### ② 시작 조건

```yaml
trigger:
```

어떤 공유현상이나 상태 변화 때문에 시작되는가.

---

### ③ 입력

```yaml
inputs:
```

업무 수행에 필요한 정보.

---

### ④ 수행 단계

```yaml
steps:
```

업무적으로 의미 있는 단계만 표현.

---

### ⑤ 판단/권한

```yaml
decision:
authority:
```

LLM이 판단하는 것인지, 사람이 승인해야 하는 것인지.

---

### ⑥ 출력 및 공유현상

```yaml
outputs:
emits:
```

세상이나 다른 Agent에게 무엇이 전달되는가.

---

### ⑦ 완료조건

```yaml
completion:
```

무엇을 만족하면 이 업무가 끝난 것인가.

이 정도면 AJD의 실행 명세로 충분합니다.

---

# 4. `code / llm / human`도 한 단계 더 추상화할 수 있습니다

기존에 제가 제안했던

```yaml
type: code
type: llm
type: human
```

은 구현 migration 단계에는 유용하지만, 최종 DSL에서는 너무 구현 중심일 수 있습니다.

오히려:

```yaml
performed_by: agent
```

그리고 필요한 경우:

```yaml
judgment:
  mode: llm
```

정도만 드러내는 것이 좋습니다.

예:

```yaml
- id: assess_research_priority

  purpose: >
    현재 지식과 연구 맥락을 기준으로
    추가 조사의 우선순위를 판단한다.

  judgment:
    mode: llm

  produces:
    - prioritized_research_questions
```

반면 저장은 DSL에서 아예 없어도 됩니다.

```text
save_to_sqlite()
append_ledger()
serialize_json()
parse_output()
```

이런 것은 **Workflow가 아니라 runtime concern**입니다.

---

# 5. 그래서 현재 만든 DSL도 한 번 Refactoring할 필요가 있습니다

지금까지 Step 1~3에서 만든 DSL은 **migration DSL**에 가깝습니다.

예:

```yaml
handler: research_cycle.dispatch_to_m1
parser: parse_auto_search_strategy
workflow: m1_auto_literature_review
```

처음 코드를 외화하기 위해서는 필요했습니다.

하지만 최종 형태에서는:

```yaml
- id: request_literature_review

  purpose: >
    선정된 연구질문에 필요한 문헌조사를
    Research Curator에게 요청한다.

  emits:
    phenomenon: curation_intent
    to: research_curator
```

가 되어야 합니다.

그 아래 mapping에서만

```text
request_literature_review
    → Python handler
```

를 연결하는 편이 좋습니다.

즉:

```text
Workflow DSL
     ↓
Capability Registry
     ↓
Python
```

구조가 필요합니다.

---

# 6. 전체 DSL 전환 Plan은 기능 파일 기준보다 “직무” 기준으로 잡는 것이 좋습니다

현재 파일별로 진행하면 다시 구현 구조에 끌려갑니다.

대신 M1/M2 직무 기준으로 봐야 합니다.

## Phase A. M2 — 연구 방향 관리

### Workflow A1. Research State Review

```text
지식 변화 감지
↓
현재 연구질문 재검토
↓
새 Research Question 생성/보강
↓
우선순위 결정
↓
문헌조사 필요 여부 판단
```

현재 `research_cycle.py`가 여기에 해당합니다.

---

### Workflow A2. Literature Review Delegation

```text
선정된 Research Question
↓
문헌조사 목적 정의
↓
Curation Intent 생성
↓
M1에 위임
```

현재는 A1 안에 들어가 있지만, 의미상 분리할 수도 있습니다.

---

### Workflow A3. Research State Update

```text
M1 조사 결과 수신
↓
기존 연구상태와 비교
↓
Research Question 상태 갱신
↓
추가 연구 필요성 판단
```

이것은 앞으로 M2의 핵심 workflow가 됩니다.

---

# 7. M1 — 문헌 조사와 지식 획득

M1은 크게 4개의 직무 workflow로 정리하는 것이 좋아 보입니다.

## Workflow B1. Literature Discovery

```text
Curation Intent 수신
↓
검색전략 수립
↓
문헌 탐색
↓
후보 문헌 평가
↓
읽을 논문 선정
```

현재 `auto_literature.py`의 전반부입니다.

---

## Workflow B2. Paper Reading

```text
논문 원문 확보
↓
논문 자체 내용 파악
↓
핵심 주장·방법·결과 요약
↓
현재 연구 맥락에서 확인할 질문 도출
```

현재 Step 3에서 일부 DSL화한 부분입니다.

여기서 중요한 점은:

> **논문 요약과 연구 맥락 해석을 개념적으로 분리**

하는 것입니다.

---

## Workflow B3. Evidence Investigation

현재 말한 second pass는 사실 이 이름이 더 적절합니다.

```text
Reading Question 선택
↓
관련 원문 부분 재검토
↓
질문에 대한 답변
↓
근거 확인
↓
주장과 근거 연결
```

`second_pass_prompt`라는 구현 이름보다 **Evidence Investigation**이 직무 관점에서는 훨씬 명확합니다.

---

## Workflow B4. Knowledge Curation

```text
논문에서 확보한 주장·근거
↓
기존 지식과 비교
↓
신규 / 보강 / 충돌 판단
↓
Knowledge Card 후보 작성
↓
승인 또는 병합
```

여기까지가 M1의 지식 획득 전체 사이클입니다.

---

# 8. 이를 상위 workflow로 조립할 수 있습니다

최종적으로 M1은 다음 하나의 상위 DSL로 읽힐 수 있습니다.

```yaml
workflow: literature_knowledge_acquisition

purpose: >
  연구질문과 관련된 문헌을 탐색하고,
  검토 가능한 근거를 확보하여
  연구 지식으로 축적한다.

steps:

  - discover_relevant_literature

  - read_selected_papers

  - investigate_evidence

  - curate_knowledge
```

이 정도 수준이면 처음 보는 사람도 Agent가 하는 일을 이해할 수 있습니다.

그리고 상세한 내용을 보고 싶으면 하위 workflow로 내려갑니다.

```text
literature_knowledge_acquisition
    │
    ├── literature_discovery
    │
    ├── paper_reading
    │
    ├── evidence_investigation
    │
    └── knowledge_curation
```

이게 DSL을 만드는 가장 큰 가치라고 봅니다.

---

# 9. M2와 M1을 연결하면 전체 Research Agent는 이렇게 됩니다

```text
                Researcher
                    │
                    │ research goal / feedback
                    ▼
           ┌─────────────────┐
           │       M2        │
           │ Research Advisor│
           └─────────────────┘
                    │
          Research State Review
                    │
                    ▼
             Research Question
                    │
              Curation Intent
                    │
                    ▼
           ┌─────────────────┐
           │       M1        │
           │Research Curator │
           └─────────────────┘
                    │
          Literature Discovery
                    │
               Paper Reading
                    │
          Evidence Investigation
                    │
           Knowledge Curation
                    │
                    ▼
              Knowledge Cards
                    │
                    └────────────► M2
```

이 그림 자체가 사실 AJD workflow architecture가 됩니다.

---

# 10. 그리고 Human Interaction을 명확히 해야 합니다

이건 DSL에서 매우 중요합니다.

현재 UI에 구현되어 있다고 해서 interaction을 그냥 코드로 취급하면 안 됩니다.

예를 들어:

```yaml
- id: select_reading_questions

  performed_by: researcher

  purpose: >
    논문을 더 깊게 검토할 가치가 있는
    질문을 선택하거나 새로운 질문을 제시한다.
```

즉 인간의 interaction도 **workflow step**입니다.

이를 넣어야:

```text
Agent autonomy
vs.
Human authority
```

가 설계에서 보입니다.

이는 논문에서 말씀하신 **책임과 자율성**과도 직접 연결됩니다.

---

# 11. Memory도 저장 API가 아니라 의미 기준으로 표현해야 합니다

DSL에서:

```yaml
memory:
  reads:
    - research_questions
    - knowledge_cards
    - literature_history

  writes:
    - paper_analysis
    - evidence
    - knowledge_cards
```

정도만 보여주면 됩니다.

SQLite table 이름이나 repository 함수는 DSL에 없어야 합니다.

그래야 저장 방식이 바뀌어도 명세가 유지됩니다.

---

# 12. Retry / parsing / JSON validation도 대부분 DSL에서 내려도 됩니다

현재 migration DSL에는:

```yaml
retry:
parser:
validator:
fallback:
```

가 꽤 있습니다.

그런데 상당수는 모든 LLM 호출에 공통입니다.

따라서 최종적으로는:

```text
LLM execution policy
```

라는 runtime policy로 빼는 것이 좋습니다.

DSL에는 정말 업무적으로 중요한 failure만 표현합니다.

예:

```yaml
exception:
  insufficient_evidence:
    action: request_additional_literature
```

이건 업무입니다.

반면:

```yaml
json_parse_error:
  retry: 3
```

은 runtime입니다.

둘을 구분해야 DSL이 깨끗해집니다.

---

# 13. 그래서 전체 작업을 4개의 큰 Phase로 나누는 것을 제안합니다

## Phase 1 — DSL Language 정리

지금 여기부터 하는 게 좋습니다.

목표:

> DSL에서 구현 세부사항을 제거하고 AJD 실행 언어로 정의

할 일:

- DSL vocabulary 정의
- workflow / step / phenomenon / memory / decision / authority 정의
- runtime concern과 business concern 분리
- 기존 Step 1~3 YAML을 기준 DSL에 맞게 다시 정리

이걸 먼저 Freeze합니다.

---

## Phase 2 — 핵심 직무 Workflow 외화

순서:

```text
M2
1. research_state_review

M1
2. literature_discovery
3. paper_reading
4. evidence_investigation
5. knowledge_curation

M2
6. research_state_update
```

여기까지 하면 연구위원 Agent의 핵심 loop가 DSL로 보입니다.

---

## Phase 3 — 다른 직무 확장

그 다음:

```text
M2
- advisory
- paper coauthor
- research direction review

M1
- ontology curation
- knowledge maintenance
- literature monitoring
```

등을 옮깁니다.

---

## Phase 4 — 실행기 정리

Workflow들이 안정화된 후:

```text
DSL
↓
generic workflow runtime
↓
capability registry
↓
Python / LLM / tools
```

로 실행기를 일반화합니다.

이 순서가 중요한 이유는 **실행기를 먼저 일반화하면 DSL이 엔진의 제약에 끌려가기 때문**입니다.

먼저 좋은 명세를 만들고,
그다음 엔진이 그 명세를 실행하도록 만드는 것이 맞습니다.

---

# 14. 현재 시점에서는 coding을 잠시 멈추고 DSL Definition을 먼저 작성하는 것을 추천합니다

현재 Step 1~3은 매우 유용한 프로토타입입니다.

이제는 여기서 발견한 것을 가지고:

### `AJD Workflow DSL v0.1`

문서를 먼저 만드는 것이 좋습니다.

대략:

```text
1. Purpose
2. Design Principles
3. Core Concepts
   - Workflow
   - Step
   - Phenomenon
   - Decision
   - Authority
   - Memory
4. Syntax
5. Runtime Boundary
6. M1 Example
7. M2 Example
```

그리고 핵심 원칙은 한 문장으로:

> **The DSL specifies what a job agent does and why, while implementation details of how each capability is executed remain outside the workflow specification.**

한글로 하면:

> **DSL은 에이전트가 직무 수행 과정에서 무엇을 왜 하는지를 명세하며, 각 능력을 어떻게 구현하는지는 명세 밖에 둔다.**

이 원칙을 먼저 세우면 이후 YAML이 훨씬 짧고 명확해질 겁니다.

---

제가 제안하는 **바로 다음 작업**은 `second_pass` 코딩이 아닙니다.

**① 지금까지 만든 DSL 3개를 사례로 삼아 `AJD Workflow DSL v0.1` 규칙을 먼저 정의하고,  
② 그 규칙으로 M1/M2 전체 Workflow Map을 작성한 뒤,  
③ 기존 YAML을 한 번 정리하고,  
④ 이후 나머지 workflow를 순차적으로 옮기는 것**이 좋겠습니다.

이렇게 해야 이번 DSL 작업이 단순 리팩터링이 아니라, 처음 의도하셨던 **“설계도가 있는 Agent”의 실행 명세 체계**로 발전합니다.