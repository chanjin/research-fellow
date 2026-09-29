# AJD Workflow DSL v0.1

## 1. 목적

AJD Workflow DSL은 AJD에 정의된 직무 책임을 **사람이 읽고 검토할 수 있는 실행 명세**로 표현한다.

DSL은 다음을 표현한다.

- 왜 이 workflow가 존재하는가
- 어떤 현상이나 요청으로 시작하는가
- 어떤 업무 capability를 어떤 순서와 조건으로 수행하는가
- 누가 판단하거나 승인할 권한을 갖는가
- 어떤 기억을 읽고 갱신하는가
- 어떤 shared phenomenon을 외부로 내보내는가
- 직무적으로 언제 완료되었다고 볼 수 있는가

DSL은 다음을 표현하지 않는다.

- Python 함수명과 module path
- prompt template 파일명
- JSON parser/validator 구현
- LLM provider/API 호출
- retry/backoff 횟수
- SQL/table/repository 구현
- logging/trace 저장 구현

핵심 원칙:

> DSL은 직무 수행의 의미와 제어 흐름을 명세하고, capability의 구현 방법은 DSL 밖에 둔다.

---

## 2. 계층

```text
APF
  World / Machine boundary
  Shared Phenomena
        ↓
AJD
  Job / Responsibility / Authority / Memory
        ↓
AJD Workflow DSL
  Trigger / Steps / Decisions / Interactions / Phenomena / Completion
        ↓
Capability Binding
        ↓
Python / LLM / Tool / Storage / Runtime
```

DSL의 step은 Python 함수가 아니라 **Job Capability**를 가리킨다.

---

## 3. 언어 구성요소

### 3.1 Workflow

Workflow는 하나의 AJD 책임을 수행하는 실행 가능한 직무 절차이다.

필수 요소:

- `id`
- `agent`
- `purpose`
- `trigger`
- `steps`
- `completion`

선택 요소:

- `version`
- `realizes`
- `inputs`
- `memory`
- `outputs`
- `invariants`

### 3.2 Trigger

`trigger`는 workflow를 시작시키는 외부 shared phenomenon 또는 업무 조건이다.

`receives`는 별도 언어 요소로 두지 않는다. 시작을 유발하는 수신 phenomenon은 `trigger`로 표현한다.

```yaml
trigger:
  phenomenon: knowledge_update
  from: m1
  when: unreviewed
```

### 3.3 Step

`step`은 workflow에서 사람이 의미를 부여할 수 있는 최소 업무 단위이다.

모든 step은 다음 공통 속성을 가질 수 있다.

- `id`
- `kind`
- `capability` 또는 `workflow`
- `actor`
- `when`
- `foreach`
- `uses`
- `produces`
- `memory`
- `authority`
- `emits`
- `description`

Step은 구현 기술이 아니라 **업무 의미**에 따라 네 종류로 나눈다.

#### action

결정적/비결정적 여부와 무관하게 직무상의 작업을 수행한다.

```yaml
- id: formulate_research_questions
  kind: action
  capability: formulate_research_questions
  actor: m2
```

#### decision

여러 가능한 업무 경로 중 하나를 선택하거나 우선순위를 정한다.

```yaml
- id: prioritize_research_questions
  kind: decision
  capability: prioritize_research_questions
  actor: m2
  authority: m2
```

`judgment`는 별도 키워드로 두지 않고 `decision`의 의미에 포함한다.

#### interaction

Researcher나 다른 외부 행위자의 선택·승인·피드백이 업무 진행에 필요한 단계이다.

```yaml
- id: select_reading_questions
  kind: interaction
  actor: researcher
  authority: researcher
```

`human_review`는 별도 언어 요소로 두지 않고 `interaction + authority`로 표현한다.

#### workflow

하위 workflow를 호출하거나 다른 Agent에게 위임된 workflow를 수행한다.

```yaml
- id: conduct_literature_reviews
  kind: workflow
  workflow: m1_auto_literature_review
  actor: m1
  foreach: dispatched_intents
```

`subworkflow`는 별도 키워드로 두지 않고 `kind: workflow`로 표현한다.

---

## 4. Control Flow

### 4.1 when

간단한 조건부 실행은 `when`으로 표현한다.

```yaml
when: actionable_questions_exist
```

### 4.2 foreach

업무 객체 집합에 동일한 step/workflow를 적용할 때 사용한다.

```yaml
foreach: dispatched_intents
```

### 4.3 branch

v0.1에서는 `branch`를 독립 키워드로 두지 않는다.

Decision 결과를 이름 있는 산출물로 만들고 후속 step의 `when`으로 흐름을 표현한다.

명시적 branch가 여러 workflow에서 반복적으로 필요하다는 근거가 쌓이면 v0.2에서 추가한다.

---

## 5. Data와 Memory

### 5.1 inputs / uses / produces

- `inputs`: workflow 시작 시 외부에서 주어지는 업무 정보
- `uses`: 해당 step이 소비하는 workflow-local 정보
- `produces`: 해당 step이 만들어 다음 step에 전달하는 workflow-local 정보

`reads/writes`는 이 세 개와 의미가 중복되므로 v0.1에서 제거한다.

### 5.2 memory

Memory는 workflow-local data와 구분되는 **지속적 직무 기억**이다.

```yaml
memory:
  reads:
    - knowledge_cards
    - existing_research_questions
  writes:
    - research_questions
```

Workflow 수준의 memory는 이 workflow가 사용하는 기억의 계약을 나타낸다.
Step 수준에 `memory`를 둘 수 있으며, 더 구체적인 read/write 책임을 나타낸다.

SQLite, table, repository 등 저장 구현은 DSL에 표현하지 않는다.

---

## 6. Authority

`authority`는 **그 판단이나 상호작용의 최종 결정권자**를 뜻한다.

가능한 값은 Agent/Actor 식별자이며 특정 기술(`llm`, `python`)이 아니다.

예:

```yaml
authority: m2
```

```yaml
authority: researcher
```

LLM은 capability의 구현 수단일 수 있으나 DSL의 authority가 아니다.

---

## 7. Shared Phenomenon

### 7.1 emits

Agent 경계를 넘어 관찰 가능한 결과를 발생시키는 경우 `emits`로 표현한다.

한 step은 0개 이상의 phenomenon을 발생시킬 수 있으므로 `emits`의 cardinality는 `0..N`이다.

```yaml
emits:
  - phenomenon: advice_report
    to: [researcher, m2]
  - phenomenon: knowledge_update
    to: [m2]
```

`outputs`와 `emits`는 구분한다.

- `outputs`: workflow 호출자가 사용할 수 있는 내부/외부 결과값
- `emits`: APF에서 Agent 경계를 넘어 발생한 shared phenomenon

---

## 8. Completion

`completion`은 runtime 실행 종료가 아니라 **직무적 완료 조건**이다.

```yaml
completion:
  when:
    - selected_questions_are_delegated_or_resolved
    - consumed_knowledge_updates_are_recorded
```

`run.status = completed`, retry count 등은 runtime concern이며 DSL completion과 다르다.

---

## 9. Invariants

`invariants`는 workflow가 성공적으로 수행되었을 때 항상 만족해야 하는 업무적 보장이다.

```yaml
invariants:
  - every_generated_rq_is_traceable_to_source_card
  - every_dispatched_intent_is_traceable_to_selected_rq
```

JSON schema validation, parser 성공 여부 같은 기술적 validation은 포함하지 않는다.

---

## 10. Runtime 밖으로 내리는 요소

다음은 DSL 키워드가 아니다.

```text
code / llm / tool                # 구현 방식
handler                           # Python binding
prompt / template                 # capability 구현
parser / validator                # 실행 메커니즘
retry / backoff                   # 기술적 복구
SQL / repository / table          # persistence 구현
run status / trace storage        # execution observation
provider exception                # infrastructure
```

Capability registry가 DSL의 semantic capability name을 구현에 연결한다.

```text
DSL capability
    formulate_research_questions
             ↓
Capability Binding
             ↓
Python/LLM implementation
```

---

## 11. v0.1 Core Grammar

```yaml
dsl: ajd-workflow/v0.1
id: <workflow-id>
version: <version>
agent: <agent-id>
purpose: <job purpose>
realizes: [<AJD responsibility ids>]

trigger:
  phenomenon: <phenomenon-id>
  from: <actor-id>
  when: <business condition>

inputs:
  <name>:
    source: <business source>
    description: <meaning>

memory:
  reads: [<memory-id>]
  writes: [<memory-id>]

steps:
  - id: <step-id>
    kind: action | decision | interaction | workflow
    capability: <capability-id>        # action/decision/interaction
    workflow: <workflow-id>            # workflow kind only
    actor: <actor-id>
    authority: <actor-id>              # decision/interaction when relevant
    when: <business condition>
    foreach: <collection>
    uses: [<data-name>]
    produces: [<data-name>]
    memory:
      reads: [<memory-id>]
      writes: [<memory-id>]
    emits:
      - phenomenon: <phenomenon-id>
        to: [<actor-id>]
    description: <business semantics>

outputs: [<workflow output>]

completion:
  when: [<business completion condition>]

invariants:
  - <business invariant>
```

---

## 12. 중복 제거 결정

| 기존 후보 | v0.1 결정 | 이유 |
|---|---|---|
| `responsibility` / `purpose` | `purpose` + optional `realizes` | 책임 ID와 자연어 목적 분리 |
| `trigger` / `receives` | `trigger` | 시작 수신 의미 중복 제거 |
| `reads/writes` / `inputs/outputs` | `inputs`, `uses`, `produces`, `outputs` | workflow-local dataflow 명확화 |
| `decision` / `judgment` | `kind: decision` | 동일한 업무 의미 |
| `authority` / `human_review` | `authority` + `kind: interaction` | 사람만 특별 취급하지 않음 |
| `subworkflow` / `workflow` | `kind: workflow` | step 계층 안으로 통합 |
| `branch` / `when` | v0.1은 `when` | 현재 사례에서는 충분 |
| `state` / `completion` | `completion`만 Core | runtime state와 업무 완료 혼동 방지 |
| `evaluation` / `invariants` | `invariants` | 업무 보장 의미를 명확히 함 |
| `emits` / `outputs` | 둘 다 유지 | shared phenomenon과 반환값의 의미가 다름 |
| `memory` / 일반 dataflow | 둘 다 유지 | 지속 기억과 workflow-local data가 다름 |

---

## 13. 향후 v0.2 후보

현재는 Core에 넣지 않고 사용 사례가 반복될 때 승격한다.

- `branch`: 명시적 exclusive/multi-way branch
- `exception`: 업무적 예외와 대체 workflow
- `evidence`: 증거 충족 조건을 1급 개념으로 표현
- `provenance`: lineage/grounding을 구조적으로 표현
- `policy`: 여러 workflow가 공유하는 업무 정책
- `deadline` / `schedule`: 지속 업무의 시간 제약

