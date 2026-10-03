좋습니다. 이제부터는 **기능 추가보다 M1 Freeze + 구조 정리**를 우선하는 게 맞습니다.

현재 누적 코드를 최신 패치까지 재구성해서 1차 usage audit을 돌려봤는데, 생각보다 리팩토링 필요성이 분명합니다.

현재 규모는 대략:

- Python: **155개**
- Jinja prompt: **54개**
- YAML: **94개**
- `app.py`: **9,538 lines**
- `storage.py`: **2,958 lines**
- `auto_literature.py`: **878 lines**
- `ontology_workflow.py`: **437 lines**

특히 `app.py` 안에는 지금 사용하는 직무 중심 UI와 예전 Developer/Legacy UI가 함께 들어 있고, Ontology도 현재의 **batch Type Graph 방식**과 예전의 **카드별 Type/Relation 방식**이 같이 남아 있습니다.

1차로 보면 구조를 다음 네 등급으로 나누는 게 좋습니다.

| 분류 | 의미 | 예 |
|---|---|---|
| **Canonical** | 현재 실제 직무 수행 경로 | Research → Literature → Paper Review → KC → Ontology |
| **Transitional** | 아직 호환 때문에 남아 있지만 제거 예정 | 기존 ontology single-card parser/prompt |
| **Legacy** | Developer mode에서만 사용하는 옛 UI/워크플로우 | 기존 M1/M2 기능 화면 |
| **Dead candidate** | 현재 코드 어디에서도 호출되지 않음 | 일부 compatibility facade / 오래된 workflow |

실제로 몇 가지가 바로 확인됐습니다.

예를 들어 M1 YAML에서 현재 runtime이 명시적으로 사용하는 중심 흐름은 `m1_auto_literature_review`인데, 다음 workflow들은 Python runtime에서 직접 사용되지 않고 YAML끼리만 연결되어 있습니다.

```text
m1_knowledge_acquisition
m1_paper_reading_summary
m1_evidence_investigation
m1_knowledge_curation
```

즉 이들은 현재 구현의 실제 workflow라기보다 **이전 설계 흔적 또는 아직 runtime binding이 안 된 설계 자산**에 가깝습니다.

Ontology도 같은 상태입니다.

현재 canonical:

```text
Knowledge Cards
    ↓
Ontology Graph Attention 1개
    ↓
assignments
new_types
type_relations
card_relations
    ↓
Ontology Change Review
    ↓
승인
```

그런데 코드에는 아직:

```text
type_suggestion_prompt()
relation_suggestion_prompt()
parse_type_suggestions()
parse_relation_suggestions()
knowledge_relation_suggestion
enqueue_type_relation_work()
```

같은 **카드별 Ontology 경로**가 남아 있습니다.

또 `app.py`에서 일반 사용자는 실제로:

```text
Operating Desk
 ├ Research
 ├ Attention
 ├ Knowledge
 └ Activity
```

만 쓰는데, 같은 파일 안에 예전:

```text
home()
m1_screen()
m2_screen()
paper coauthor UI
legacy knowledge UI
developer UI
```

까지 모두 들어 있습니다.

그래서 저는 리팩토링을 다음 순서로 진행하는 게 좋다고 봅니다.

### R1 — M1 Canonical Path Freeze

먼저 현재 실제 사용 경로를 고정합니다.

```text
Research Question
 → Initial Literature Round
 → External LLM Input
 → Candidate Review
 → Paper Review
 → Knowledge Card Approval
 → Additional Literature
 → Ontology Graph
 → Ontology Approval
```

각 단계에 대해:

```text
UI
→ Application Service
→ Workflow / Research Task
→ Capability
→ Prompt
→ Parser
→ Storage
```

를 1:1로 맵핑합니다.

이게 앞으로 “살려야 할 코드”의 기준입니다.

---

### R2 — Ontology legacy 제거

가장 먼저 실제 코드를 줄일 수 있는 영역입니다.

최종적으로는:

```text
ontology_workflow.py
ontology_curation_prompts.py
ontology_curation_parsers.py
```

에서 **batch graph 방식만 남깁니다.**

다음은 Legacy로 내려보냅니다.

```text
single-card type suggestion
single-card type relation suggestion
knowledge_relation_suggestion task
enqueue_type_relation_work()
```

Developer UI에서만 필요하다면 별도 legacy module로 옮깁니다.

---

### R3 — `app.py` 분리

이게 가장 효과가 큽니다.

현재 9,500줄짜리 `app.py`를 최종적으로:

```text
app.py
  └ bootstrap / workspace / provider 설정

ui/
  shell.py
  research.py
  attention.py
  knowledge.py
  activity.py

legacy/
  legacy_app.py
  legacy_m1.py
  legacy_m2.py
  legacy_paper.py
```

정도로 나누는 게 좋습니다.

일반 실행 시에는 Legacy module 자체를 import하지 않게 합니다.

그러면 앞으로 M1/M2 개발하면서 예전 UI 코드를 건드릴 일이 거의 없어집니다.

---

### R4 — Prompt 정리

현재 가장 혼란스러운 부분 중 하나입니다.

지금은:

```text
Python inline prompt
Jinja prompt
YAML capability → prompt
Legacy prompt
```

가 섞여 있습니다.

예를 들어 현재 RQ-conditioned Paper Review는 일부가 `auto_literature.py` 안에 inline string으로 남아 있습니다.

최종 원칙은:

```text
LLM을 호출하는 의미 있는 capability
        ↓
*.j2 prompt
        ↓
parser
```

로 통일하는 게 좋습니다.

Python에는 prompt 문장을 넣지 않습니다.

---

### R5 — Workflow YAML 정리

현재 YAML 94개가 있다고 해서 모두 실제 workflow인 것은 아닙니다.

다음처럼 나눕니다.

```text
Active Workflow
Design-only Workflow
Legacy Workflow
Unused Workflow
```

그리고 Active가 아닌 것은 runtime directory 밖으로 옮깁니다.

예:

```text
workflows/
  active/
  experimental/
  legacy/
```

혹은 더 단순하게 legacy 삭제 후 Git history에 맡겨도 됩니다.

---

### R6 — Storage 정리

`storage.py` 2,958줄은 기능이 계속 누적된 결과입니다.

여기도 도메인 기준으로:

```text
ResearchRepository
PaperRepository
KnowledgeRepository
OntologyRepository
AttentionRepository
RuntimeRepository
```

로 나누는 게 맞습니다.

다만 **DB schema를 지금 바꾸지는 않는 게 좋습니다.**

우선 Python repository interface만 분리하고 SQLite schema는 유지합니다.

Portable Workspace도 이 repository 위에서 export/import하도록 하면 됩니다.

---

### R7 — Workspace persistence 정리

마지막에 기존:

```text
workspace_sync.py
server/local sync
```

를 걷어내고 최근 만든:

```text
workspace.json
pdf/
optional db
```

구조로 통일합니다.

이 부분은 집/학교 데이터 merge 구현까지 확인한 뒤 제거하는 게 안전합니다.

---

제가 보기엔 첫 실제 코드 리팩토링은 **Ontology부터 시작**하는 게 가장 좋습니다.

이유는 현재 M1 마지막 단계이고, 최근 수정 과정에서 legacy와 canonical 경로가 가장 명확하게 드러났기 때문입니다.

따라서 다음 작업은:

```text
R1. M1 Canonical Path Map 확정
        ↓
R2. Ontology legacy 경로 제거
        ↓
Targeted regression test
```

로 진행하겠습니다.

이 단계가 끝나면 **삭제된 파일/함수, 남은 canonical 경로, Transitional 항목**을 짧게 정리하고 바로 `app.py` 분리로 넘어가면 됩니다.