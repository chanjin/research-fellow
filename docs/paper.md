
# Job-First Engineering of Persistent AI Agents: From Specification to Dependable Job Performance

## 1. Introduction

> 현재 LLM의 발전은 AI Agent 기술
> 
Recent advances in large language models (LLMs) have substantially expanded the capabilities of AI agents. Agents can reason over increasingly complex problems, invoke external tools, maintain memory, coordinate with other agents, and execute long sequences of actions. Consequently, much of the current progress in agentic AI is framed around an increasingly ambitious question: How complex a problem can an AI agent successfully solve?

> 얼마나 복잡하고 어려운 문제까지 해결할 수 있는가?
> 
This emphasis is natural from the perspective of frontier model development. For developers of foundation models, demonstrating that a new model can solve problems that previous models could not is an important measure of technological progress. Agent workflows extend these capabilities by orchestrating models, tools, and reusable skills to accomplish tasks that cannot be completed through a single model invocation. Planning, tool use, reflection, memory, and multi-agent coordination have therefore become important mechanisms for achieving successful task completion.


> 산업 현장에서의 문제는 다르다. 
> 
The engineering problem encountered in industrial and professional environments, however, is often different. Many industrial jobs do not require an agent to repeatedly solve frontier-level problems. Instead, they require a bounded responsibility to be performed repeatedly, consistently, and according to operational requirements. An individual task may be relatively simple, yet unreliable execution, inappropriate judgment, or failure to respond to changes in the operating environment can have significant consequences.

> 타스크 완료 보다 지속적 직무 성능이 중요
> 
This distinction motivates a shift from task completion to persistent job performance. A task normally terminates when a specified outcome is achieved. A job persists: new events arrive, knowledge accumulates, exceptions occur, organizational rules change, and the same responsibility must continue to be fulfilled. For industrial agents, the relevant question is therefore not only whether an agent can successfully solve a difficult task once, but whether it can continue to satisfy the requirements of its assigned job over time.

### 1.1 From Technology-First to Job-First Agent Engineering

Two common perspectives on AI agents illustrate the need for this distinction.

The first treats an agent primarily as a tool for solving problems. New models, skills, and tools are assembled into increasingly capable workflows, after which suitable problems and use cases are identified. The development direction can be summarized as:

Models and Skills → Workflows → Problems → Applications.

The second perspective extrapolates improvements in AI capability toward comprehensive automation. As models become increasingly capable, they are expected to perform progressively larger portions of human work and potentially entire organizational jobs.

Although these perspectives differ substantially in scale, both begin with technology. One asks what problems can be solved with available capabilities; the other asks how much human work future capabilities may replace.

This study takes the opposite direction:

We do not start from what an LLM can do. We start from what a job requires.

The relevant engineering unit is therefore neither an individual skill nor an entire occupation, but a bounded job responsibility that can be explicitly specified, repeatedly performed, evaluated, and improved. This reframes the engineering question as:

What does it take for an AI agent to become dependable at a bounded job?

This job-first perspective also changes the meaning of modularity. A system may appear modular because its prompts, tools, skills, and workflows are separated into components. Such implementation modularity does not necessarily make the system modular with respect to change. When a regulation, organizational rule, information source, or operating procedure changes, engineers may still need to inspect multiple agents, prompts, and workflows to determine the consequences.

A job boundary provides a different form of modularity. If an agent has an explicit responsibility, boundary, inputs, outputs, interactions, and performance criteria, environmental changes can be analyzed in relation to the job responsibilities they affect.

In this sense:

Skills modularize what an agent can do; jobs modularize what an agent is responsible for.

A job boundary is therefore not only a guardrail for agent behavior but also an engineering boundary for reasoning about change.

### 1.2 From a Smart New Employee to a Trusted Colleague

A useful analogy is a highly intelligent employee who has just graduated from university and entered an unfamiliar workplace. High intelligence and extensive general knowledge do not make the employee immediately dependable in a particular job. The employee must understand the workplace, learn what is expected, perform under supervision, and gradually earn trust.

Three problems are particularly important.

The first is the boundary problem. A new employee must know what falls within the responsibility of the job, what actions are permitted, and when a decision should be escalated. Without clear boundaries, even a capable employee may act beyond the authority or responsibility of the assigned role.

The second is the grounding problem. General knowledge is insufficient when a job depends on local information, organizational history, domain-specific knowledge, current operating conditions, and prior decisions. Without sufficient context, a new employee may fill gaps using plausible assumptions based on general knowledge. LLMs exhibit a similar tendency. Their ability to infer plausible answers is valuable for open-ended problem solving but may become a liability when a job requires judgments grounded in specific evidence.

The third is the feedback problem. New employees improve because their work is observed, evaluated, and corrected. Without feedback, an incorrect interpretation or procedure may persist and become repeated practice. Persistent agents similarly require mechanisms through which execution outcomes can be evaluated against job requirements and incorporated into subsequent performance.

These three problems can be summarized as follows:

Without boundaries, an agent may overstep. Without grounding, it may guess. Without feedback, its mistakes may persist and compound.

A capable model therefore does not automatically constitute a dependable worker. General capability must be situated within a bounded job, grounded in the world in which that job is performed, and continually corrected through experience. Trust emerges through repeated satisfactory performance rather than from model capability alone.

This suggests a progression from:

Capability → Job-Specific Competence → Dependability → Trust.

### 1.3 Persistent Jobs in a Changing World

Persistence introduces another engineering requirement: the world surrounding a job does not remain fixed. Regulations change, organizational policies are revised, new information becomes available, tools and data sources are replaced, and the roles of collaborating humans or agents evolve.

A workflow developed for the current context may therefore become inappropriate even if its implementation remains technically correct. An agent that executes yesterday's procedure perfectly may fail today's requirement.

Persistent agents must consequently be designed not only for reliable execution but also for recoverability and evolution. Some changes may be addressed by updating contextual information or domain knowledge. Others may require changes to procedures or workflows. More fundamental changes may alter the responsibility or boundary of the job itself and require explicit engineering intervention.

We therefore argue that the central engineering problem for industrial AI agents is not simply how to orchestrate increasingly capable models to solve complex tasks, but how to specify, realize, evaluate, and evolve bounded jobs under changing operational requirements.

To address this problem, we propose a specification-based, job-first development approach that progressively transforms an organizational mission into executable agent behavior.


## 2. Designing and Specifying Persistent Job Agents

직무 에이전트 개발은 LLM의 능력에서 출발하여 적용 가능한 업무를 찾는 과정이 아니다. 먼저 세상에서 지속적으로 만족시켜야 할 요구사항과 이를 책임질 직무를 정의하고, 그 직무를 수행할 Agent Machine을 설계해야 한다. LLM은 Agent 자체가 아니라 Agent Machine 내부에서 해석과 추론을 담당하는 구성요소이다. Agent Machine에는 이 밖에도 운영 맥락 구성, 도구 실행, 결과 검증, 상태 관리 및 메모리 갱신을 위한 메커니즘이 포함된다.

우리가 대상으로 하는 것은 일회성 목표를 달성하면 종료되는 project agent가 아니라, 주어진 직무를 반복적으로 수행하는 persistent job agent이다. 예를 들어 장비 이상을 분석하여 보고서를 생성하는 것은 하나의 workflow가 될 수 있지만, 장비의 신뢰성과 운영 안전을 지속적으로 유지하는 것은 직무이다. 마찬가지로 특정 주제의 논문을 검색하고 요약하는 것은 workflow인 반면, 새로운 연구를 지속적으로 탐색하여 지식체계를 유지하고 연구자에게 근거 있는 조언을 제공하는 것은 직무이다. Persistent job agent는 반복된 수행에서 경험을 축적하고, 그 경험을 바탕으로 개선되며, 운영 환경과 지식의 변화에도 대응해야 한다.

이러한 Agent에게 직무를 맡긴다는 것은 일정한 책임을 부여한다는 의미이다. 따라서 핵심 질문은 LLM이 무엇을 생성할 수 있는가가 아니라, 조직이 Agent에게 무엇을 신뢰하고 맡길 수 있는가이다. Agent의 잘못된 행동 역시 단순히 “LLM이 틀렸다”고 설명해서는 안 된다. 불완전한 직무 요구사항, 잘못 설정된 권한 경계, 부족한 운영 맥락, 부적절한 지식, 누락된 상호작용 또는 결과를 확인할 피드백의 부재에서 원인을 찾아야 한다. 이를 위해 본 방법론은 APF, AJD, Workflow라는 세 가지 추상화 수준을 사용한다. APF는 직무가 정의되는 세상과 상호작용을 모델링하고, AJD는 그 직무를 맡은 Agent Machine을 명세하며, Workflow는 직무가 실제로 수행되는 절차를 구현한다.

### 2.1 Modeling the World and Shared Phenomena with APF

직무는 Agent 내부가 아니라 세상에서 정의되므로, Agent 설계는 World의 모델링에서 시작해야 한다. 장비관리 직무의 의미는 설비의 상태, 생산환경, 작업자, 정비조직과 안전규정에 의해 결정된다. 연구위원 직무 역시 연구자, 외부 자문 요청자, 연구문헌과 조직의 지식체계 속에서 정의된다. LLM에는 이러한 세상에 대한 책임이나 권한이 본래 주어져 있지 않다. Agent Machine은 세상이 요구하는 결과를 실현하기 위해 LLM을 비롯한 내부 구성요소를 조직하고 World와 연결한다.

Agentic Problem Frames(APF)는 Requirement, World, Machine을 구분하여 이러한 관계를 모델링한다. Requirement는 World에서 달성하거나 지속적으로 유지해야 할 상태를 나타낸다. World는 Agent가 상호작용하는 사람, 설비, 문서, 정보시스템, 조직규칙과 다른 Agent를 포함한다. Machine은 직무 책임을 부여받은 Agent이며, LLM은 이 Machine 안에 포함된다. 이러한 구분을 통해 LLM의 내부 능력보다 Agent Machine과 World 사이의 경계를 먼저 명확히 할 수 있다.

APF에서 특히 중요한 것은 shared phenomena이다. 직무는 Agent가 내부적으로 답변을 생성하는 것만으로 수행되지 않는다. Agent는 World로부터 이벤트와 요청을 받고, 문서와 시스템을 조회하며, 사람이나 다른 시스템에 판단과 행동을 요청하고, 그 결과를 다시 관찰해야 한다. 이상 신호의 수신, 정비 이력 조회, 점검 요청, 조치 결과 확인은 장비관리 Agent와 World 사이의 shared phenomena이다. 연구질문의 수신, 논문 검색, 지식 검토 요청, 자문 결과 전달 역시 연구위원 Agent의 shared phenomena이다. 이들은 단순한 데이터 인터페이스가 아니라 Agent가 세상에 대한 책임을 실현하는 구체적인 접점이다.

Shared phenomenon의 명세에는 교환되는 데이터뿐 아니라 현상의 발생 주체, 관찰 가능성, 제어 권한, 의미 해석과 결과 확인 방법이 포함되어야 한다. 이를 통해 Agent가 무엇을 알 수 있는지, 무엇에 영향을 줄 수 있는지, 그리고 실행 결과를 어떻게 확인할 수 있는지를 구분한다. 필요한 정보가 관찰되지 않거나 행동 결과를 확인할 피드백이 없다면, 그 직무는 아직 Agent에게 책임 있게 맡길 수 없다.

명시적인 상호작용 모델은 변화 관리도 국소화한다. 새로운 센서나 정보시스템이 도입되면 해당 World domain과 shared phenomena를 수정하고, 관련 맥락 구성과 도구 연결을 변경할 수 있다. 승인 정책이 바뀌면 요청·승인·실행에 해당하는 shared phenomena와 권한 규칙을 갱신할 수 있다. World 전체의 변화를 하나의 거대한 프롬프트에 반영하는 대신, 변화가 발생한 도메인과 상호작용을 중심으로 Agent 명세와 구현을 조정하는 것이다.

### 2.2 Specifying Agent Machines with AJD

APF에서 World와 shared phenomena를 파악하면, 상호작용과 기대 결과를 중심으로 책임의 집합이 드러난다. 이 책임들을 응집도 있게 묶어 Agent의 경계를 결정하고, 각 Agent Machine이 맡을 직무를 Agent Job Description(AJD)으로 구체화한다. APF가 World를 모델링한다면, AJD는 그 World를 고려하여 정의된 직무를 Machine 관점에서 종합한다.

Agent의 경계는 LLM 기능이나 개별 workflow가 아니라 지속적으로 책임질 결과를 기준으로 정해야 한다. 이는 소프트웨어 설계의 Single Responsibility Principle과 유사하다. 하나의 Agent Machine은 독립적으로 책임을 부여하고, 권한을 관리하며, 결과를 평가할 수 있는 일관된 직무를 담당해야 한다. 다만 SRP는 논문 검색, 주장 추출, 보고서 작성과 같은 모든 기능을 각각 별도의 Agent로 분리하라는 뜻은 아니다. 여러 workflow가 동일한 목적, 판단 기준과 평가 방법을 공유한다면 하나의 직무 안에 포함될 수 있다.

연구위원 사례에서 APF의 shared phenomena를 분석하면 Curator와 Advisor라는 두 개의 책임 집합이 나타난다. Curator는 논문을 탐색하고 주장을 검토하여 내부 지식체계의 정합성과 품질을 유지한다. 핵심 질문은 새로 확보한 정보가 충분한 근거를 가지며 기존 지식과 일관되게 연결될 수 있는가이다. Advisor는 연구자나 외부 자문 요청자가 무엇을 원하는지를 해석하고, 승인된 내부 지식을 활용하여 답변이나 보고서를 구성한다. 핵심 질문은 요청자가 필요로 하는 판단이 무엇이며, 내부 지식을 어떻게 해당 맥락에 맞게 적용할 것인가이다.

두 직무는 같은 지식을 사용하지만 지식을 판단하는 기준이 다르다. 이를 하나의 Agent에 결합하면 검증되지 않은 탐색 결과가 확정적인 자문에 사용되거나, 특정 요청자의 일시적 관점이 일반 지식처럼 저장되는 context contamination이 발생할 수 있다. 따라서 Curator와 Advisor를 별도의 Agent Machine으로 정의하고 각각의 AJD를 작성한다. Curator가 검증된 지식을 게시하거나 Advisor가 지식 부족을 발견해 추가 탐색을 요청하는 과정은 두 Agent 사이의 새로운 shared phenomena로 APF에 반영한다.

AJD는 다음 여섯 요소를 통해 Agent Machine의 직무를 명세한다.

| AJD 요소                      | 명세 내용                                        |
| --------------------------- | -------------------------------------------- |
| Mission                     | Agent가 지속적으로 책임지는 포괄적인 요구사항과 World-side 결과   |
| Scope and Authority         | 직무의 범위, 자율적으로 판단·행동할 수 있는 영역과 권한 밖의 영역       |
| Workplace                   | 직무 수행에 사용할 수 있는 도구, 시스템과 실행 채널               |
| Operational Context         | 트리거 발생 시 수집·선택하여 이번 실행에 제공해야 하는 상황 정보와 조건    |
| Memory and Knowledge Assets | Agent 내부에 명시적으로 유지되는 World에 대한 이해, 경험과 절차 지식 |
| Evaluation Method           | Agent의 산출물과 그에 따른 World의 변화를 확인하는 방법         |

Workplace는 APF의 World domain 전체를 반복하여 기술하지 않는다. 이는 Agent Machine이 World에 접근하고 영향을 주기 위해 사용할 수 있는 도구와 실행 환경을 뜻한다. 반면 Operational Context는 해당 도구와 정보원으로부터 특정 실행을 위해 선택된 현재 상태와 판단 정보를 의미한다. 예를 들어 논문 데이터베이스는 World의 정보 자원이고, 검색 기능은 Workplace의 도구이며, 검색을 통해 선별되어 현재 연구질문의 판단에 사용되는 논문과 주장은 Operational Context의 일부가 된다.

Memory and Knowledge Assets는 Agent Machine이 명시적으로 보유하고 관리하는 World에 대한 이해이다. 이는 LLM 파라미터에 암묵적으로 포함된 일반 지식과 구별된다. 지식카드, 온톨로지, 사례 기록과 검증된 절차처럼 외화된 형태로 관리되어야 검토, 수정, 이전과 재사용이 가능하다. 이 여섯 요소를 통해 AJD는 Agent가 왜 존재하는지, 무엇을 책임지는지, 어떤 경계와 수단을 가지는지, 무엇을 근거로 판단하는지, 결과를 어떻게 확인하며 경험을 어떻게 유지하는지를 포괄한다.

AJD는 Agent가 무엇을 해야 하는가와 어떤 조건 아래 수행해야 하는가를 정의한다. 구체적인 task 순서, 프롬프트와 제어 로직은 다음 단계인 Workflow에서 설계한다.

### 2.3 From AJD to Executable Workflows

Workflow는 AJD에 정의된 직무를 수행하기 위해 여러 task를 연결한 절차적 시퀀스이다. 하나의 Agent는 트리거와 목적에 따라 여러 workflow를 수행할 수 있다. Curator의 문헌 탐색, 논문 분석과 지식 등록은 서로 다른 workflow일 수 있지만, 모두 검증된 지식체계를 구축하고 유지한다는 하나의 직무를 실현한다.

각 task는 결정론적 소프트웨어, 확률론적 LLM 추론 또는 인간 판단 중 적합한 수행 주체에 할당한다. 상태 전이, 데이터 변환, 정책 검사, 도구 호출과 정형적 검증은 가능한 한 일반 소프트웨어로 구현한다. 의미 해석, 주장 비교, 종합과 맥락 의존적 판단에는 LLM을 사용한다. 높은 위험을 가지거나 Agent의 권한을 벗어나는 결정은 사람에게 이관한다. Agent의 모든 실행을 비결정적인 LLM에 맡기는 것이 아니라, 불확실한 추론이 필요한 지점을 식별하여 제한적으로 배치하는 것이다.

이를 bifurcated architecture로 구현할 수 있다. LLM은 결정론적인 execution envelope 안에서 실행된다. Envelope는 실행에 필요한 맥락을 준비하고, 프롬프트를 구성하며, LLM과 도구를 호출하고, 결과를 파싱·검증하며, 실패 시 재시도하거나 사람에게 이관한다. 또한 승인된 결과를 저장하고 다음 workflow 상태로 전환한다. LLM은 이 경계 안에서 요구되는 확률적 추론을 수행한다.

구현에서는 YAML로 workflow, task, 입력·출력과 검증 조건을 선언하고, Jinja 템플릿으로 실행 시점의 맥락을 LLM 입력으로 렌더링할 수 있다. Python과 같은 일반 프로그래밍 언어는 상태 관리, 데이터 변환, 도구 호출, 검증, 예외 처리와 workflow 제어를 담당한다. 다만 파일 형식이 결정론적 부분과 확률론적 부분을 구분하는 것은 아니다. 중요한 것은 어떤 판단을 LLM에 위임하며, 어떤 조건과 결과를 소프트웨어가 보장하는지를 명시하는 것이다.

Guardrails 역시 구현 단계에서 임의로 추가되는 제약이 아니다. AJD의 Scope and Authority를 workflow의 guard condition, 도구 접근권한과 runtime check로 변환한 것이다. 예를 들어 Advisor가 승인된 지식만 사용하고 새로운 지식을 직접 확정할 권한은 없다면, workflow는 승인된 지식만 검색하도록 제한하고 지식이 부족한 경우 Curator에게 탐색을 요청하도록 분기해야 한다. 이러한 추적 관계를 통해 직무 수준의 책임과 권한이 실제 실행에서 유지된다.

### 2.4 Dynamic Specification and the Agent Execution Loop

Persistent job agent는 고정된 명세를 반복 실행하지 않는다. 트리거 이벤트가 발생할 때마다 현재의 Operational Context를 바탕으로 해당 실행에 필요한 동적 명세를 구성한다. 입력 이벤트는 불완전하거나 모호할 수 있으며, 동일한 입력도 현재의 World 상태, 요청자의 의도, 적용되는 규칙과 축적된 지식에 따라 다르게 해석될 수 있다. 환경과 지식자산 역시 지속적으로 변화한다.

따라서 Agent Machine은 트리거를 그대로 LLM에 전달하지 않는다. 현재 World의 상태, 관련 지식, Agent의 권한, 기대 결과와 평가 조건을 결합하여 시점 \(t\)의 동적 명세 \(S_t\)를 구성한다. 프롬프트는 동적 명세 자체가 아니라, 이를 LLM이 실행할 수 있는 형태로 변환한 runtime representation이다. Agent Machine은 이 명세에 따라 LLM을 호출하고 Workplace의 도구를 사용해 World와 상호작용한다.

LLM 실행은 비결정적이고 확률적이므로, 생성된 결과를 그대로 직무의 완료로 간주할 수 없다. 먼저 산출물이 요구된 형식, 근거, 규칙, 권한과 완료 조건을 만족하는지 Machine 내부에서 평가해야 한다. 이어서 도구 사용이나 사람과의 상호작용 이후 실제 World에 기대한 변화가 일어났는지를 callback이나 confirmation을 통해 확인해야 한다. 보고서가 생성되었다는 사실과, 그 보고서가 올바른 근거를 사용하고 적절한 의사결정에 기여했다는 것은 서로 다른 평가 대상이다.

이 과정을 다음과 같이 표현할 수 있다.

$$
\{S_t\}A_t\{K_{t+1}\}
$$

여기서 \(A_t\)는 LLM의 텍스트 생성뿐 아니라 도구 사용과 World와의 상호작용을 포함한 Agent Machine의 실행이다. \(K_{t+1}\)은 실행 로그 자체가 아니라, 실행 결과와 World의 반응을 평가하여 새롭게 확인된 지식이다. 이 지식은 다음 실행의 Operational Context와 동적 명세 \(S_{t+1}\)을 구성하는 데 사용된다.

$$
S_t \rightarrow A_t \rightarrow Evaluation
\rightarrow K_{t+1} \rightarrow S_{t+1}
$$

이러한 반복은 LLM 자체가 자동으로 개선된다는 의미가 아니다. 검증된 수행 경험이 명시적인 지식자산과 맥락 구성 규칙으로 전환되고, 이것이 다음 LLM 실행을 더 정확하게 만든다는 뜻이다. 실패 원인이 맥락 누락에 있다면 Operational Context 구성 규칙을 수정하고, 권한 또는 평가 기준에 있다면 AJD를 갱신한다. World domain이나 shared phenomenon이 변했다면 APF도 함께 변경한다.

### 2.5 Progressive Autonomy through Governed Learning

Agent의 자율성은 통제를 제거하고 LLM에 직무를 방임하는 것이 아니다. 자율성은 명시된 책임 안에서 행동을 선택하고 실행할 수 있도록 부여된 권한이며, 검증된 지식과 경험에 따라 점진적으로 확대되어야 한다.

초기 Agent는 정적인 SOP와 사전에 정의된 workflow를 따라 수행하고 주요 결과를 사람의 검토에 맡길 수 있다. 이 과정에서 개별 실행과 결과는 episodic memory에 축적되고, 여러 사례를 통해 확인된 World 지식과 판단 기준은 semantic memory로 구조화된다. 반복적으로 유효성이 확인된 SOP, workflow와 예외 처리 방식은 procedural memory로 관리된다.

충분한 경험이 축적되면 Agent는 기존 procedural memory를 바탕으로 현재 상황에 적합한 실행 절차나 workflow 변경안을 탐색할 수 있다. Episodic memory는 과거의 성공과 실패 사례를 제공하고, semantic memory는 새로운 절차가 지켜야 할 World 지식과 제약을 제공한다. 그러나 절차의 탐색과 운영 적용은 구분해야 한다. 새롭게 제안된 절차는 기존 요구사항과 권한 경계를 충족하는지 평가하고, 과거 사례 재실행, 시뮬레이션, shadow operation 또는 인간 검토를 거쳐 승인된 후에만 procedural memory와 운영 workflow에 반영한다.

자율성은 정적 SOP의 통제된 실행에서 시작하여, 맥락에 따른 추천, 제한된 범위의 자율 실행, 검증을 전제로 한 절차 재구성으로 발전할 수 있다. 각 단계의 확대는 LLM의 일반적 성능이 아니라 해당 직무에서 축적된 성공과 실패, 권한 준수 및 World-side 결과를 근거로 결정해야 한다. 확대된 권한은 AJD의 Scope and Authority에 반영하고, 새로운 도구나 상호작용이 추가되면 APF의 shared phenomena도 갱신한다.

따라서 산업현장에서 Agent에게 직무를 이관하는 과정은 LLM 기술을 먼저 도입하고 적용 범위를 넓히는 과정이 아니다. 인간의 직무 지식과 SOP를 외화하고, 통제된 실행과 OJT를 통해 경험을 축적하며, 검증된 부분부터 책임과 권한을 단계적으로 이전하는 조직적 학습 과정이다.

APF, AJD와 Workflow는 한 번 작성한 후 고정되는 순차적 산출물이 아니다. APF는 World와 상호작용 경계를, AJD는 책임 있는 Agent Machine을, Workflow는 그 직무의 실행 방식을 서로 다른 추상화 수준에서 명세한다. 실행 과정에서 얻은 증거는 세 명세를 지속적으로 수정한다. 이들의 공진화를 통해 Agent는 일회성 작업 자동화에서 벗어나, 변화하는 World에서 책임 있는 직무를 지속적으로 수행하고 개선하는 persistent job agent로 발전한다.


## 3. Lessons from Building a Persistent Research Fellow Agent

We implemented a research fellow agent to examine what changes when an AI agent is expected to perform a job continuously rather than complete a bounded task. We deliberately chose a knowledge-intensive job as our first implementation case. Industrial jobs such as equipment maintenance or quality management provide compelling examples of persistent agency, but their realization depends heavily on site-specific equipment, sensors, control systems, physical constraints, and safety requirements. These dependencies make it difficult to separate general principles of agent engineering from the particulars of a given industrial environment.

Research work provides a useful intermediate case. Its primary working environment is informational rather than physical, while the job itself remains persistent and open-ended. A research fellow must continuously accumulate knowledge, preserve research context, relate new evidence to prior understanding, respond to questions, and initiate further investigation when existing knowledge is insufficient. This allowed us to study persistence, memory, knowledge growth, and human governance before introducing the additional complexity of physical-world interaction.

The implementation also changed our understanding of what should be engineered around an LLM. We initially regarded dynamic specification largely as a mechanism for constructing better inputs to the model. Through implementation, however, we came to view the agent itself as the persistent machine and the LLM as only one probabilistic reasoning component within it. This distinction led to several lessons.

### 3.1 Start from the Job, Not the Project

A project is bounded by a goal, a time horizon, and a completion condition. “Complete a paper on topic A by December,” for example, defines a finite objective. Once the paper is completed, the project ends.

A job is different. A researcher may conduct a sequence of projects over many years while retaining continuity in expertise, responsibility, research interests, accumulated experience, and ways of framing problems. Individual projects are temporary expressions of that longer-lived professional role.

This distinction became fundamental in our implementation. The research fellow agent was not defined by a current paper, literature-search request, or advisory question. Its mission was to remain knowledgeable in a research domain, support the researcher over time, preserve the history of prior investigations, and use accumulated experience to improve subsequent work.

This changed the order of design. We did not begin with functions such as search, summarization, question answering, or report generation. We began with the job and its enduring mission. Responsibilities, interactions, workflows, memory structures, and tools were then derived from that job.

This distinction also provides a stable basis for adaptation. The job and mission remain relatively stable, while the procedures used to fulfill them may change. A workflow that is effective today may be reorganized or replaced tomorrow without redefining the job itself.

In this sense, persistent agents should be engineered around enduring responsibilities rather than finite goals.

### 3.2 Build an Agent Layer Around the LLM

The implementation led us to distinguish the agent from its underlying LLM more explicitly.

The machine in our formulation is not the LLM alone. It is the complete job-performing agent: its mission, memory, knowledge, procedures, tools, operating policies, and mechanisms for interacting with the surrounding world. The LLM is a reasoning engine operating inside this machine.

The agent layer therefore mediates between a general-purpose probabilistic model and a domain-specific operational world. For each event or request, it determines what the event means in relation to the job, gathers the relevant knowledge and prior experience, constructs sufficient context, invokes the LLM or other tools, and interprets the result in terms of the agent's responsibilities.

Execution is only one part of this process. The agent must also observe what happened after execution, detect changes in the external world that affect its job, evaluate new information, and update what should influence subsequent action.

At a high level, this produces a continuing loop:

**observe → contextualize → execute → evaluate → update**

Persistence therefore does not reside in an individual LLM invocation. It resides in the surrounding agent layer that maintains continuity across invocations.

This distinction also clarified our interpretation of dynamic specification. We initially treated dynamic specification mainly as the dynamically constructed context presented to the LLM. We now regard it more broadly as the evolving specification and state of the agent itself.

Let \(S_t\) represent the agent's specification and internal state at time \(t\), including its mission, accumulated memories, operating knowledge, procedures, and policies. Let \(K_t\) represent the relevant state of the external world as known to the agent, and \(E_t\) a newly observed event. The execution context for an LLM invocation can then be regarded as a function of these elements:

$$
C_t = f(S_t, K_t, E_t)
$$

The important form of persistence is not \(C_t\), which is transient, but the continuity of \(S_t\). Through execution results and human feedback, the agent can evolve from \(S_t\) to \(S_{t+1}\).

This gives the agent a form of operational identity. Two agents may use the same underlying LLM but behave differently because they have accumulated different knowledge, experiences, procedures, and feedback histories.

### 3.3 Externalize Knowledge to Build Expertise and Control

Persistent agents cannot rely solely on knowledge embedded in LLM parameters.

Model-internal knowledge is difficult to inspect and selectively control. It is also difficult to determine whether newly introduced domain knowledge has been incorporated consistently with existing knowledge. Model upgrades further complicate the problem: knowledge critical to a job should not depend entirely on the particular version of the underlying model.

More importantly, much of the knowledge required by a real job does not exist beforehand in a complete form.

A research fellow can retrieve papers, reports, and known findings, but professional knowledge also includes judgments that emerge through work: which evidence proved useful, which claims were later revised, what search strategy worked for a particular class of questions, which assumptions repeatedly failed, and how the researcher interpreted conflicting evidence. Such knowledge is created through the process of performing the job.

We therefore externalized important knowledge rather than attempting to encode it back into model parameters. In the research fellow implementation, this included structured research context, claims and supporting evidence, knowledge cards, accumulated search experience, and later, relationships among knowledge types.

The appropriate representation is domain dependent. A research agent may need to represent claims, evidence, counterarguments, research questions, and source credibility. An equipment-maintenance agent may instead require symptoms, causal hypotheses, maintenance history, interventions, and observed outcomes. The representation should reflect what the job needs to remember and what humans need to inspect.

The purpose of this externalized knowledge is not archival. It is operational.

The agent should not answer a question merely from the generic prior of its LLM. It should answer from the professional knowledge accumulated through previous work. Prior evidence, validated interpretations, and researcher feedback should influence subsequent search, analysis, and judgment.

Externalized knowledge therefore serves two related purposes.

First, it creates specialization. The agent progressively develops responses that reflect its domain and history rather than producing a generic LLM answer.

Second, it creates governability. A human can inspect the basis for a response, identify which knowledge influenced it, correct incorrect interpretations, and revise what should be used in subsequent work.

In this sense, externalized knowledge is not simply a record of past work. It is a control surface for improving future work.

### 3.4 Experience Must Be Converted into Reusable Knowledge

Storing interaction histories alone did not provide this improvement.

Execution logs tell us what happened, but not necessarily what should be learned from it. For a persistent agent, experience must be transformed into representations that can affect future behavior.

This transformation may involve several forms of memory. Semantic memory captures domain knowledge that has become sufficiently stable to reuse. Episodic memory preserves significant prior situations, judgments, and outcomes. Procedural memory captures ways of performing work that have proven effective.

The distinctions are less important than the underlying principle: an experience becomes valuable when it can influence a future decision.

For example, a literature search may initially be a transient workflow execution. If the agent discovers that a particular framing consistently retrieves more relevant work, that lesson can become procedural knowledge. If a researcher repeatedly rejects a particular interpretation of a concept, the corrected interpretation can become semantic knowledge. If an unusual research episode later provides a useful analogy, that episode becomes reusable experience.

This process also means that deploying an agent can itself become a knowledge-creation activity. Rather than assuming that all required domain knowledge must exist before AI deployment, organizations can progressively create structured knowledge while the agent performs its job.

### 3.5 Transfer the Job First, Expand Autonomy Later

Our approach is intentionally conservative at the beginning.

We do not ask the agent to discover from scratch how an existing professional job should be performed. Instead, we first transfer the current job into an explicit form. Existing procedures, domain meanings, accumulated knowledge, interaction patterns, and decision criteria are identified and made available to the agent.

This requires more initial effort than granting an agent broad autonomy. Existing work must be examined, domain knowledge represented, memory structures designed, and early outputs reviewed by knowledgeable humans.

A mentor is therefore necessary during early operation.

The mentor's role is not simply to approve or reject outputs. The mentor communicates what matters in the job: why a particular source is credible, why a conclusion is too strong, what context should have been considered, or when additional investigation is necessary. These corrections should become part of the agent's evolving knowledge and procedures rather than disappear as one-time feedback.

This resembles on-the-job training more than conventional software deployment. The initial specification defines the job, but the agent becomes competent through repeated performance and feedback.

The cost of this approach is evident. Job analysis, knowledge externalization, and mentoring require substantial human participation. The benefit is that the starting behavior is understandable and governable.

Autonomy can then increase progressively.

At first, the agent may follow explicitly defined procedures. Later, accumulated procedural memory can allow it to recommend alternative search strategies, reorder workflow steps, or propose a more efficient way to satisfy its mission. A human can evaluate these proposals before they become part of the agent's operating specification.

The strategy is therefore:

**transfer the job first, then progressively expand autonomy.**

This differs from open-ended self-improvement approaches in which the primary objective is to discover increasingly capable behaviors. Our goal is narrower: growth should remain anchored to a delegated job and should improve fulfillment of that job's mission.

### 3.6 Specification and Implementation Must Co-evolve

The research fellow case also showed that APF, AJD, and workflow specifications cannot be completed once and then handed over for implementation.

Implementation repeatedly revealed missing distinctions in the specification. Designing interactions exposed responsibilities that had not been explicit. Implementing literature discovery revealed that a search intent without sufficient research context produced weak results. Designing advisory responses forced us to distinguish between established knowledge, uncertain claims, and cases requiring further investigation. Memory design raised questions about what should be treated as external-world knowledge and what should remain part of the agent's evolving internal state.

Conversely, changes in the specification affected workflows, prompts, interfaces, and memory representations.

The development process therefore resembled a Twin-Peaks-style co-evolution rather than a linear requirements-to-code pipeline. Job specification, world modeling, workflow design, and implementation became progressively more concrete together.

This was especially important because the object being specified was itself capable of changing through experience. A persistent agent cannot be engineered solely as a static artifact. Its specification must define not only what the agent currently does, but also which parts may evolve, how they may evolve, and how humans retain control over those changes.

### 3.7 From a Research Fellow to Persistent Job Agents

The research fellow case was chosen to isolate the foundations of persistent agency in a knowledge-intensive environment. It does not remove the need for physical-world modeling in industrial applications. Rather, it provides a foundation for extending the approach toward them.

An equipment-maintenance agent, for example, would require additional domains representing equipment state, sensors, control actions, operators, and safety constraints. A quality-management agent would interact with process conditions, measurements, production history, and downstream quality outcomes. Yet the same underlying engineering concerns remain: the agent must be given an enduring job, maintain continuity across executions, accumulate and externalize domain knowledge, receive feedback, and improve while remaining governable.

The broader lesson from the research fellow implementation is therefore that building a persistent agent is not primarily a matter of orchestrating more LLM calls. It is the process of transferring a job into a governable agent layer whose knowledge, experience, and procedures can evolve through continued work.

The LLM provides general reasoning capability. The persistent agent provides continuity, specialization, responsibility, and controlled growth.
