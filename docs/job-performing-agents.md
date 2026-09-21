# From Goal-Driven Agents to Job-Performing Agents: A Specification-Driven Approach


## Abstract

* 최근 대규모 언어모델(LLM)의 발전은 AI 에이전트가 수행할 수 있는 일의 범위를 빠르게 확대하고 있다. 현재의 많은 에이전트 연구와 개발은 계획, 도구 사용, 메모리, 반성, 멀티에이전트 협업을 통해 얼마나 복잡한 문제를 해결할 수 있는가에 초점을 맞춘다. 그러나 산업 및 전문 업무 환경에서 더 중요한 문제는 어려운 문제를 한 번 해결하는 것이 아니라, 주어진 책임을 지속적이고 신뢰할 수 있게 수행하는 것이다.

본 논문은 이러한 시스템을 persistent job agent라고 부르고, 에이전트 개발을 LLM의 능력이 아니라 직무의 요구사항에서 시작하는 job-first engineering 관점으로 재구성한다. 이를 위해 Agentic Problem Frames(APF), Agent Job Description(AJD), Workflow라는 세 수준의 명세를 사용한다. APF는 직무가 수행되는 세계와 요구사항, 그리고 에이전트와 세계 사이의 상호작용을 모델링한다. AJD는 에이전트가 지속적으로 책임지는 임무, 권한, 작업환경, 운영 맥락, 지식과 메모리, 평가 방법을 명세한다. Workflow는 이러한 책임을 결정론적 소프트웨어, 확률적 LLM 추론, 인간 판단의 조합으로 실행 가능하게 만든다.

우리는 이러한 접근을 persistent research fellow agent 개발에 적용하였다. 구현 과정에서 LLM 자체보다 이를 둘러싼 지속적 agent layer가 중요하며, 전문지식의 외화, 경험의 재사용 가능한 지식으로의 전환, 인간 멘토를 통한 OJT, 명세와 구현의 공진화, 그리고 검증된 수행 경험에 기반한 점진적 자율성 확대가 필요함을 확인하였다. 이러한 결과는 산업용 에이전트 개발을 단순한 capability orchestration이 아니라 직무를 명세하고 이전하며 지속적으로 진화시키는 engineering problem으로 볼 필요가 있음을 시사한다.



## 1. Introduction

Large language models are rapidly transforming software from passive tools into systems that can reason, use tools, and act with increasing autonomy. Recent agent engineering reflects this shift: agents are increasingly designed to control workflow execution, determine intermediate actions, and dynamically select tools rather than simply follow predefined procedures. OpenAI and Anthropic both emphasize this distinction between fixed workflows and agents whose execution is directed dynamically by the model.

This capability-driven approach has produced increasingly powerful agents. At the same time, greater autonomy introduces new reliability and control challenges, leading current engineering practice to emphasize guardrails, evaluations, permissions, and human oversight.

For industrial applications, however, a more fundamental engineering question arises before autonomy is granted: **what part of a persistent job should be delegated to an agent, and under what knowledge, responsibility, and interaction boundaries?**

An industrial job is not merely a sequence of tasks leading to a one-time goal. It is an enduring responsibility performed repeatedly in a changing environment. A maintenance engineer, quality manager, or research advisor must interpret evolving situations, apply accumulated organizational knowledge, interact with people and systems, and remain accountable for outcomes over time. The challenge is therefore not simply to provide an LLM with sufficient tools and allow it to discover an execution path. Organizations must determine what the agent is responsible for, what it must observe, what knowledge should ground its decisions, which actions it may perform autonomously, and when responsibility must remain with humans.

This suggests a complementary perspective on agent engineering. Rather than beginning with an autonomous LLM and subsequently constraining its behavior, development can begin with the job itself. The job's responsibilities, operating environment, interactions, knowledge, and authority boundaries are first made explicit. Its execution can then be decomposed into deterministic software tasks, LLM-based reasoning tasks, and human tasks. As reliability and operational knowledge accumulate, the scope of LLM reasoning and decision-making can be progressively expanded.

The central engineering question therefore changes from **“How do we constrain an autonomous agent?”** to **“How do we progressively delegate a job to an agent while preserving trust and responsibility?”**

To support this process, we present a specification-driven approach for designing persistent job-performing agents. The approach begins by modeling the world in which the agent operates using an Agentic Problem Frame (APF), specifies its responsibilities and interactions through an Agent Job Description (AJD), and progressively refines these specifications into executable workflows and implementation decisions. We illustrate the approach through the development of a persistent research advisor agent and derive practical lessons for progressively transferring job responsibility from humans and deterministic software to LLM-based agents.



## Problem — Agents can solve tasks, but can they own jobs?
* 현재 agent engineering의 goal/task 중심성을 설명하고, persistent job에서는 responsibility, continuity, knowledge, accountability가 부족함을 제시합니다.
* Many contemporary agent systems begin with model autonomy and progressively introduce tools, constraints, and verification mechanisms to improve reliability.
* We take the opposite starting point. For persistent industrial jobs, we begin by explicitly modeling the job, its environment, responsibilities, and interactions. Autonomy is then introduced progressively as the agent demonstrates its ability to perform those responsibilities reliably.
* Current agent engineering focuses primarily on how an agent can accomplish a goal, but provides limited guidance on how an organization can systematically transfer an enduring job and its responsibilities to an AI agent.


이전에 객체지향 개념이 처음 나왔을 때, 만줄짜리 객체 메소드로 구현한 사례도 본 적이 있다.
구현에 대한 접근 생각 방법을 바꾸는 것이다 

* “문제 해결 중심의 자율 탐색”과 “직무 이관 중심의 점진적 자율화”

| 관점       | 자율 탐색형            | 직무 이관형                   |
| -------- | ----------------- | ------------------------ |
| 출발점      | 해결해야 할 목표/문제      | 지속적으로 수행할 직무             |
| 주된 설계 대상 | 탐색 환경, 도구, 스킬, 검증 | 책임, 지식, 절차, 상호작용         |
| LLM의 역할  | 해결 방법 탐색의 중심      | 명시된 직무 구성요소 중 추론이 필요한 부분 |
| 개선 방법    | 더 좋은 탐색과 검증       | 더 많은 책임의 점진적 이관          |
| 초기 자율성   | 상대적으로 높음          | 상대적으로 낮음                 |
| 최종 지향점   | 목표 달성 능력 향상       | 신뢰 가능한 직무 수행과 점진적 자율화    |

처음에는 사람이 이미 알고 있는 직무 지식을 최대한 외화하여 에이전트에게 줍니다.
* Job → APF → AJD → Workflow → Code / LLM / Human

그 다음 운영 과정에서 축적되는 일화적·절차적·의미적 지식을 기반으로 기존의 고정된 부분을 점차 LLM 판단으로 전환합니다.
* Explicit procedure → bounded reasoning → procedure selection → procedure generation


* “Autonomy first, then constraints”를 보여주는 최근 LLM agent 연구
  * ReAct는 LLM이 추론과 행동을 교차하면서 스스로 다음 행동을 선택하는 전형적인 goal-driven agent 구조입니다. 정해진 workflow보다는 목표와 환경 피드백에 따라 action plan을 갱신합니다.
  * Reflexion은 여기서 한 단계 더 나아가 실패나 환경 피드백을 언어적 reflection으로 기억하여 다음 시도의 의사결정을 개선합니다. 즉, 기본적으로 goal → autonomous trial → feedback → improved trial 구조입니다.
  * Voyager는 이 극단을 가장 잘 보여주는 사례입니다. Minecraft에서 사람이 세부 절차를 정의하지 않고, agent가 자동 curriculum을 만들고 탐색하며 skill library를 축적합니다. 저자들도 이를 “open-ended embodied lifelong learning agent”로 규정합니다.

이 세 논문을 이용하면 첫 번째 접근을 다음처럼 정의할 근거가 생깁니다.
* Contemporary LLM-agent research has largely explored goal-driven autonomy, in which models determine intermediate plans and actions while feedback, tools, and verification mechanisms constrain and improve their search.




## A Job-Centered Alternative
* 직무에서 시작하여 APF → AJD → workflow → implementation으로 내려가는 방법을 설명합니다.
## Building a Research Advisor Agent
연구위원 에이전트 개발 사례를 통해 실제로 job modeling, interaction specification, memory, workflow decomposition이 어떻게 적용됐는지 보여줍니다.
## What We Learned
사례에서 얻은 교훈을 정리합니다. 이미 논의했던 내용들이 정확히 여기 들어갑니다.
* Start from jobs, not workflows.
* Model the world before specifying behavior.
* Interactions are part of the specification.
* Externalize job knowledge.
* Specifications and implementations co-evolve.
* OJT and mentoring remain necessary.
* Autonomy should expand progressively.
## From Controlled Execution to Autonomy
마지막으로 이 방법론이 autonomy를 억제하는 것이 아니라 오히려 신뢰 가능한 자율성을 만드는 출발점이라는 것을 설명합니다.