# MAAD: Automate Software Architecture Design through Knowledge-Driven Multi-Agent Collaboration

Source URL: https://arxiv.org/html/2507.21382v1

MAAD: Automate Software Architecture Design through Knowledge-Driven Multi-Agent Collaboration Thanks: This research is supported by the National Natural Science Foundation of China (NSFC) with Grant No. 62402348 and 62172311; National Research Foundation, Prime Minister’s Office, Singapore under the Campus for Research Excellence and Technological Enterprise (CREATE) Programme; the National Research Foundation, Singapore, and DSO National Laboratories under the AI Singapore Programme (AISG Award No: AISG2-GC-2023-008). The authors would also like to thank the architects who participated in the interviews in this study. 
Journal: TOSEM Volume: 0 0 0 0 CCS: Software and its engineering Software development techniques CCS: Software and its engineering Designing software Ruiyin Li email: ryli_cs@whu.edu.cn Affiliation: School of Computer Science, Wuhan University , Wuhan , China , Yiran Zhang email: yiran002@e.ntu.edu.sg Affiliation: Nanyang Technological University , Singapore , Singapore , Xiyu Zhou email: xiyuzhou@whu.edu.cn Affiliation: School of Computer Science, Wuhan University , Wuhan , China , Peng Liang email: liangp@whu.edu.cn Affiliation: School of Computer Science, Wuhan University , Wuhan , China , Weisong Sun email: weisong.sun@ntu.edu.sg Affiliation: Nanyang Technological University , Singapore , Singapore , Jifeng Xuan email: jxuan@whu.edu.cn Affiliation: School of Computer Science, Wuhan University , Wuhan , China , Zhi Jin email: zhijin@pku.edu.cn Affiliation: School of Computer Science, Wuhan University , Wuhan , China and Yang Liu email: yangliu@ntu.edu.sg Affiliation: Nanyang Technological University , Singapore , Singapore Abstract. 
Software architecture design is a critical, yet inherently complex and knowledge-intensive phase of software development. It requires deep domain expertise, development experience, architectural knowledge, careful trade-offs among competing quality attributes, and the ability to adapt to evolving requirements. Traditionally, this process is time-consuming and labor-intensive, and relies heavily on architects, often resulting in limited design alternatives, especially under the pressures of agile development. While Large Language Model (LLM)-based agents have shown promising performance across various software engineering tasks, their application to architecture design remains relatively scarce and requires more exploration, particularly in light of diverse domain knowledge and complex decision-making. In addition, relying on a single LLM agent often leads to unreliable and inconsistent outcomes, especially when handling tasks that require collaborative reasoning and deliberation. To address the challenges, we proposed MAAD (Multi-Agent Architecture Design), an automated framework that employs a knowledge-driven Multi-Agent System (MAS) for architecture design. MAAD orchestrates four specialized agents (i.e., Analyst , Modeler , Designer and Evaluator ) to collaboratively interpret requirements specifications and produce architectural blueprints enriched with quality attributes-based evaluation reports. We then evaluated MAAD through a case study and comparative experiments against MetaGPT, a state-of-the-art MAS baseline. Our results show that MAAD’s superiority lies in generating comprehensive architectural components and delivering insightful and structured architecture evaluation reports. Feedback from industrial architects across 11 requirements specifications further reinforces MAAD’s practical usability. We finally explored the performance of the MAAD framework with three LLMs (GPT-4o, DeepSeek-R1, and Llama 3.3) and found that GPT-4o exhibits better performance in producing architecture design, emphasizing the importance of LLM selection in MAS-driven architecture design. 
Keywords: Large Language Model, Generative AI, Multi-Agent System, Software Architecture Design 
1. Introduction 

Software architecture design lies at the heart of any successful software project. It defines the system’s high - level structure, allocates responsibilities to components, and prescribes the interactions that satisfy both functional requirements and quality attributes ( 1 ) . In practice, architects must translate often-ambiguous requirements into concrete modules and connectors, select appropriate architectural patterns, and balance competing concerns on quality attributes (e.g., performance, security, maintainability) ( 2 ) . This process is inherently knowledge - intensive and demands deep domain expertise, extensive engineering experience, and careful trade - off analysis. As requirements evolve or new constraints emerge (e.g., legacy dependencies, regulatory mandates, or shifting business goals), the architecture must adapt to the latest requirements without undermining system integrity. Such complexity often leads to cognitive overload, reliance on tacit personal knowledge, and limited exploration of alternative designs, creating bottlenecks that delay delivery and hinder the ability to scale design efforts across projects. 

The advent of Large Language Models (LLMs) has profoundly revolutionized the landscape of Software Engineering (SE) practices, introducing a new paradigm that integrates Generative AI (GenAI) into various development workflows. Especially, LLM-based tools such as ChatGPT are already being adopted across a broad spectrum of SE activities ( 3 , 4 ) . However, when a single LLM agent tackles complex, multi-step tasks, the results can be unreliable or inconsistent, as isolated reasoning tends to overlook cross-cutting concerns and introduce hallucinations ( 5 ) . To mitigate these issues, recent research has turned to Multi - Agent Systems (MAS), where several role - specific LLM agents communicate and refine outputs. By emulating the collaborative dynamics of human design teams, each agent focuses on a particular subtask, such as requirements analysis ( 6 , 7 ) . 

Unlike single LLM deployments, MAS architectures emulate human development teams by enabling agents to focus on specific roles, reason independently, and communicate iteratively toward a shared goal ( 7 ) . This distributed intelligence approach excels in scenarios that require diverse expertise, complex problem decomposition, parallel task execution, and collaborative deliberation. As shown in recent work ( 8 , 9 , 10 ) , MAS-based GenAI frameworks can reduce hallucinations, increase robustness, and improve overall productivity, especially in dynamic environments where rapid iteration is essential. These capabilities are especially critical in the context of modern software development, where the accelerated pace of software delivery demands rapid iteration and market responsiveness to evolving market requirements. 

However, despite promising advances, the high-abstraction and knowledge-intensive nature of software architecture design remains under - automated. Existing LLM and MAS applications excel at coding tasks but receive comparatively less focus on the architectural phase, where decisions about module decomposition, protocol selection, and non - functional trade - offs are both interdependent and domain - specific. Architects still carry the bulk of this work, leading to several challenges: limited reuse of prior designs when diving into unfamiliar domains, low generalizability of the architecting process due to the difficulty of applying and transferring knowledge across teams, and insufficient integration of domain-specific external knowledge. As a result, organizations struggle to accelerate the architecture formulation and maintain consistency as projects evolve over time and grow in scope. 

To bridge this gap, our goal is to propose and validate an automated software architecture design framework Multi-Agent Architecture Design (MAAD) , which orchestrates four role-specific LLM agents (i.e., Analyst , Modeler , Designer , and Evaluator ) to jointly derive, model, and evaluate software architectures from given Software Requirements Specifications (SRS) ( 1 ) . Our results show that MAAD distinguishes itself as a knowledge-driven architecture design methodology, and it outperforms general MAS like MetaGPT ( 11 ) , particularly in terms of architectural completeness. Moreover, MAAD seamlessly integrates external knowledge sources, supporting the addition of authoritative literature and private knowledge bases for domain-specific best practices. This extensibility enables MAAD to tailor architecture generation to specialized domains, mitigate hallucinations through logical reasoning, and deliver consistent, high-quality architecture designs with minimal human intervention. By further comparing three LLMs (i.e., GPT-4o, DeepSeek-R1, and Llama 3.3) as the foundational LLMs supporting the agents, the results show that the MAAD framework equipped with GPT-4o can achieve relatively better performance than the other two LLMs in our study. By pioneering a domain-aware, automated approach to architecture design, MAAD lays the foundation for next-generation development platforms that deliver rapid, reliable, and maintainable software architectures with minimal human oversight. Our contributions are threefold: 

• 
Framework Design : We present the architecture of MAAD and detail inter-agent protocols for collaborative reasoning. The four agents of MAAD can mirror the realistic architecture design process and generate complete architecture designs with evaluation reports. 

• 
Knowledge Integration : We demonstrate how MAAD ingests and applies external knowledge sources to ground architectural decisions and mitigate hallucinations. MAAD supports the automation of architecture design based on customized domain knowledge. 

• 
Empirical Evaluation : Through quantitative and qualitative assessments, we show that MAAD yields better architecture designs than the existing MAS baseline MetaGPT ( 11 ) and significantly reduces manual intervention. 

This paper is an extension of our previously published vision paper ( 12 ) by delivering the following substantial contributions. In comparison, this extended version (1) introduces the detailed description of each agent’s implementation and their interaction mechanism, (2) evaluates the MAAD’s performance and compares it with a baseline MetaGPT ( 11 ) , (3) explores the influence of incorporating external knowledge on architecture design based on the same requirements specifications, (4) investigates the performance differences among three leading foundational LLMs (i.e., GPT-4o, DeepSeek-R1, and Llama 3.3), (5) validate the MAAD’s practical effectiveness with the challenges and suggestions for future improvement through interviews with three architects, and (6) analyzes the advantages of MAAD over MetaGPT with a discussion of the impact of infusing external knowledge. 

The remainder of this paper is organized as follows: Section 2 introduces related studies of this work. Section 3 elaborates on the design of the MAAD framework, and Section 4 presents the research questions. Section 5 describes the study results and our findings, and Section 6 discusses the results of this study. Section 7 examines the threats to the validity of this study. Section 8 summarizes this study and outlines the future work. 

2. Related Work 

2.1. Software Architecture Design 

Software architecture design has evolved significantly over the past few decades, transitioning from experience - based heuristics to systematic, model - driven engineering practices ( 1 ) . Early approaches in the 1990s emphasized layered architectures and object-oriented decomposition, which were mainly guided by expert judgment and best practices ( 13 ) . These foundational efforts defined architecture as a high-level abstraction encompassing system structure, behavior, and key quality attributes ( 14 ) . As systems grew in complexity, researchers introduced Architecture Description Languages (ADLs) to bring formality to architectural modeling and analysis. Prominent ADLs such as AADL ( 15 ) allowed researchers and practitioners to define system components, connectors, and configurations systematically ( 16 ) . 

With the rise of distributed and service-oriented computing in the 2000s, Service-Oriented Architecture (SOA) became a dominant paradigm. SOA promoted loose coupling and service abstraction, facilitating scalability and adaptability in enterprise environments. Around the same time, Model-Driven Architecture (MDA) further emphasized the transformation of abstract architectural models into platform-specific implementations through model transformations, bridging the gap between abstract design and executable systems ( 17 ) . 

In the past decade, the rise of cloud-native systems, microservices, and event-driven architectures has reshaped software architecture around modularity, scalability, and deployment agility. These trends were accompanied by tools like Kubernetes, practices such as Domain-Driven Design (DDD) ( 18 ) , and architectural styles like serverless computing and containerization. In parallel, architectural decision modeling and quality attribute-driven design approaches like Architecture Tradeoff Analysis Method (ATAM) ( 19 ) have provided systematic ways to align architectural choices with business goals and non-functional requirements ( 1 ) . 

Looking forward, the integration of Artificial Intelligence (AI) into architecture design is emerging as a transformative force ( 20 ) . Knowledge - based systems and AI - assisted tooling are beginning to automate routine architectural decisions, generate candidate designs, and adapt architectures in response to evolving requirements ( 21 ) . 

Overall, the software architecture landscape is undergoing a transformation driven by AI. Architectural practices are increasingly integrating intelligent tooling and AI-based components capable of learning, adapting, or generating architecture elements. The modern architecture design process is becoming more iterative, model-centric, quality-aware, and AI-powered, supported by both theoretical frameworks and practical intelligent tools. 

2.2. Large Language Models for Software Architecture Design 

Large Language Models (LLMs) have received significant attention from both academia and industry community due to their remarkable performance across a wide range of Software Engineering (SE) tasks ( 3 ) . Recent studies have begun to investigate the intersection between LLMs and software architecture design, highlighting the potential of LLMs to enhance architecture design processes and decision-making. 

Schmid et al . ( 22 ) conducted a systematic literature review by analyzing 18 studies on the application of LLMs in architectural tasks (e.g., design - decision classification, pattern detection). Their work identifies emerging use of LLM techniques but also highlights underexplored areas (e.g., code - generation from architecture and architecture conformance checking) and calls for stronger architecture evaluation frameworks. Esposito et al . ( 21 ) conducted a multivocal literature review synthesizing 37 sources, including both academic and gray literature. They identified key challenges regarding the use of LLMs for architecture design, such as LLMs’ accuracy issues, hallucinations, ethical and privacy concerns, the absence of architecture - specific datasets, and a dearth of architecture evaluation frameworks. Moreover, they advocated for research into general architecture evaluation methodologies, LLMs’ transparency and explainability, ethical guidelines, and tailored benchmarks to support real - world adoption. Eisenreich et al . ( 23 ) proposed a semi-automated approach for generating candidate software architectures directly from requirements using LLMs. Their work demonstrates the feasibility of leveraging natural language requirements to guide early-stage architectural decision-making. Dhar et al . ( 24 ) examined the use of LLMs for generating Architecture Decision Records. Their study found that while GPT-4 can generate relevant design decisions in zero-shot settings, its performance does not yet match human-level reasoning. Interestingly, their results suggest that more cost-efficient models, such as GPT-3.5, can reach competitive performance under few-shot settings. 

Despite these promising advancements, the application of LLM-based agents, particularly multi-agent systems, to software architecture design remains relatively underexplored compared to their use in other SE activities ( 21 , 7 ) . Current research primarily focuses on single-agent reasoning or generation tasks regarding certain architecture activities, resulting in a notable gap in understanding of how distributed or collaborative LLM agents might co-design, evaluate, and iteratively refine software architecture in a more autonomous or interactive way. 

3. MAAD Framework 

In this section, we present our proposed MAAD framework in three parts: first, we provide an overview of the MAAD framework (see Section 3.1 ); second, we detail the design and specifications of the constituent agents of the MAAD framework (see Section 3.2 ); and third, we describe the collaborative mechanisms of the MAAD framework (see Section 3.3 ). 

3.1. Overview 

The MAAD framework implements a knowledge - driven, multi - agent pipeline that autonomously transforms a Software Requirements Specification (SRS) into a complete architecture design (see Figure 1 ). MAAD comprises four specialized agents, and each individual agent is equipped with perception, reasoning, and action capabilities. 
Figure 1 . Overview of the MAAD framework 
As illustrated in Figure 1 , after receiving the SRS, the Analyst agent examines its content, identifies and distills key aspects of the requirements into four decomposed requirements artifacts. Once the Modeler agent perceives the generated artifacts in the artifacts pools from the Analyst agent, it constructs the system’s multi - view representation according to the “4+1” architecture view models proposed by Philippe Kruchten ( 25 ) , which is a widely used architecture description framework in practice. When the Designer agent perceives the artifacts by the Analyst and Modeler agents, it synthesizes the final architecture documentation. The documentation formulates system goals and detailed design specifications. Finally, the Evaluator agent assesses the generated architecture by checking the architecture views against the original SRSs. It produces an ATAM Evaluation report and a Mismatch Analysis report, pinpointing any deviations between generated artifacts and SRSs, and trade - offs during architecture design. Overall, the four agents enact a cohesive and feedback - driven workflow. Through iterative communication and artifact exchange, MAAD ensures that each agent’s autonomous activities cumulatively yield a robust, traceable architecture design. 

3.2. Agent Design 

3.2.1. Analyst Agent 

The Analyst agent serves as a critical component responsible for comprehending the SRS, filtering and classifying requirements, and documenting those requirements that exert significant influence on the architectural design. This agent plays a pivotal role in ensuring alignment between subsequent architectural decisions and overarching business objectives. The Analyst agent is designed to produce four decomposed requirements artifacts: Functional Requirements , Non-Functional Requirements , Architecturally Significant Requirements , and Design Constraints . 

To accomplish this task, the Analyst agent possesses capabilities to comprehend the SRS through the following systematic actions: (1) parsing and structuring the SRS to extract all requirements comprehensively; (2) identifying and filtering Architecturally Significant Requirements (ASRs); (3) classifying requirements into functional and non-functional categories, where functional requirements specify system features and operational tasks, while non-functional requirements encompass quality attributes, resource constraints, and other pertinent considerations; and (4) extracting system design constraints from the SRS. 

3.2.2. Modeler Agent 

The Modeler agent aims to translate the requirements into a software architecture blueprint, specifically to model the overall architecture of the system based on the refined requirements provided by the Analyst agent. This agent generates the “4+1” architecture view models ( 25 ) as its generated artifacts, including Logical Views , Development View , Process View , Physical View , and Scenario View . 

To fulfill these responsibilities, the Modeler agent executes the following actions: (1) prioritizing non-functional requirements to facilitate strategic trade-offs among competing quality attributes; (2) selecting appropriate technology stacks that align with system requirements and constraints; (3) identifying and applying suitable architectural styles and patterns to structure the system effectively; and (4) analyzing and modeling domain components along with their interrelationships to construct comprehensive architecture views. 

3.2.3. Designer Agent 

The Designer agent is in charge of generating detailed architecture documentation based on the architecture views from the Modeler agent, thereby establishing a robust foundation for subsequent code implementation. This documentation encompasses the following components: 

• 
Goals : Define the primary objectives and quality attributes that the proposed architecture aims to achieve, such as performance targets, scalability requirements, and maintainability. 

• 
Detailed Architecture Design : A detailed design specification that defines the system’s components, modules, and subsystems while outlining their interactions, interfaces, and behavioral contracts. 

• 
Component & Connector Specifications : Detailed descriptions of communication protocols (e.g., REST API contracts, event-driven messaging formats, database connection specifications) with explicit error-handling mechanisms, performance thresholds, and reliability constraints. 

• 
Key Technologies : Comprehensive infrastructure specifications, such as resource allocation strategies, deployment configurations, and explicit analysis of scalability versus fault-tolerance trade-offs. 

• 
Design Decisions : Systematic explanation of architectural decisions, encompassing selected architecture patterns, technology choices underlying key structural and behavioral design choices. 

• 
Design Decision Rationale : In-depth justification of technology selections, pattern adoption, and architectural trade-off analysis with consideration of alternative approaches and their respective limitations. 

• 
Executable Prototype Skeleton : Generation of structured code scaffolding and skeletal implementations for critical system modules, including interface and basic operational logic. 

To fulfill its responsibilities, the Designer agent executes the following actions: (1) analyzing and interpreting architecture views to extract design requirements and constraints; (2) synthesizing comprehensive design specifications that bridge conceptual models with implementable solutions; (3) defining precise component interfaces and interaction protocols to ensure seamless system integration; (4) documenting architectural decisions with explicit rationale and trade-off analysis to support future maintenance and evolution; and (5) generating code skeleton that provides structured foundations for development teams. 

3.2.4. Evaluator Agent 

The Evaluator agent is tasked with rigorously assessing the architectural artifacts generated by other agents to ensure their alignment with the input SRS. This agent produces two main evaluation reports: 

• 
ATAM Evaluation Report : A comprehensive architectural assessment based on the Architecture Tradeoff Analysis Method (ATAM), systematically evaluating quality attribute scenarios, identifying architectural strengths and weaknesses, analyzing potential risks, and documenting trade-off implications across competing quality attributes. 

• 
Mismatch Analysis Report : A detailed diagnostic analysis that identifies and categorizes the discrepancies between the generated architectural solutions and the original requirements specifications, providing critical insight into areas requiring architecture refinement, requirements clarification, or design iteration. 

To fulfill these responsibilities, the Evaluator agent performs the following actions: (1) conducting a comprehensive evaluation of generated architectural artifacts, including structural diagrams (class, component, package), behavioral diagrams (sequence, activity, state), and deployment specifications to identify potential risks with ASRs and design constraints as specified in the SRS; and (2) performing systematic mismatch analysis through requirements traceability matrices to quantify and categorize identified discrepancies. 

3.2.5. Knowledge-driven Agent Setting 

In a multi-agent framework like MAAD, the infusion of external knowledge is essential to equip agents with the insights they need for high - quality, context - aware artifact generation. Software architecture design is inherently complex: agents cannot rely solely on initial requirements and intermediate artifacts but also leverage existing, external knowledge, ranging from industry standards, architecture patterns, and domain - specific best practices. By integrating external knowledge, agents can reduce informational gaps, mitigate the risk of omitting critical considerations, and ensure that the generated artifacts reflect both theoretical principles and empirical insights. 

In our MAAD framework, we focus knowledge infusion on the two agents whose tasks most critically depend on external context: 

Modeler Agent : Once receiving the artifacts generated by the Analyst agent, the Modeler agent enhances its understanding by performing a similarity search over a vectorized knowledge base. This database is segmented into thematic categories (e.g., layered architecture, component - and - connector style), and the segments are then incorporated into the prompt fed to the Modeler agent. The integration of this external knowledge allows the Modeler agent to incorporate broader perspectives on architecture design, ensuring that the architecture views it generates are grounded in both the provided requirements artifacts and additional contextually relevant information. 

Designer Agent : Similarly, before generating the architecture documentation, the Designer agent performs a vector search in the same external knowledge base. It retrieves the three most pertinent text segments, which are then used as background knowledge to inform the architecture design process. By referencing these external insights, the Designer agent can ensure that the design rationale and decisions align with industry standards and proven architectural patterns, further refining the quality of the generated architecture documentation. 

Through pilot experiments, we specifically chose to retrieve the three most similar text segments to the prompt provided to the agents. This decision was driven by the need to strike a balance between relevance and conciseness for the agents’ generated artifacts. By selecting only the most relevant segments, we ensure that agents receive targeted knowledge without overwhelming them with excessive information, which could lead to unnecessary verbosity or potential confusion in the generated artifacts. 

This approach of infusing external knowledge not only enhances the agents’ decision-making but also ensures that the generated artifacts are aligned with established best practices and external expertise, ultimately contributing to the robustness and effectiveness of the architectural design process within the MAAD framework. 

3.3. Agent Collaboration 

The architecture design process of the MAAD framework begins with the Analyst agent , which meticulously analyzes the input Software Requirements Specification (SRS). The agent extracts key requirements and constraints that are critical to the architecture design, ensuring a thorough understanding of the SRSs that will guide subsequent steps. 

Building on the results of the Analyst agent’s analysis, the Modeler agent takes on two primary responsibilities: 1) formulating high-level architectural decisions that establish a clear direction for the system’s design, and 2) identifying the relevant domains and prioritizing the quality attributes (e.g., performance, security, scalability) that the generated system must address. These tasks ensure that the architecture is both robust and aligned with the functional and non-functional requirements outlined in the SRS. 

Following the high-level decisions made by the Modeler agent, the Designer agent focuses on the concrete implementation of the architecture. The Designer agent creates detailed Unified Modeling Language (UML) diagrams, including class diagrams, sequence diagrams, and deployment diagrams that serve as comprehensive blueprints for the system’s design. These diagrams not only capture the structure and interactions of the system, but also provide an essential reference for guiding the subsequent code development, ensuring alignment between design and implementation. 

Then, the Evaluator agent verifies the consistency and alignment of the generated architecture artifacts with the original SRS. If mismatches are identified, the Evaluator agent works to pinpoint the root cause of the discrepancies. It collaborates with the relevant agents (the Analyst , Modeler , or Designer agents) to resolve the issue. Following the resolution, the affected agents update their respective artifacts, ensuring that the entire architecture design is coherent and consistent with the SRS. 

Once the Evaluator agent confirms that all artifacts align with the SRS and that all issues have been addressed, the architecture design process concludes. The final architecture design, which includes detailed UML diagrams, conceptual views, and documented architectural decisions, forms the foundation for subsequent code implementation. This structured approach ensures that the system’s architecture is both comprehensive and well-aligned with the original SRSs, providing a solid basis for efficient and effective software development. 

4. Study Design 

In this section, we present the Research Questions (RQs) in Section 4.1 and provide an overview of the research process in Section 4.2 . 

4.1. Research Questions 

Rationale : MAAD employs a multi-agent framework to automate the software architecture design. While such automation presents considerable potential, it is essential to evaluate its practical effectiveness. Therefore, RQ1 seeks to evaluate the effectiveness of MAAD in generating viable and relevant software architectures. To achieve a comprehensive evaluation, we will first perform a comparative analysis of MAAD’s performance against baseline methods. Then we will present a case study where MAAD is applied to a realistic architecture design scenario to illustrate its strengths and weaknesses. Moreover, we conduct a human evaluation of the generated architecture with industry architecture experts to gain more insights on MAAD’s performance. By answering this question via this multifaceted evaluation, we aim to provide empirical evidence of MAAD’s capability to effectively automate architecture design, benchmark its performance, and validate its potential to enhance efficiency and scalability in real-world software development contexts. 

Rationale : The MAAD framework can leverage the infusion of external knowledge (e.g., knowledge from existing system designs, authoritative literature, and architecture experts) to improve trustworthiness and correctness. The integration of external knowledge is crucial to ensure that agents in the MAAD framework can make informed decisions that align with best practices and real-world requirements. RQ2 investigates methods for infusing and synthesizing this knowledge, such as knowledge extraction and knowledge representation frameworks. Addressing this RQ will help improve the accuracy and effectiveness of the artifacts generated by the agents, ensuring that the architecture aligns with both theoretical foundations and practical constraints. 

Rationale : Different LLMs vary in terms of their generations of underlying architecture views and specialization, which may influence their ability to generate high-quality software architecture designs. RQ3 investigates the impact of various LLMs, such as GPT-4o, DeepSeek R1, and Llama 3.3 within the MAAD framework. By understanding which LLMs are most suitable for specific architecture tasks (e.g., requirements analysis, trade-off evaluations), we can refine the framework to ensure optimal performance and reliability in automated software architecture design. 

4.2. Experiment Settings 

This section outlines the experimental design used to validate our proposed Multi-Agent Architecture for Development (MAAD) framework. Figure 2 presents the study design of our work, which includes requirements dataset selection, comparison between MAAD and a baseline (i.e., MetaGPT ( 11 ) ), and validation from practitioners (architects). 
Figure 2 . Process of the sudy design 
4.2.1. Dataset 

The requirements dataset is collected from Jin et al. ( 26 ) , which was gathered from public datasets, including PURE ( 27 ) (a dataset of 79 publicly available natural language requirements documents collected from the Web) and private industrial requirements documents ( 28 ) . We then select the requirements of the “Space Fraction System” (SFS) as the input to conduct a case study in this work. 

4.2.2. Baseline 

To compare our proposed MAAD framework with peer MAS, we select MetaGPT ( 11 ) as the baseline. MetaGPT is a state-of-the-art multi-agent system that generates comprehensive software artifacts from a single requirements statement. Structurally, MetaGPT encompasses four agents Product Managers , Architects , Project Managers and Engineers . MetaGPT provides the entire process of a software company along with carefully orchestrated Standard Operating Procedures (SOPs) ( 11 ) . 

4.2.3. Large Language Model Selection 

To ensure comprehensive evaluation across diverse model configurations and capabilities, we select three representative high-performance LLMs as the foundational LLMs for MAAD based on their technical diversity, capabilities, and availability. 

• 
GPT-4o : OpenAI’s proprietary multimodal LLM developed ( 29 ) with approximately 200 billion parameters. It excels at code generation and mathematical reasoning, with fast 320ms response times for real-time interactions, and it has been commonly used in business applications, education, and content creation. 

• 
Llama 3.3 : Meta’s latest open-source LLM (70 billion parameters) with an extended 32K-token context window ( 30 ) . It improves multilingual understanding and generation, matches GPT-4 on many NLP benchmarks, and offers efficient fine-tuning for custom applications. 

• 
DeepSeek-R1 : DeepSeek’s open-source 671-billion-parameter model ( 31 ) that cuts training costs by 60% and boosts inference throughput by over 2.3× compared to dense 70B models. Its support for domain-specific tuning across specialized fields makes it relevant for evaluating adaptability in diverse SE contexts. 

4.2.4. Interviews 

To further validate the correctness and practical utility of MAAD, we solicit feedback from experienced industry practitioners through semi-structured interviews. 

Interview Protocol : We design an interview protocol by following the guidelines for empirical studies in software engineering proposed by Wohlin et al. ( 32 ) . We conduct semi-structured interviews with 3 open-ended questions that are designed to elicit practitioners’ perspectives on automated architecture design using the MAAD framework. The interview questions allow the participants to freely and openly express their experiences and insights on the generated artifacts by MAAD from 11 real-world SRSs. The interview procedure consists of three parts: first, participants receive a concise overview of the study’s objectives and are asked to review the artifacts generated by our framework for representative user requirements cases. Second, interviewees are asked demographic questions (e.g., role, years of professional experience). Third, the first author conducts the interviews, each of which lasted 35 to 50 minutes. With the interviewees’ consent, we audio-record the interviews and fully transcribed them for an in-depth analysis. The interview protocol and open questions are available in our replication package ( 33 ) . 

Data Analysis : The first author conducts a qualitative analysis of the transcripts, with the fourth author independently reviewing all coded segments to ensure consistency and mitigate bias. The data analysis proceeds as follows: (1) Extracting data: Transcripts are carefully read to identify salient comments regarding MAAD’s artifact quality; (2) Coding data: Initial codes are generated to categorize participants’ views on MAAD’s generated artifacts. These codes guide subsequent thematic analysis; (3) Examination: To ensure analytical rigor, the fourth author independently reviews the coded data and resolves any discrepancies through discussion. 

5. Results 

5.1. Results of RQ1 

5.1.1. Generated Artifacts of the MAAD Framework 

To answer RQ1, we employed the MAAD approach to generate software architecture design using the Software Requirements Specifications (SRSs). We use requirements of the “Space Fraction System” (SFS) as a running example from our dataset (see Section 4.2.1 ). SFS is an interactive web-based educational platform that targets sixth-grade students and gamifies fraction arithmetic by presenting questions, providing immediate feedback, and tracking study performance. Students can answer fraction-related arithmetic questions, receive immediate feedback, and track their scores. 

Upon ingesting the SRSs of the SFS case, the Analyst agent (see Section 3.2.1 ) automatically produces structured documentation and stores the four sets of artifacts: architecturally significant requirements (ASRs), functional requirements, non-functional requirements, and design constraints. Each artifact type is generated based on a predefined prompt template in which the complete SRS text is embedded. Subsequently, the remaining three agents generate the other artifacts as defined in Section 3 : the Modeler agent produces the “4+1” architecture view models ( 25 ) , which comprise various UML diagrams organized as follows. 

• 
Logical View includes 3 UML diagrams: Class Diagram, Object Diagram, and State Diagram. 

• 
Development View includes Package Diagram and Component Diagram. 

• 
Process View includes Activity Diagram, Sequence Diagram, and Collaboration Diagram. 

• 
Physical View includes Deployment Diagram and Container Diagram. 

• 
Scenario View is also known as Use Case Diagram. 

The Designer and Evaluator agents then generate the relevant architecture documentation and architecture evaluation reports as defined in Section 3.2.3 and Section 3.2.4 . All architecture artifacts generated by the MAAD approach are publicly available in the replication package ( 33 ) . 

5.1.2. Comparison between MAAD and MetaGPT 

To evaluate MAAD’s performance against the baseline, we conducted a comparative analysis with MetaGPT ( 11 ) (see Section 4.2.2 ). Since MetaGPT was not specifically designed for architecture design tasks, we focused our comparison on the overlapping artifact types produced by both systems, including artifacts of requirements analysis, system design, and documentation. This way allowed for a meaningful comparison of the capabilities of MetaGPT and MAAD within the same types of generated artifacts. To compare the artifacts generated by MAAD and MetaGPT, we conducted experiments on the two MAS frameworks, respectively. 

In terms of requirements analysis , the Product Manager agent of MetaGPT reads the input SRSs and generates a structured SRSs, including: 

• 
Product Goals : define concise statement of the system’s primary features. 

• 
User Stories : describe user usage scenarios and interaction processes. 

• 
Competitive Analysis : compares competitors’ features of similar products, highlights their strengths and weaknesses, and provides feature optimization recommendations. 

• 
Requirement Analysis : refines functional and non-functional requirements (such as performance and compatibility). 

• 
Requirements Pool : includes a prioritized requirements list. 

• 
UI Design Draft : presents sketch layouts of UI design and shows basic interface design descriptions. 

Given the characteristics of MetaGPT, which “ takes a one-line requirement as input and outputs user stories, competitive analysis, requirements, data structures, APIs, documents, etc. ’’ 1 1 1 https://github.com/FoundationAgents/MetaGPT/ , it will automatically “complete” certain unreal details based on LLM’s text generation capability, for example, specify the specific demands of students and teachers (e.g., “track progress” and “update questions independently”) of the SFS project. Regarding the results of requirements analysis, MetaGPT cannot categorize requirements but only produce five top priority requirements for SFS, such as labeled as P0 (i.e., core functional requirements) or P1 (i.e., secondary functional requirements). By contrast, MAAD’s Analyst agent produced 6 functional requirements categories encompassing 21 detailed requirements, alongside 11 non-functional requirements and 8 architecture-related requirements. This finer granularity and greater coverage yield a more robust foundation for downstream architecture modeling. 

Regarding system design (that is, architecture modeling), MetaGPT employs its Architect agent to derive and visualize the system architecture. The Architect agent operates on the product requirements document produced by the Product Manager agent, and the Architect agent can generate two mermaid files that record the UML syntax of a class diagram and a sequence diagram, respectively, and a brief Implementation Approach description and Anything UNCLEAR declaration. Figure 3 and Figure 4 present a comparison of the class and sequence diagrams produced by MAAD and MetaGPT, respectively. To enable visualization of the UML diagrams produced by both systems, we employed PlantUML 2 2 2 https://plantuml.com/ to convert the textual UML specifications into graphical representations. In particular, this visualization capability is integrated directly into the MAAD framework, allowing it to automatically generate visual UML diagrams from its generated artifacts. 
(a) Class diagram generated by MAAD (b) Class diagram generated by MetaGPT Figure 3 . Comparison of class diagrams generated by MAAD and MetaGPT (a) Sequence diagram generated by MAAD (b) Sequence diagram generated by MetaGPT Figure 4 . Comparison of sequence diagrams generated by MAAD and MetaGPT 
In terms of documentation , MetaGPT produces one JSON file to record technical solutions, including the following fields: 

• 
Required Python Packages include the necessary Python packages. 

• 
Required Other language third-party packages refer to the selection of technology stack, which belongs to the technology dependency decision. 

• 
Logic Analysis defines the division of responsibilities of modules or files (e.g., game.js handles game logic, admin.js manages backend functions), reflecting the system architecture design. 

• 
Task List presents the code files to be implemented, which belong to the development task split. 

• 
Full API Spec describes API design specification. 

• 
Shared Knowledge describes the general design principles of the system (e.g., front-end framework selection, class or function sharing mechanism), which belongs to the architecture constraint description. 

• 
Anything UNCLEAR identifies issues that need to be clarified (such as browser compatibility, administrator interface design), which belong to requirements defect tracking and provide input for subsequent iterations. 

Ideally, all the fields of the MetaGPT JSON files should contain content, but in fact, the experimental results show that certain fields are null values, such as Required Python packages and Full API spec . Compared to MetaGPT, MAAD generates one architectural document that records detailed architecture design solutions (see Section 3.2.3 ). Furthermore, MetaGPT lacks the ability to comprehensively evaluate the quality of its generated architecture design. Consequently, we are unable to assess the differences between MAAD and MetaGPT that use architectural evaluation approaches executed by LLMs. To address this, we conducted interviews with industry practitioners. In general, we summarized the results of the comparison between MAAD and MetaGPT with respect to the automated architecture design in Table 1 . 
Table 1 . Comparison between MAAD and MetaGPT MAS ASR Extraction Architecture Model Integrity Documentation granularity Architecture Evaluation MAAD Extract ASRs and classify requirements based on the original requirements. Generate “4+1” architectural view models. Generate detailed architecture documentation. Generate ATAM architecture evaluation report and mismatch report. MetaGPT Generate a software requirements specification. Contain unreal requirements generated by LLMs. Generate only class diagrams and sequence diagrams. Generate a simple technical solution. No evaluation mechanism. 
5.1.3. Interviews with Architects 

To further evaluate the performance of MAAD, we interviewed three practitioners to get their viewpoints. Three main themes emerged: reasons for positive ratings ( ), reasons for negative ratings ( ), and suggestions for improving generations ( ). 

Correctness . Participants agreed that MAAD’s generated architectural view models align with established design principles, even though they are less complex than those found in industrial systems. Three participants conveyed the same viewpoint, “ the generated architecture view models confirm the principles of architecture design, although the models are not as complex as industrial systems ” (P1, P2, P3). 

Practicability . All interviewees reported that MAAD is highly useful for assisting architects, particularly through its detailed mismatch reports. P1 emphasized that: “ MAAD is undoubtedly useful to assist architects, especially mismatch reports. Its strength lies in the utilization of extensive external knowledge: the human brain has limited memory, whereas an LLM can schedule and recall knowledge much more comprehensively and rapidly ”. Moreover, the knowledge-driven feature of MAAD is highly rated by two participants, “ External knowledge libraries can better support customized domain-specific software design ” (P1, P3). 

Trustworthiness . Despite acknowledging MAAD’s practical benefits and potential, participants remained cautious about relying solely on LLM-generated artifacts. One participant remarked: “ Trustworthiness and explainability issues persist — not because of MAAD per se, but as a general challenge for all LLM outputs, especially in safety-critical domains ” (P1). 

Limitations in Correctness . An interviewee flagged specific inaccuracies in the MAAD’s performance. For example, “ MAAD extracts certain non-functional requirements and links them to quality attributes (e.g., maintainability), but these associations sometimes lack clear justification and may be incorrect ” (P1). 

Level of Detail . Several participants noted that the generated models lack sufficient granularity. One participant commented: “ The generated architectural view models omit detailed UML descriptions, and the relationships between entities are not always represented ” (P1). Another participant similarly noted: “ Compared with industrial applications, the complexity and granularity of requirements analysis and architecture documentation do not meet industrial standards ” (P2). 

Suggestions . P2 and P3 proposed that future multi-agent systems could assign each agent a specialized LLM, for instance, using a code-focused LLM for programmer agents, and equipping fine-tuned and domain-specific LLMs for other agent roles (P2, P3). Regarding reuse, P3 mentioned that “ agent design should consider adding a memory mechanism and reusing the architectural design based on previous business scenarios ” (P3). 

Challenges . Assessing MAAD’s Evaluator agent remains challenging due to the subjective nature of design; as one participant observed, “ Design solutions that satisfy most requirements are good designs, but human evaluators can exhibit bias; architects often favor their own solutions over those generated by LLMs ”. This participant also highlighted the enduring issue of tacit knowledge: “ During software architecture design, much tacit knowledge is hard to capture and cannot yet be leveraged, which remains a major obstacle in the automated design driven by knowledge ” (P1). 

5.2. Results of RQ2 

In the MAAD framework, both the Modeler and Designer agents leverage reference knowledge stored in a vector database to generate artifacts. We selected the third (2012) and fourth (2021) editions of the book Software Architecture in Practice ( 34 , 1 ) as authoritative external knowledge sources and embedded them into a vector database via Retrieval-Augmented Generation (RAG) ( 35 ) . To evaluate the impact of external knowledge on the generation of architecture design by MAAD, we conducted a comparative analysis of the artifacts produced by MAAD using two distinct prompt templates: one incorporating reference knowledge and the other devoid of such knowledge. Here, we then compared (1) the structural comparison of the system design (via architectural views) produced by the Modeler agent and (2) the mismatch rates reported by the Evaluator agent. 

Structural Comparison . To answer RQ2, we still use the Space Fraction System (SFS) requirements as a running example. Figure 5 and Figure 6 respectively depict the component diagrams generated with and without external knowledge. We focus on the UML Component Diagram, since it clearly illustrates component decomposition and interface-based abstraction at the design level. Figure 5 shows a modular, interface-driven design: for instance, the User Interface component interacts with multiple front-end modules (e.g., Main Menu , Help Section , Navigation ) via the IUserInteraction interface, thereby decoupling presentation logic from its implementations and enhancing maintainability. Within the core system, components such as Scoring Engine , Storyline Engine , and Question Management System communicate through IScoring , IStoryline , and IQuestionManagement interfaces, supporting encapsulation and facilitating future substitution. Explicit dependencies (e.g., IUserInteraction relies on IScoring ) illustrate well-defined collaboration paths and adhere to contract-driven architectural principles. In contrast, Figure 6 illustrates a different component-level interaction diagram, emphasizing functional relationships and the execution flow among system modules. Despite this view clarifying execution sequences among system components, it lacks the abstraction and separation of concerns that interface-oriented modeling provides. Consequently, Figure 6 aligns more closely with a Logical or Runtime View than a proper component view. 
Figure 5 . The component diagram of SFS with reference knowledge Figure 6 . The component diagram of SFS without reference knowledge 
Mismatch Report . In the mismatch report of SFS, the Evaluator agent identifies six mismatches under both conditions (i.e., with or without external knowledge), yielding an identical mismatch rate of 0.188 in each case by Equation 1 . Through the infusion of external knowledge, the identified mismatches include reliance on outdated technologies (e.g., Adobe Flash), omission of essential functions (e.g., real-time score calculation, dynamic storyline adaptation), and inadequate support for key non-functional requirements (e.g., security and reliability), which is related to the quality of external knowledge. In contrast, when external knowledge is not provided, the mismatches are primarily associated with functional-detail discrepancies (e.g., inconsistent input-method specifications), missing architectural support for dynamic content updates, and misalignment between non-functional requirements and design elements (e.g., vague security requirements absent corresponding modules). 
(1) Mismatch Rate = Number of Mismatches Total Number of Requirements \text{Mismatch Rate}=\frac{\text{Number of Mismatches}}{\text{Total Number of Requirements}} 
Overall, integrating external knowledge into MAAD’s vector database produced component diagrams that more closely align with established architectural principles, especially in component decomposition, separation of concerns, and interface design. 

5.3. Results of RQ3 

To answer RQ3, we extended our evaluation beyond GPT-4o by generating SFS architecture designs with two additional LLMs, i.e., DeepSeek-R1 and Llama 3.3. Likewise, to conduct structural and mismatches comparison , we equipped MAAD with DeepSeek-R1 (671B) and Llama 3.3 (70B) as foundational LLMs to design the architecture using the same SRS input. Figure 5 , Figure 7 , and Figure 8 depict the component diagrams of SFS by GPT-40, DeepSeek-R1 and Llama 3.3, respectively. 
Figure 7 . The component diagram of SFS generated by DeepSeek-R1 Figure 8 . The component diagram of SFS generated by Llama 3.3 
Structural Comparison . The diagram generated by DeepSeek-R1 emphasizes concrete deployment and runtime aspects: it models explicit artifacts such as client, server, and database instances, and traces interaction paths (e.g., HTTP requests and API calls). Within each unit, it further decomposes functionality (e.g., UI Controller and Narrative Generator), which offers richer visibility into implementation-level responsibilities. In contrast, GPT-4o’s representation centers on logical component dependencies and abstracts away most deployment details and internal module structure. By comparison, Llama 3.3 produces a streamlined, high-level component diagram, which highlights major subsystems (user interface, core system, administrative interface) and their data flows, but omits both fine-grained internal operations and deployment specifics. Moreover, GPT-4o’s diagram tends to capture more detailed interface contracts and internal structures, and Llama 3.3’s output favors conceptual clarity over implementation precision, making it more suitable for early-stage architectural planning rather than detailed design analysis. 

In terms of the component diagrams generated by the three LLMs, all three LLMs yielded functionally plausible architectures within the MAAD framework. Although variations exist across architectures, such differences are reasonable, as software requirements can be satisfied by multiple valid architectural solutions. 

Mismatch Report . As for the mismatch reports generated by the Evaluator agent, six requirement-architecture mismatches (mismatch rate of 0.188) were produced by MAAD with GPT-4o and Llama 3.3 as the base LLMs, including gaps in cross-browser compatibility, maintainability, real-time update strategies, security provisions, administrative interface design, and scalability planning. The architecture design produced by DeepSeek-R1 exhibited ten mismatches, yielding a mismatch rate of 0.313. These mismatches can fall into five categories, including unsupported security mechanisms, outdated technologies, ambiguous adaptivity definitions, missing multi-admin consistency checks, and misaligned enhancements. The results indicate a weaker alignment with the SRS. 

Overall, for the SFS requirements case, the mismatches in the architecture design generated by DeepSeek-R1 are higher than those in GPT-4o and Llama 3.3. This indicates that the selection of basic LLMs for MASs can significantly impact requirements coverage and architectural consistency. 

6. Discussions 

In this section, we explain the experimental results and outline the implications for subsequent practices and research. 

6.1. Interpretations 

6.1.1. Interpretations on RQ1 Results 

According to the results of RQ1, we found that MAAD is capable of generating architecture designs that are both plausibly complete and highly relevant, as demonstrated through the case study of the SFS requirements. Compared to MetaGPT, MAAD exhibits superior performance in analyzing and categorizing user requirements, as well as in producing more fine-grained and comprehensive architectural solutions . These include detailed documentation and support for the “4+1” architecture view models, which enhance traceability (i.e., the ability to trace architectural elements back to original requirements) and architectural consistency (i.e., the internal alignment and integration among different architecture views and design artifacts). 

MetaGPT is designed to automate the entire software development lifecycle, transforming user requirements directly into executable code. In contrast, MAAD is specifically intended for generating architecture designs, with a distinct emphasis on high-level structural design. Their methodologies and design emphases diverge considerably. MAAD emphasizes in-depth requirements analysis, architectural integrity, and systematic evaluation of architectural quality. MetaGPT, on the other hand, operates as an end-to-end software development MAS along with orchestrated Standard Operating Procedures (SOPs), focusing on delivering complete implementations. Due to its specialization in architectural design, MAAD prioritizes robustness of the architectural design process and long-term system maintainability, making it more suitable for projects where system reliability and architectural completeness are critical. 

6.1.2. Interpretations on RQ2 Results 

The results of RQ2 show that the infusion of external knowledge enables MAAD to produce more abstract, interface-driven component diagrams that align more closely with established architectural best practices . In contrast, when external knowledge is absent, the generated diagrams tend to resemble lower-level runtime views. These qualitative improvements highlight the importance of domain grounding in architectural design generation. However, the Evaluator agent’s mismatch rate remains unchanged at 0.188, which implies that the mismatch rate might be influenced by multiple factors, as architectural design involves balancing various requirements and quality attributes. One possible reason could be that the fundamental LLMs used in MAAD already contain architectural knowledge, as previously demonstrated by Soliman et al . ( 36 ) . This implies that the bottleneck may lie in domain-specific knowledge gaps rather than in a lack of general architectural competence. 

Owing to its knowledge-driven feature, MAAD can integrate external knowledge sources, including authoritative literature and proprietary domain-specific knowledge databases. This extensibility supports the integration of specialized domain knowledge, enabling more precise tailoring of architecture generation to the needs of specific application domains. By grounding its reasoning in curated knowledge, we believe that MAAD could help to increase trustworthiness and consistently deliver higher-quality designs with minimal human oversight. 

6.1.3. Interpretations on RQ3 Results 

The results of RQ3 reflect that LLM-generated architectures are not merely “correct” or “incorrect” but reflect inherent priorities and differences during the trainin
