# LLM-Based Multi-Agent Systems for Software Engineering: Literature Review, Vision and the Road Ahead

Source URL: https://arxiv.org/html/2404.04834v4

LLM-Based Multi-Agent Systems for Software Engineering: Literature Review, Vision and the Road Ahead 
CCS: Software and its engineering Software development techniques CCS: Software and its engineering Collaboration in software development Junda He email: jundahe@smu.edu.sg Affiliation: Singapore Management University , 80 Stamford Rd. , 178902 , Singapore , Singapore , Christoph Treude email: ctreude@smu.edu.sg Affiliation: Singapore Management University , 80 Stamford Rd. , 178902 , Singapore , Singapore and David Lo email: davidlo@smu.edu.sg Affiliation: Singapore Management University , 80 Stamford Rd. , 178902 , Singapore , Singapore Abstract. 
Integrating Large Language Models (LLMs) into autonomous agents marks a significant shift in the research landscape by offering cognitive abilities that are competitive with human planning and reasoning. This paper explores the transformative potential of integrating Large Language Models into Multi-Agent (LMA) systems for addressing complex challenges in software engineering (SE). By leveraging the collaborative and specialized abilities of multiple agents, LMA systems enable autonomous problem-solving, improve robustness, and provide scalable solutions for managing the complexity of real-world software projects. In this paper, we conduct a systematic review of recent primary studies to map the current landscape of LMA applications across various stages of the software development lifecycle (SDLC). To illustrate current capabilities and limitations, we perform two case studies to demonstrate the effectiveness of state-of-the-art LMA frameworks. Additionally, we identify critical research gaps and propose a comprehensive research agenda focused on enhancing individual agent capabilities and optimizing agent synergy. Our work outlines a forward-looking vision for developing fully autonomous, scalable, and trustworthy LMA systems, laying the foundation for the evolution of Software Engineering 2.0. 
Keywords: Large Language Models, Autonomous Agents, Multi-Agent Systems, Software Engineering 
1. Introduction 

Autonomous agents, defined as intelligent entities that autonomously perform specific tasks through environmental perception, strategic self-planning, and action execution ( Franklin and Graesser, 1996 ; Albrecht and Stone, 2018 ; Mele, 2001 ) , have emerged as a rapidly expanding research field since the 1990s ( Maes, 1993 ) . Despite initial advancements, these early iterations often lack the sophistication of human intelligence ( Unland, 2015 ) . However, the recent advent of Large Language Models (LLMs) ( Kasneci et al., 2023 ) has marked a turning point. This LLM breakthrough has demonstrated cognitive abilities nearing human levels in planning and reasoning ( Achiam et al., 2023 ; Kasneci et al., 2023 ) , which aligns with the expectations for autonomous agents. As a result, there is an increased research interest in integrating LLMs at the core of autonomous agents ( Lo, 2023 ; Xi et al., 2023 ; Wang et al., 2023c ) (for short, we refer to them as LLM-based agents in this paper). 

Nevertheless, the application of singular LLM-based agents encounters limitations, since real-world problems often span multiple domains, requiring expertise from various fields. In response to this challenge, developing LLM-Based Multi-Agent (LMA) systems represents a pivotal evolution, aiming to boost performance via synergistic collaboration. An LMA system harnesses the strengths of multiple specialized agents, each with unique skills and responsibilities. These agents work in concert towards a common goal, engaging in collaborative activities like debate and discussion. These collaborative mechanisms have been proven to be instrumental in encouraging divergent thinking ( Liang et al., 2023 ) , enhancing factuality and reasoning ( Du et al., 2023 ) , and ensuring thorough validation ( Wu et al., 2023b ) . As a result, LMA systems hold promise in addressing a wide range of complicated real-world scenarios across various sectors ( Horton, 2023 ; Wang et al., 2023e ; Wang et al., 2023a ) , such as software engineering ( Lo, 2023 ; Qian et al., 2024c ; Li et al., 2024a ; Hong et al., 2023 ) . 

The study of software engineering (SE) focuses on the entire lifecycle of software systems ( Kan, 2003 ) , including stages like requirements elicitation ( Goguen and Linde, 1993 ) , development ( Abrahamsson et al., 2017 ) , and quality assurance ( Tian, 2005 ) , among others. This multifaceted discipline requires a broad spectrum of knowledge and skills to effectively tackle its inherent challenges in each stage. Integrating LMA systems into software engineering introduces numerous benefits: 

(1) 
Autonomous Problem-Solving: LMA systems can bring significant autonomy to SE tasks. It is an intuitive approach to divide high-level requirements into sub-tasks and detailed implementation, which mirrors agile and iterative methodologies ( Larman, 2004 ) where tasks are broken down and assigned to specialized teams or individuals. By automating this process, developers are freed to focus on strategic planning, design thinking, and innovation. 

(2) 
Robustness and Fault Tolerance: LMA systems address robustness issues through cross-examination in decision-making, akin to code reviews and automated testing frameworks, thus detecting and correcting faults early in the development process. On their own, LLMs may produce unreliable outputs, known as hallucination ( Zhang et al., 2023 ; Yang et al., 2023b ) , which can lead to bugs or system failure in software development. However, by employing methods like debating, examining, or validating responses from multiple agents, LMA systems ensure convergence on a single, more accurate, and robust solution. This enhances the system’s reliability and aligns with best practices in software quality assurance. 

(3) 
Scalability to Complex Systems: The growth in complexity of software systems, with increasing lines of code, frameworks, and interdependencies, demands scalable solutions in project management and development practices. LMA systems offer an effective scaling solution by incorporating additional agents for new technologies and reallocating tasks among agents based on evolving project needs. LMA systems ensure that complex projects, which may be overwhelming for individual developers or traditional teams, can be managed effectively through distributed intelligence and collaborative agent frameworks. 

Existing research has illuminated the critical roles of these collaborative agents in advancing toward the era of Software Engineering 2.0 ( Lo, 2023 ) . LMA systems are expected to significantly speed up software development, drive innovation, and transform the current software engineering practices. This article aims to delve deeper into the roles of LMA systems in shaping the future of software engineering. It spotlights the current progress, emerging challenges, and the road ahead. We provide a systematic review of LMA applications in SE, complemented by two case studies that assess current LMA systems’ capabilities and limitations. From this analysis, we identify key research gaps and propose a comprehensive agenda structured in two phases: (1) enhancing individual agent capabilities and (2) optimizing agent collaboration and synergy. This roadmap aims to guide the development of autonomous, scalable, and trustworthy LMA systems, paving the way for the next generation of software engineering 

To summarize, this study makes the following key contributions: 

• 
We conduct a systematic review of 71 recent primary studies on the application of LMA systems in software engineering. 

• 
We perform two case studies to illustrate the current capabilities and limitations of LMA systems. 

• 
We identify the key research gaps and propose a structured research agenda that outlines potential future directions and opportunities to advance LMA systems for software engineering tasks. 

2. Preliminary 

2.1. Autonomous Agent 

An autonomous agent is a computational entity designed for independent and effective operation in dynamic environments ( Mele, 2001 ) . Its essential attributes are: 

• 
Autonomy: Independently manages its actions and internal state without external controls. 

• 
Perception: Detects the changes in the surrounding environment through sensory mechanisms. 

• 
Intelligence and Goal-Driven: Aims for specific goals using domain-specific knowledge and problem-solving abilities. 

• 
Social Ability: Can interact with humans or other agents, manages social relationships to achieve goals. 

• 
Learning Capabilities: Continuously adapts, learns, and integrates new knowledge and experiences. 

2.2. LLM-based Autonomous Agent 

Formally speaking, an LLM-based agent can be described by the tuple ⟨ L , O , M , P , A , R ⟩ \langle L,O,M,P,A,R\rangle ( Cheng et al., 2024 ) , where: 

• 
L L symbolizes the Large Language Model , serving as the agent’s cognitive core. It is equipped with extensive knowledge, potentially fine-tuned for specific domains, allowing it to make informed decisions based on observations, feedback, and rewards. Typically, an LLM suited for this role is trained on vast corpora of diverse textual data and comprises billions of parameters, such as models like ChatGPT 1 1 1 https://openai.com/chatgpt/ , Claude 2 2 2 https://claude.ai/ , and Gemini 3 3 3 https://gemini.google.com/app . These models exhibit strong zero-shot and few-shot learning capabilities, meaning they can generalize well to new tasks with little to no additional training. Interaction with the LLM typically occurs through prompts, which guide its reasoning and responses. 

• 
O O stands for the Objective , the desired outcome or goal the agent aims to achieve. This defines the agent’s focus, driving its strategic planning and task breakdown. 

• 
M M represents Memory , which holds information on both historical and current states, as well as feedback from external interactions. 

• 
P P represents Perception , which represents the agent’s ability to sense, interpret, and understand its surroundings and inputs. This perception can involve processing structured and unstructured data from various sources such as text, visual inputs, or sensor data. Perception allows the agent to interpret the environment, transforming raw information into meaningful insights that guide decision-making and actions. 

• 
A A signifies Action , encompassing the range of executions of the agent, from utilizing tools to communicating with other agents. 

• 
R R refers to Rethink , a post-action reflective thinking process that evaluates the results and feedback, along with stored memories. Guided by this insight, the LLM-based agent then takes subsequent actions. 

2.3. LLM-Based Multi-Agent Systems 

A multi-agent system is a computational framework composed of multiple interacting intelligent agents that interact and collaborate to solve complex problems or achieve goals beyond the capability of any single agent ( Wooldridge, 2009 ) . These agents communicate, coordinate, and share knowledge, often bringing specialized expertise to address tasks across diverse domains. 

With the integration of LLMs, LLM-Based Multi-Agent Systems have emerged. In this paper, we define that an LMA system comprises two primary components: an orchestration platform and LLM-based agents . 

2.3.1. Orchestration Platform 

The orchestration platform serves as the core infrastructure that manages interactions and information flow among agents. It facilitates coordination, communication, planning, and learning, ensuring efficient and coherent operation. The orchestration platform defines various key characteristics: 

(1) 
Coordination Models : Defines how agents interact, such as cooperative (collaborating towards shared goals) ( Abdelnabi et al., 2023 ) , competitive (pursuing individual goals that may conflict) ( Wu et al., 2024 ) , hierarchical (organized with leader-follower relationships) ( Zhao et al., 2024 ) , or mixed models. 

(2) 
Communication Mechanisms : Determines how the information flows between the agents: It defines the organization of communication channels, including centralized (a central agent facilitates communication ( Agashe, 2023 ) ), decentralized (agents communicate directly ( Chen et al., 2023a ) ), or hierarchical (information flows through layers of authority ( Zhao et al., 2024 ) ). Moreover, it specifies the data exchanged among agents, often in text form. In software engineering contexts, this may include code snippets, commit messages ( Zhou et al., 2023 ) , forum posts ( He et al., 2022 ; He et al., 2024a ; He et al., 2024b ) , bug reports ( Bettenburg et al., 2008 ) , or vulnerability reports ( Imtiaz et al., 2021 ) . 

(3) 
Planning and Learning Styles : The orchestration platform specifies how planning and learning are conducted within the multi-agent system. It determines how tasks are allocated and coordinated among agents. It includes strategies like Centralized Planning, Decentralized Execution (CPDE) – planning is conducted centrally, but agents execute tasks independently, or Decentralized Planning, Decentralized Execution (DPDE) – both planning and execution are distributed among agents. 

2.3.2. LLM-Based Agents 

Each agent may have unique abilities and specialized roles, enhancing the system’s ability to handle diverse tasks effectively. Agents can be: 

(1) 
Predefined or Dynamically Generated : Agent profiles can be explicitly predefined ( Hong et al., 2023 ) or dynamically generated by LLMs ( Wang et al., 2023c ) , allowing for flexibility and adaptability. 

(2) 
Homogeneous or Heterogeneous : Agents may have identical functions (homogeneous) or diverse functions and expertise (heterogeneous). 

Each LLM-based agent can be represented as a node v i v_{i} in a graph G ⁡ ( V , E ) G(V,E) , where edges e i , j ∈ E e_{i,j}\in E represent interactions between agents v i v_{i} and v j v_{j} . 

3. Literature Review 

In this section, we review recent studies on LMA systems in software engineering, organizing these applications across various stages of the software development lifecycle, including requirements engineering, code generation, quality assurance, and software maintenance. We also examine studies on LMA systems for end-to-end software development, covering multiple SDLC phases rather than isolated stages. 

Search Strategy: We conduct a keyword-based search on the DBLP publication database ( DBLP Computer Science Bibliography, 2024 ) to match paper titles. DBLP is a widely used resource in software engineering surveys ( Chen et al., 2020 ; Zhang et al., 2018 ; Chen et al., 2024c ) , which indexes over 7.5 million publications across 1,800 journals and 6,700 academic conferences in computer science. 

Our search included two sets of keywords: one set targeting LLM-based Multi-Agent Systems (called [agent words]) and the other focusing on specific software engineering activities (called [SE words]). Papers may use variations of the same keyword. For example, the term “vulnerability” may appear as “vulnerable” or “vulnerabilities.” To address this, we use truncated terms like “vulnerab” to capture all related forms. For LMA systems, we used keywords: “Agent” OR “LLM” OR “Large Language Model” OR “Collaborat” . To ensure comprehensive coverage of SE activities, we incorporated phase-specific keywords for each stage of the SDLC into our search queries: 

(1) 
Requirements Engineering : requirement, specification, stakeholder 

(2) 
Code Generation : software, code, coding, program 

(3) 
Quality Assurance : bug, fault, defect, fuzz, test, vulnerab, verificat, validat 

(4) 
Maintenance : debug, repair, review, refactor, patch, maintenanc 

We focus on four key phases of the SDLC: requirements engineering, code generation, quality assurance, and software maintenance. For each phase, the relevant SE keywords are combined using the OR operator to capture all variations. The final search query for each SDLC phase follows the format: [agent words] AND [SE words]. 

Following the guide of previous work ( Meline, 2006 ; Zhou et al., 2024 ; Van Dinter et al., 2021 ) , we design the following inclusion and exclusion criteria. In the first phase, we filtered out short papers (exclusion criterion 1) and removed duplicates (exclusion criterion 2). In the second phase, we manually screened each paper’s venue, title, and abstract, excluding items such as books, keynote speeches, panel summaries, technical reports, theses, tool demonstrations, editorials, literature reviews, and surveys (exclusion criteria 3 and 4). In the third phase, we conducted a full-text review to further refine relevant studies. Following Section 2.3 , we exclude papers that do not describe LMA systems (exclusion criterion 5). Papers that rely solely on LLMs using non-agent-based methods or single-agent approaches are excluded. Further, we focused on LMA systems powered by LLMs with strong planning capabilities, such as ChatGPT and LLaMA, excluding models like CodeBERT and GraphCodeBERT. Since the release of ChatGPT is in November 2022, we limited our review to papers published after this date (exclusion criterion 6). Furthermore, we excluded papers unrelated to software engineering (exclusion criterion 7) and those that mention LMA systems only in discussions or as future work, without presenting experimental results (exclusion criterion 8). After the third phase, we identified 41 primary studies directly relevant to our research focus. The search process is conducted on November 14th, 2024. 

✓ 
The paper must be written in English. 

✓ 
The paper must have an accessible full text. 

✓ 
The paper must adopt LMA techniques to solve software engineering-related tasks. 

✗ 
The paper has less than 5 pages. 

✗ 
Duplicate papers or similar studies authored by the same authors. 

✗ 
Books, keynote records, panel summaries, technical reports, theses, tool demos papers, editorials 

✗ 
The paper is a literature review or survey. 

✗ 
The paper does not utilize LMA systems, e.g., using a single LLM agent. 

✗ 
The paper is published before November 2022 (the release date of ChatGPT). 

✗ 
The paper does not involve software engineering related tasks. 

✗ 
The paper lacks experimental results and mentions LMA systems only in future work or discussions. 

Snowballing Search To expand our review, we conducted both backward and forward snowballing ( Wohlin, 2014 ) on the relevant papers identified in previous steps. This process involved examining the references cited by the relevant studies as well as publications that have cited these studies. We repeated the snowballing process until reaching a transitive closure fixed point, where no new relevant papers were found, resulting in an additional 30 papers identified. 

3.1. Requirements Engineering 

Requirements Engineering ( Van Lamsweerde, 2000 ; Lo, 2024 ) focuses on defining and managing software system requirements. This discipline is divided into several key stages to ensure requirements meet quality standards and align with stakeholder needs. These stages include elicitation, modeling, specification, analysis, and validation ( Hickey and Davis, 2004 ; Christel and Kang, 1992 ) . 

Elicitron ( Ataei et al., 2024 ) is an LMA framework that focuses specifically on the elicitation stage. It utilizes LLM-based agents to represent a diverse array of simulated users. These agents engage in simulated product interactions, providing insights into user needs by articulating their actions, observations, and challenges. MARE ( Jin et al., 2024 ) is an LMA framework that covers multiple phases of requirements engineering, including elicitation, modeling, verification, and specification. It employs five distinct agents, i.e., stakeholder, collector, modeler, checker, and documenter, performing nine actions to help generate high-quality requirements models and specifications. Sami et al. ( Sami et al., 2024b ) propose another LMA framework to generate, evaluate, and prioritize user stories through a collaborative process involving four agents: product owner, developer, quality assurance (QA), and manager. The produce owner generates user stories and initiates prioritization. The QA agent assesses story quality and identifies risks, while the developer prioritizes based on technical feasibility. Finally, the manager synthesizes these inputs and finalizes prioritization after discussions with all agents 

3.2. Code Generation 

Code generation ( Herrington, 2003 ; Budinsky et al., 1996 ) has consistently been a longstanding focus of software engineering research, aiming to automate coding tasks to boost productivity and minimize human error. 

A prominent multi-agent setup for code generation typically on role specialization and iterative feedback loops to optimize collaboration among agents. We summarize the common roles identified in the literature, including the Orchestrator, Programmer, Reviewer, Tester, and Information Retriever . 

The Orchestrator acts as the central coordinator, managing high-level planning and ensuring smooth task execution across all agents. Its responsibilities include defining high-level strategic goals, breaking them into actionable sub-tasks, delegating these tasks to the appropriate agents, monitoring progress, and ensuring that workflows align with overall project objectives ( Li et al., 2024c ; Zhang et al., 2024a ; Ishibashi and Nishimura, 2024 ; Zan et al., 2024 ; Cai et al., 2023 ; Phan et al., 2024 ; Li et al., 2024a ; Josifoski et al., 2023 ) . For instance, PairCoder ( Zhang et al., 2024a ) features a Navigator agent that interprets natural language descriptions to create high-level plans outlining solutions and key implementation steps. The Driver agent then follows these plans to handle code generation and refinement. The Self-Organized Agents (SoA) framework ( Ishibashi and Nishimura, 2024 ) employs a hierarchical design, with Mother agents managing high-level abstractions and delegating subtasks to specialized Child agents. In CODES ( Zan et al., 2024 ) , the Orchestrator role is performed by the RepoSketcher, which converts high-level natural language requirements into a repository sketch. This sketch outlines the project structure, including directories, files, and inter-file dependencies. The RepoSketcher then delegates tasks to the FileSketcher and SketchFiller, ensuring the efficient and seamless creation of a complete, functional code repository. 

During the implementation phase, the process typically begins with the Programmer, who is responsible for writing the initial version of the code. Once the initial code is produced, roles like the Reviewer and Tester step in to evaluate it, providing constructive feedback on quality, functionality, and adherence to requirements. This feedback initiates an iterative cycle, where the Programmer refines the code or the Debugger resolves identified issues, ensuring that the final code meets the desired standards and performs as expected ( Mathews and Nagappan, 2024 ; Wang et al., 2024b ; Olausson et al., 2023 ; Chen et al., 2023b ; Le et al., 2024 ; Liu et al., 2023 ; Lei et al., 2024a ; Lin et al., 2024a ; Dong et al., 2023 ) . For example, INTERVENOR ( Wang et al., 2024b ) pairs a Code Learner with a Code Teacher. The Code Learner generates the initial code and then compiles it to evaluate its correctness. If issues are identified, the Code Teacher analyzes the bug reports and the buggy code, subsequently providing repair instructions to address the errors. Self-repair ( Olausson et al., 2023 ) and TGen ( Mathews and Nagappan, 2024 ) refine code by utilizing feedback obtained from running pre-defined test cases. 

When predefined test cases are unavailable, the Tester can generate a variety of test cases, ranging from common scenarios to edge cases. These tests help uncover subtle issues that might otherwise go unnoticed and provide actionable feedback to guide subsequent refinement iterations ( Huang et al., 2023 ; Shinn et al., 2024 ; Ishibashi and Nishimura, 2024 ; Hu et al., 2024 ) . 

Some frameworks employ the Information Retriever to gather relevant information to assist code generation. For instance, Agent4PLC ( Liu et al., 2024d ) and MapCoder ( Islam et al., 2024 ) incorporate a Retrieval Agent tasked with sourcing examples of similar problems and extracting related knowledge. This agent provides essential contextual information and references tailored to the user’s input, ensuring that solutions are well-informed and adhere to domain-specific best practices. Similarly, CodexGraph ( Liu et al., 2024b ) employs a translation agent to facilitate interaction with graph databases, which are built using static analysis to extract code symbols and their relationships. By converting user queries into graph query language, this agent enables precise and structured information retrieval, enhancing the capability of LLM-based agents to navigate and utilize code repositories effectively. 

Agent Forest ( Li et al., 2024d ) adopts a different paradigm instead of role specialization. Instead, it utilizes a sampling-and-voting framework, where multiple agents independently generate candidate outputs. Each output is then evaluated based on its similarity to the others, with a cumulative similarity score calculated for each. The output with the highest score—indicating the greatest consensus among the agents—is selected as the final solution. 

3.3. Software Quality Assurance 

In this subsection, we review related work on testing, vulnerability detection, bug detection, and fault localization, with a focus on how LMA systems are being employed to enhance software quality assurance processes. 

Testing. Fuzz4All ( Xia et al., 2024 ) generates testing input for software systems across multiple programming languages. In this framework, a distillation agent reduces user input while a generation agent creates and mutates inputs. AXNav ( Taeb et al., 2024 ) is designed to automate accessibility testing. It interprets natural language test instructions and executes accessibility tests, such as VoiceOver, on iOS devices. AXNav includes a planner agent, an action agent, and an evaluation agent. WhiteFox ( Yang et al., 2023a ) is a fuzzing framework that tests compiler optimizations. It uses two LLM-based agents: one extracts requirements from source code, and the other generates test programs. Additionally, LMA systems are employed for tasks such as penetration testing ( Deng et al., 2023 ) , user acceptance testing ( Wang et al., 2024e ) , and GUI testing ( Yoon et al., 2024 ) . 

Vulnerability Detection. GPTLens ( Hu et al., 2023 ) is an LMA framework for detecting vulnerabilities in smart contracts. The system includes LLM-based agents acting as auditors, each independently identifying vulnerabilities. A critic agent then reviews and ranks these vulnerabilities, filtering out false positives and prioritizing the most critical ones. MuCoLD ( Mao et al., 2024 ) assigns roles like tester and developer to evaluate code. Through discussions and iterative assessments, the agents reach a consensus on vulnerability classification. Widyasari et al. ( Widyasari et al., 2024 ) introduces a cross-validation technique, where multiple LLM’s answer is validated against each other. 

Bug Detection. Intelligent Code Analysis Agent (ICAA) ( Fan et al., 2023 ) is used for bug detection in static code analysis. The agents have access to tools like web search, static analysis, and code retrieval tools. A Report Agent generates bug reports, while a False Positive Pruner Agent refines these reports to reduce false positives. Additionally, ICAA includes Code-Intention Consistency Checking, which ensures the code aligns with the developer’s intended functionality by analyzing code comments, documentation, and variable names. 

Fault Localiztion. RCAgent ( Wang et al., 2023b ) performs root cause analysis in cloud environments by using LLM-based agents to collect system data, analyze logs, and diagnose issues. AgentFL ( Qin et al., 2024 ) breaks down fault localization into three phases. The Comprehension Agent identifies potential fault areas, the Navigation Agent narrows down the codebase search, and the Confirmation Agent uses debugging tools to validate the faults. 

3.4. Software Maintenance 

In this subsection, we explore related work on debugging and code review, highlighting how LMA systems contribute to automating and improving software maintenance processes. 

Debugging. Debugging involves identifying, locating, and resolving software bugs. Several frameworks, including MASAI ( Arora et al., 2024 ) , MarsCode ( Liu et al., 2024a ) , AutoSD ( Kang et al., 2023 ) , and others ( Tao et al., 2024 ; Ma et al., 2024 ; Chen et al., 2024a ; Lei et al., 2024b ) , follow a structured process consisting of stages like bug reproduction, fault localization, patch generation, and validation. Specialized agents are typically responsible for each stage. FixAgent ( Lee et al., 2024 ) includes a debugging agent and a program repair agent that work together to iteratively fix code by analyzing both errors and repairs. The system refines fault localization by incorporating repair feedback. The agents also articulate their thought processes, improving context-aware debugging. The MASTER framework ( Yang et al., 2024 ) employs three specialized agents. The Code Quizzer generates quiz-like questions from buggy code, the Learner proposes solutions, and the Teacher reviews and refines the Learner’s responses. AutoCodeOver ( Zhang et al., 2024c ) uses an agent for fault localization via spectrum-based methods, collaborating with others to refine patches using program representations like abstract syntax trees. SpecRover ( Ruan et al., 2024 ) extends AutoCodeOver by improving program fixes through iterative searches and specification analysis based on inferred code intent. ACFIX ( Zhang et al., 2024b ) targets access control vulnerabilities in smart contracts, focusing on Role-Based Access Control. It mines common RBAC patterns from over 344,000 contracts to guide agents in generating patches. DEI ( Zhang et al., 2024f ) resolves GitHub issues by using a meta-policy to select the best solution, integrating and re-ranking patches generated by different agents for improved issue resolution. SWE-Search ( Antoniades et al., 2024 ) consists of three agents: the SWE-Agent for adaptive exploration, the Value Agent paired with a Monte Carlo tree search module for iterative feedback and utility estimation, and the Discriminator Agent for collaborative decision-making through debate. RepoUnderstander ( Ma et al., 2024 ) constructs a knowledge graph for a full software repository and also uses Monte Carlo tree search to assist in understanding complex dependencies. 

Code Review. Rasheed et al. ( Rasheed et al., 2024a ) developed an automated code review system that identifies bugs, detects code smells, and provides optimization suggestions to improve code quality and support developer education. This system uses four specialized agents focused on code review, bug detection, code smells, and optimization. Similarly, CodeAgent ( Tang et al., 2024 ) performs code reviews with sub-tasks such as vulnerability detection, consistency checking, and format verification. A supervisory agent, QA-Checker, ensures the relevance and coherence of interactions between agents during the review process. 

Test Case Maintenance. Lemner et al. ( Lemner et al., 2024 ) propose two multi-agent architectures to predict which test cases need maintenance after source code changes. These agents perform tasks including summarizing code changes, identifying maintenance triggers, and localizing relevant test cases. 

3.5. End-to-end Software Development 

End-to-end software development encompasses the entire process of creating a software product. While conventional code generation is often limited to producing isolated components such as functions, classes, or modules, end-to-end development starts from high-level software requirements and progresses through design, implementation, testing, and ultimately delivering a fully functional and ready-to-use product. 

In practice, developers and stakeholders typically adopt established software process models to guide collaboration, such as Agile ( Cohen et al., 2004 ) and Waterfall ( Petersen et al., 2009 ) . Similarly, the design of LMA systems for end-to-end software development draws inspiration from these software process models. The development process is organized into distinct phases, such as requirements gathering, software design, implementation, and testing. Each phase is managed by specialized agents with domain expertise. 
Figure 1. Multi-Agent Systems in Software Development: Waterfall vs. Agile Models 
It is important to note that works such as FlowGen ( Lin et al., 2024a ) and Self-Collaboration ( Dong et al., 2023 ) emulate various software process models. However, their experiments focus on generating code segments rather than delivering fully developed software products. As a result, in this paper, these approaches are not considered to be designed for true end-to-end software development. 

Several works ( Qian et al., 2024c ; Hong et al., 2023 ; Zhang et al., 2024d ; Du et al., 2024 ; Zan et al., 2024 ; Sami et al., 2024a ; Rasheed et al., 2024b ; Holt et al., 2023 ) adopt the Waterfall model to automate software development. The Waterfall model used in these multi-agent methods organizes the software development process into distinct, sequential phases, where each stage must be completed before proceeding to the next. The primary phases typically include Requirement Analysis, Architecture Design, Code Development, Testing, and Maintenance. For instance, in MetaGPT ( Hong et al., 2023 ) , the Product Manager agent thoroughly analyzes user requirements. The Architect agent then transforms these requirements into detailed system design components. Subsequently, the Engineer implements the specified classes and functions as outlined in the design. Finally, the Quality Assurance Engineer creates and executes test cases to ensure rigorous code quality standards are met. These approaches emphasize a linear and sequential design process, ensuring structured progression and clear accountability at each stage. 

AgileCoder ( Nguyen et al., 2024 ) and AgileGen ( Zhang et al., 2024e ) adopt Agile process models for software development, emphasizing iterative development by breaking complex tasks into small, manageable increments. AgileCoder ( Nguyen et al., 2024 ) assigns Agile roles such as Product Manager and Scrum Master to facilitate sprint-based collaboration and development cycles. AgileGen enhances Agile practices with human-AI collaboration, integrating close user involvement to ensure alignment between requirements and generated code. A notable feature of AgileGen is its use of the Gherkin language to create testable requirements, bridging the gap between user needs and code implementation. 

While most methods rely on predefined roles and fixed workflows for software development, a few work ( Wang et al., 2024d ; Li et al., 2023 ; Lin et al., 2024b ) investing in dynamic process models. Think-on-Process (ToP) ( Lin et al., 2024b ) introduces a dynamic process generation framework. Since software development processes can vary significantly depending on project requirements, ToP moves beyond the limitations of static, one-size-fits-all workflows to enable more flexible and efficient development practices. Given a software requirement, this framework leverages LLMs to create tailored process instances based on their knowledge of software development. These instances act as blueprints to guide the architecture of the LMA system, adapting to the specific and diverse needs of different projects. Similarly, in MegaAgent ( Wang et al., 2024d ) , agent roles and tasks are not predefined but are generated and planned dynamically based on project requirements. Both ToP and MegaAgent highlight the shift from rigid, static workflows to dynamic, adaptive systems. These frameworks promise more efficient, flexible, and context-aware software development practices, aligning processes with project-specific requirements and complexities. 

Additionally, instead of focusing on the process model, several works ( Qian et al., 2024a ; Qian et al., 2024b ) explore leveraging experiences from past software projects to enhance new software development efforts. Co-Learning ( Qian et al., 2024a ) enhances agents’ software development abilities by utilizing insights gathered from historical communications. This framework fosters cooperative learning between two agent roles—instructor and assistant—by extracting and applying heuristics from their task execution histories. Building on this, Qian et al. ( Qian et al., 2024b ) propose an iterative experience refinement (IER) framework that enables agents to continuously adapt by acquiring, utilizing, and selectively refining experiences from previous tasks, improving agents’ effectiveness and collaboration in dynamic software development scenarios. 

4. Case Study 

To demonstrate the practical effectiveness of LMA systems, we conduct two case studies. Specifically, we utilize the state-of-the-art LMA framework, ChatDev ( Qian et al., 2024c ) , to autonomously develop two classic games: Snake and Tetris. ChatDev structures the software development process into three phases: designing, coding, and testing. ChatDev employs specialized roles, including CEO, CTO, programmer, reviewer, and tester. ChatDev’s agents are powered by GPT-3.5-turbo 4 4 4 https://platform.openai.com/docs/models/gp#gpt-3-5-turbo . The temperature setting controls the randomness and creativity of the GPT-3.5’s responses. Following the original ChatDev setting, we set the temperature of GPT-3.5-turbo as 0.2. 

4.1. Snake Game 

For the Snake game, we provide the following prompt to ChatDev to generate the game: 

While the first attempt to generate the Snake game was unsuccessful, we resubmitted the same prompt to ChatDev, and the second attempt successfully produced a playable version. ChatDev also generated a detailed manual that included information on dependencies, step-by-step instructions for running the game, and an overview of its features. Figure 2 displays the graphical user interfaces (GUIs) of the generated Snake game, showing the starting state, in-game state, and game-over state. The development process was consistently efficient, taking an average of 76 seconds and costing $0.019. Upon playing the game, we confirmed that it fulfilled all the requirements outlined in the prompt. 

4.2. Tetris Game 

We present the following prompt to ChatDev to guide the generation of the Tetris game: 
Figure 2. Screen shots of the Snake Game generated by ChatDev. 
During development, ChatDev faced challenges in producing functional gameplay across the first nine attempts. Notice that the same prompt was used for each run. On the tenth attempt, ChatDev successfully produced a Tetris game that met most of the prompt requirements, as shown in Figure 3 . The figure illustrates the game’s key states: the starting state, in-game states, and game-over state. However, the game still lacks the core functionality to remove completed rows, as demonstrated in the third subplot of Figure 3 . Overall, the development process remained efficient, with an average time of 70 seconds and a cost of $0.020 per attempt. 
Figure 3. Screen shots of the Tetris Game generated by ChatDev. 
Summary of Findings . From our case studies, current LMA systems demonstrate strong performance in reasonably complex tasks like developing a Snake game. The generated Snake game meets all requirements in the prompt within just a few iterations. The process was efficient and cost-effective, with an average completion time of 76 seconds and a cost of $0.019 per attempt. These results emphasize the suitability of LMA systems for moderately complex software engineering tasks. However, when tasked with more complex challenges like developing a Tetris game, ChatDev successfully generates a playable Tetris game only by the tenth attempt. The game still lacks the core functionality, i.e., removing completed rows. This highlights the limitations of current LMA systems in handling more complex tasks that require deeper logical reasoning and abstraction. Nevertheless, development remains efficient and cost-effective, averaging 70 seconds and $0.020 per run, making the system a promising tool for rapid prototyping. 

5. Research Agenda 

Previous research has laid the groundwork for the exploration of LMA systems in software engineering, yet this domain remains in its nascent stages, with many critical challenges awaiting resolution. In this section, we outline our perspective on these challenges and suggest research questions that could advance this burgeoning field. As illustrated in Figure 4 , we envision two phases for the development of LMA systems in software engineering. We discuss each of these phases below and suggest a series of research questions that could form the basis of future research projects. 
Figure 4. Research Agenda for LLM-Based Multi-Agent Systems in Software Engineering 
5.1. Phase 1: Enhancing Individual Agent Capabilities 

Indeed, the effectiveness of an LMA system is closely linked to the capabilities of its individual agents. This first phase is dedicated to improving these agents’ skills, with a particular focus on adaptability and the acquisition of specialized skills in SE. The potential of individual LLM-based agents in SE is further explored through our initial research questions: 

(1) 
What SE roles are suitable for LLM-based agents to play and how can their abilities be enhanced to represent these roles? 

(2) 
How to design an effective, flexible, and robust prompting language that enhances LLM-based agents’ capabilities? 

5.1.1. Refining Role-Playing Capabilities in Software Engineering 

The role-playing capabilities of LLM-based agents are pivotal within LMA systems ( Wang et al., 2023d ) . To address the complexity of software engineering tasks, we need specialized agents capable of adopting diverse roles to tackle intricate challenges throughout the software development lifecycle. 

Current State. Existing LMA systems, such as ChatDev ( Qian et al., 2024c ) , MetaGPT ( Hong et al., 2023 ) , and AgileCoder ( Nguyen et al., 2024 ) , effectively simulate roles like generic software developers and product managers. The agents in these systems rely on general-purpose LLMs such as ChatGPT. Although LLMs like ChatGPT exhibit strong programming skills, they still lack the nuanced expertise required in SE ( Hou et al., 2024 ) . This limitation hampers their ability to simulate other SE-specific roles. For example, roles involving vulnerability detection or security auditing require a deep understanding of security protocols, threat modeling, and the latest vulnerabilities. However, multiple studies have identified deficiencies in ChatGPT’s ability to accurately detect and repair vulnerabilities ( Fu et al., 2023 ; Sridhara et al., 2023 ; Chen et al., 2024b ) . This shortcoming underscores the need to integrate domain-specific expertise into LLMs to better support specialized software engineering roles. 

Opportunities. To address this limitation, we propose a structured and actionable three-step approach encompassing the identification, assessment, and enhancement of role-playing abilities, which are: 

Step 1: Identifying and Prioritizing Key SE Roles. 

Step 2: Assessing LLM-Based Agents’ Competencies Against Role Requirements. 

Step 3: Enhancing Role-Playing Abilities Through Targeted Training. 

The first step focuses on identifying key SE roles, prioritizing those with high industry demand and the potential to substantially boost productivity. This involves: 

(1) 
Market Analysis : To begin, we embark on a comprehensive market analysis. It is crucial to assess not only the current trends and needs within the SE sector but also to anticipate future shifts influenced by the integration of LLM-based agents. This analysis involves leveraging various resources such as market reports, job postings, industry forecasts, and technology trend analyses. Platforms like LinkedIn Talent Insights 5 5 5 https://www.linkedin.com/products/linkedin-talent-insights/ , Gartner reports 6 6 6 https://www.gartner.com/en/products/special-reports , and Stack Overflow Developer Surveys 7 7 7 https://survey.stackoverflow.co/2024/ may also offer valuable data to inform this assessment. The focus should be on identifying roles that are in high demand and demonstrate rapid growth, especially those requiring specialized skills not typically found among generalist developers. For example, machine learning engineers or cloud architects. Additionally, positions where LLM-based agents could significantly enhance productivity, reduce costs, or accelerate innovation should be evaluated. A key component of this analysis should be determining whether the market has already begun shifting away from recruiting humans for tasks that LLM-based agents can perform, such as routine coding or simple bug fixing. Identifying these trends will help distinguish between roles that are still in demand and those where LLMs have reduced the need for human expertise. 

(2) 
Stakeholder Engagement : Engaging comprehensively with a diverse group of stakeholders is essential. This process validates the findings from the market analysis and ensures that the selected roles align with real-world needs. It involves consulting industry professionals who have hands-on experience in the identified roles. This engagement can provide practical insights and challenges associated with these positions. Collaboration with HR departments from leading technology companies is also important. It helps gather perspectives on current hiring trends, skill shortages, and the most sought-after competencies. Additionally, academic experts and researchers can offer forward-thinking views on emerging technologies and methodologies. By incorporating feedback from these various sources, the selection of key roles becomes more robust to reflect both current industry demands and future directions. 

(3) 
Value Addition Modeling : The next crucial step is value addition modeling ( Mendes et al., 2018 ) , which evaluates the potential advantages that LLM-based agents could bring to each prioritized role. This process involves constructing detailed, data-driven models to analyze key performance indicators such as efficiency improvements, cost reductions, quality enhancements, and the acceleration of innovation resulting from the integration of agents. Pilot projects can be deployed to gather empirical data on these metrics when LLM-based agents are applied to specific tasks. Important factors to consider include the automation of repetitive tasks, the augmentation of human capabilities, and the inclusion of new functionalities that were previously unattainable. It is important to note that the value added by LLM-based agents can differ significantly across different domains; for example, roles in software development may prioritize automation, whereas domains like systems architecture might see more value in LLMs augmenting complex decision-making around resource allocation or performance optimization, where human expertise and contextual understanding remain essential. By quantifying these value propositions, organizations can allocate resources more strategically to roles where LLM-based agents are likely to yield the highest return on investment. 

The second step involves understanding the limitations of LLM-based agents relative to the demands of the identified SE roles: 

(1) 
Competency Mapping : Competency mapping ( Kaur and Kumar, 2013 ) entails developing comprehensive competency frameworks for each specialized role. These frameworks define the essential skills, knowledge areas, and competencies required, encompassing both technical and soft skills. For instance, technical skills might encompass proficiency in specific programming languages, tools, methodologies, and domain-specific knowledge. For a machine learning engineer, this would include expertise in algorithms, data preprocessing, model training, and tools such as TensorFlow 8 8 8 https://www.tensorflow.org/ or PyTorch 9 9 9 https://pytorch.org/ . Soft skills include skills like problem-solving, critical thinking, and collaboration. Clearly outlining these competencies creates a benchmark against which the agents’ abilities can be measured. 

(2) 
Performance Evaluation : The next phase is performance evaluation, which involves designing or selecting tasks that closely replicate the real-world challenges associated with each role. These tasks should be practical and scenario-based to accurately gauge the agents’ capabilities. They should assess a wide range of competencies, from technical execution to critical thinking. For example, in evaluating a DevOps engineer, the agent might be tasked with automating a deployment pipeline using tools like Jenkins 10 10 10 https://www.jenkins.io/ or Docker 11 11 11 https://www.docker.com/ , or troubleshooting a continuous integration failure. Such tasks allow for a thorough assessment of both technical and soft skills. 

(3) 
Gap Analysis : This step compares the agents’ outputs with the expected outcomes for each task. Key areas where the agents underperform–such as misunderstanding domain-specific terminology, neglecting security best practices, or failing to optimize code–are identified and documented. This analysis emphasizes both the agents’ strengths and weaknesses, offering valuable insights into recurring patterns of errors or misconceptions. 

(4) 
Expert Consultation and Iterative Refinement : To further refine the evaluation process, expert consultation and iterative refinement are essential. By engaging with SE professionals who specialize in the assessed roles, qualitative feedback on the agent’s performance can be obtained. These experts provide insights into subtle nuances that may not be captured through quantitative metrics. For instance, while the agent’s code may work, it might not follow best practices or address scalability. This feedback helps refine evaluation methods, update competency frameworks, and uncover deeper issues in the agent’s understanding. 

The final step involves tailoring the LLM-based agents to effectively represent the identified SE roles through specialized training and prompt engineering. : 

(1) 
Curating Specialized Training Data: At first, this involves creating training datasets that reflect the unique requirements of each specific role. A comprehensive corpus should be built from a variety of sources, including technical documentation such as API guides, technical manuals, and user guides to provide in-depth knowledge of specific technologies. It is also important to incorporate academic and industry research papers, case studies, and whitepapers to capture the latest developments, best practices, and theoretical foundations. Additionally, discussions from forums and software Q&A sites like Stack Overflow 12 12 12 https://stackoverflow.com/ , Reddit 13 13 13 https://www.reddit.com/ , and specialized industry forums can provide practical problem-solving approaches and real-world challenges faced by professionals. 

(2) 
Fine-tuning the LLM: After preparing the data, the curated datasets are used to fine-tune the LLM-based agents. Advanced techniques like parameter-efficient fine-tuning (PEFT) ( Liu et al., 2022 ) are often employed to optimize both efficiency and accuracy. 

(3) 
Designing Customized Prompts: A key step is designing prompts tailored to improve the agents’ role adaptability. These prompts should clearly define the role, tasks, and goals to ensure the agent understands the requirements. For instance, in a cybersecurity analyst role, the prompt should outline specific security protocols, potential vulnerabilities, and compliance standards. Contextual instructions, including relevant background, constraints, and examples, help the agent grasp task nuances. Creating a library of effective prompts for various scenarios can also serve as reusable templates for future tasks. 

(4) 
Continuous Learning and Adaptation: To keep agents aligned with industry developments, continuous adaptation mechanisms are essential. Training data should be regularly updated, and models may be retrained to incorporate new technologies, best practices, and trends in software engineering. Monitoring systems can track agent performance over time, enabling proactive adjustments and continuous improvement. Additionally, agents should be guided to consistently reference the latest documentation and standards to ensure their outputs remain relevant and accurate. 

While LMA roles may overlap with traditional software engineering roles, it is important to recognize that they are not necessarily the same, as LMA roles often involve specialized, collaborative tasks suited for agent-based systems. By systematically identifying key roles, assessing agent competencies, and enhancing their capabilities through targeted fine-tuning, we aim to significantly improve the effectiveness of LLM-based agents in specialized SE roles. 

5.1.2. Advancing Prompts through Agent-oriented Programming Paradigms 

Effective prompts are crucial for the performance of LLM-based agents. However, creating such prompts is challenging due to the need for a framework that is versatile, effective, and robust across diverse scenarios. Natural language, while flexible, often contains ambiguities and inconsistencies that LLMs may misinterpret. Natural language is inherently designed for human communication, where human communication relies on shared context and intuition that LLMs lack. In contrast, LLMs interpret text based on statistical patterns from large datasets, which may lead to different interpretations than those intended for humans ( Sun et al., 2024 ; Zeng et al., 2022 ) . This highlights the need for a specialized prompting language designed to augment the cognitive functions of LLM-based agents and treats LLMs as the primary audience. Such a language can minimize ambiguities and ensure clear instructions, resulting in more reliable and accurate outputs. 

Current State. Multiple prompting frameworks are released to facilitate the usage of LLMs. For example, DSPy ( Khattab et al., 2023 ) and Vieira ( Li et al., 2024b ) enable fully automated generation of prompts. AutoGen ( Wu et al., 2023a ) and LangChain ( Mavroudis, 2024 ) support retrieval-augmented generation (RAG) ( Gao et al., 2023 ) and agent-based workflows. However, these frameworks are still human-centered. They often prioritize human readability and developer convenience. As a result, there is a lack of research on a language that treats LLMs as the primary audience for prompts. 

Opportunities. Agent-oriented programming (AOP) ( Shoham, 1993 ) offers a promising foundation for this approach. Similar to how Object-Oriented Programming (OOP) ( Wegner, 1990 ) organize objects, AOP treats agents as fundamental units, focusing on their reasoning, objectives, and interactions. An AOP-based prompting language could enable the precise expression of complex tasks and constraints, allowing LLM-based agents to perform their roles with greater efficiency and accuracy. Extending this concept to Multi-Agent-Oriented Programming (MAOP) ( Boissier et al., 2020 ; Bordini et al., 2009 ) allows for the creation of systems where multiple LLM-based agents can collaborate, communicate, and adapt to evolving contexts. By explicitly defining agent behaviors, communication patterns, and task hierarchies, we can reduce ambiguity, mitigate hallucinations, and improve task execution in LMA systems. 

Further, such a prompting language must be expressive enough to handle diverse and complex tasks, yet simple enough for users to easily adopt. Conversely, overly simplified languages may lack the expressive power needed to represent complex software engineering workflows. A complex language that introduce a steep learning curve due to their syntax, hinder adoption, especially for users who require simpler interfaces for prompt creation and modification. Balancing functionality and usability will be another key research question to its success. 

Additionally, this process may involve tailoring prompts specifically for different LLM models and their versions, as variations in model architectures, training data, and capabilities can affect how they interpret and respond to prompts. What works effectively for one model may not perform as well for another, necessitating careful adjustments. Current prompting languages lack mechanisms to easily adapt prompts across models, requiring manual adjustments and experimentation to achieve consistent performance. 

While AOP-based prompting may not be the final solution, it represents an important step toward developing an AI-oriented language with grammar tailored specifically for LLMs. This new approach could further refine communication with LLM-based agents, reducing misinterpretation and significantly enhancing overall performance. 

5.2. Phase Two: Optimizing Agent Synergy 

In Phase Two, the spotlight turns towards optimizing agent synergy, underscoring the importance of collaboration and how to leverage the diverse strengths of individual agents. This phase delves into both the internal dynamics among agents and the role of external human intervention in enhancing the efficacy of the LMA system. Key research questions guiding this phase include: 

(3) 
How to best allocate tasks between humans and LLM-based agents? 

(4) 
How can we quantify the impact of agent collaboration on overall task performance and outcome quality? 

(5) 
How to scale LMA systems for large-scale projects? 

(6) 
What industrial organization mechanisms can be applied to LMA systems? 

(7) 
What strategies allow LMA systems to dynamically adjust their approach? 

(8) 
How to ensure security among private data sharing within LMA systems? 

5.2.1. Human-Agent Collaboration 

Optimally distributing tasks between humans and LMA agents to leverage their respective strengths is essential. Humans bring unparalleled creativity, critical thinking, ethical judgment, and domain-specific knowledge ( Markauskaite et al., 2022 ) . In contrast, LLM-based agents excel at rapidly processing large datasets, performing repetitive tasks with high accuracy, and detecting patterns that might elude human observers. 

Current State. Several LMA systems incorporate human-in-the-loop designs. For instance, AISD ( Zhang et al., 2024d ) involves human input during requirement analysis and system validation, where users provide feedback on use cases, system designs, and prototypes. Similarly, MARE ( Jin et al., 2024 ) leverages human assessment to refine generated requirements and specifications. Although these works demonstrate the feasibility of human contributions, key research questions, including optimizing human roles, enhancing feedback mechanisms, and identifying appropriate intervention points, are still underexplored. 

Opportunity. Developing role-specific guidelines that outline when and how human intervention should occur is essential. These guidelines should assist in identifying critical decision points where human judgment is indispensable, such as ethical considerations, conflict resolution, ambiguity handling, and creative problem-solving. For example, ethical decisions necessitate human oversight to ensure alignment with societal norms and values, and conflict resolution may req
