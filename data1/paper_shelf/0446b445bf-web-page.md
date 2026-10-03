# AFlow: Automating Agentic Workflow Generation

Source URL: https://arxiv.org/html/2410.10762v4

AFlow : Automating Agentic Workflow Generation 
Jiayi Zhang 1 1 footnotemark: 1 Affiliation: DeepWisdom Affiliation: The Hong Kong University of Science and Technology (Guangzhou) Jinyu Xiang † † thanks: These authors contributed equally to this work. Affiliation: DeepWisdom Zhaoyang Yu Affiliation: Renmin University of China Fengwei Teng Affiliation: Renmin University of China Xiong-Hui Chen Affiliation: Nanjing University Jiaqi Chen Affiliation: Fudan University Mingchen Zhuge Affiliation: King Abdullah University of Science and Technology Xin Cheng Affiliation: Renmin University of China Sirui Hong Affiliation: DeepWisdom Jinlin Wang Affiliation: DeepWisdom Bingnan Zheng Affiliation: Fudan University Bang Liu Affiliation: Université de Montréal & Mila Yuyu Luo 2 2 footnotemark: 2 Affiliation: The Hong Kong University of Science and Technology (Guangzhou) Affiliation: The Hong Kong University of Science and Technology Chenglin Wu † † thanks: Corresponding authors: Yuyu Luo (E-mail:yuyuluo@hkust-gz.edu.cn), Chenglin Wu (E-mail: alexanderwu@deepwisdom.ai) Affiliation: DeepWisdom Abstract 
Large language models (LLMs) have demonstrated remarkable potential in solving complex tasks across diverse domains, typically by employing agentic workflows that follow detailed instructions and operational sequences. However, constructing these workflows requires significant human effort, limiting scalability and generalizability. Recent research has sought to automate the generation and optimization of these workflows, but existing methods still rely on initial manual setup and fall short of achieving fully automated and effective workflow generation. To address this challenge, we reformulate workflow optimization as a search problem over code-represented workflows, where LLM-invoking nodes are connected by edges. We introduce AFlow , an automated framework that efficiently explores this space using Monte Carlo Tree Search, iteratively refining workflows through code modification, tree-structured experience, and execution feedback. Empirical evaluations across six benchmark datasets demonstrate AFlow ’s efficacy, yielding a 5.7% average improvement over state-of-the-art baselines. Furthermore, AFlow enables smaller models to outperform GPT-4o on specific tasks at 4.55% of its inference cost in dollars. The code is available at https://github.com/FoundationAgents/AFlow . 

1 Introduction 

Large Language Models (LLMs) have emerged as powerful tools for solving complex tasks across various domains, including code generation, data analysis, decision-making, and question answering ( Liu et al., 2024 ; Li et al., 2024a ; Zhu et al., 2024 ; Xie et al., 2024b ; Sun et al., 2024 ; Wang et al., 2024b ; Song et al., 2023 ; Xie et al., 2024a ; Zhong et al., 2024a ) . However, the rapid advancement of LLMs heavily relies on manually designed agentic workflows – structured sequences of LLM invocations accompanied by detailed instructions. Designing and refining these workflows requires significant human effort, which limits the scalability and adaptability of LLMs to new, complex domains and hinders their ability to transfer skills across diverse tasks ( Tang et al., 2024 ) . 

Recent efforts have focused on automating the discovery of effective agentic workflows to reduce the reliance on human intervention ( Khattab et al., 2024 ; Yüksekgönül et al., 2024 ; Liu et al., 2023 ; Hu et al., 2024 ) . Despite these advancements, full automation has not been achieved. For instance, Khattab et al. (2024) requires manual workflow setup before automated prompt optimization. Similarly, methods proposed by Yüksekgönül et al. (2024) and Zhuge et al. (2024) fail to capture the full diversity of workflows necessary for a wide range of tasks ( Yu et al., 2023 ; Yang et al., 2024b ; Sun et al., 2023 ) , as their optimization objectives struggle to represent the breadth of possible workflows. The inability to effectively model diverse workflow structures within these automated systems limits their utility and impact. ADAS ( Hu et al., 2024 ) represents workflows using code, achieving a relatively complete representation. However, due to the efficiency limitations of its linear heuristic search algorithm, ADAS struggles to generate effective workflows within a limited number of iterations. This highlights the need for more effective techniques to represent and automate the generation of agentic workflows, which would accelerate the application of LLMs across domains. 
Figure 1: Performance comparison with other methods. To assess the method’s performance, we employ various metrics across different datasets: solve rate for Math and GSM8K, F1 score for HotpotQA and DROP, and pass@1 for HumanEval and MBPP. Our AFlow (highlighted in yellow) consistently outperforms all automated workflow optimization and manually designed methods across all six benchmarks. 
In response to these challenges, we introduce an innovative framework for automatically generating agentic workflows. Our key idea is to model the workflow as a series of interconnected LLM-invoking nodes, where each node represents an LLM action and the edges define the logic, dependencies, and flow between these actions. This structure transforms the workflow into a vast search space, encompassing a wide variety of potential configurations. Our goal is to efficiently navigate this space, automatically generating optimized workflows that maximize task performance while minimizing human intervention. 

However, the diversity and complexity of tasks present significant challenges. Specifically, each task can have different requirements, operations, and dependencies, which makes it difficult to represent them in a unified yet flexible manner ( Chen et al., 2021 ; Cobbe et al., 2021 ; Yang et al., 2018 ; Luo et al., 2018 ) . Furthermore, the search space for possible workflows, comprising an immense number of code structures and node configurations, is virtually boundless, creating an additional challenge for efficient exploration and optimization. 

To address these challenges, we propose AFlow , a Monte Carlo Tree Search (MCTS)-based framework designed to systematically explore and discover optimal agentic workflows. AFlow represents workflows as flexible nodes connected by code-based edges, which encapsulate possible relationships such as logical flows, conditions, and dependencies. These edges allow the workflow to be modeled as a graph ( Zhuge et al., 2024 ) or network ( Liu et al., 2023 ) , offering a powerful structure for capturing complex interactions between LLM invocations. 

To enhance the search process and improve efficiency, AFlow introduces a novel concept of operators – predefined, reusable combinations of nodes representing common agentic operations (e.g., Ensemble, Review & Revise). These operators serve as foundational building blocks for constructing workflows and are integrated into the search space, ensuring that the exploration process leverages known patterns of effective agentic operations. 

AFlow employs the MCTS algorithm to navigate this infinite search space. The framework’s workflow optimization process incorporates several key innovations: a soft mixed-probability selection mechanism for node exploration, LLM-driven node expansion to introduce new possibilities, execution evaluation to assess workflow performance, and backpropagation of experience to refine future search iterations. This combination of techniques ensures that AFlow efficiently discovers workflows that adapt to the complexity of diverse tasks while reducing reliance on manual intervention. 

We make the following key contributions: (1) Problem Formulation : We formalize the workflow optimization problem, generalizing prior approaches as specific cases. This provides a unified framework for future research at both the node and workflow optimization levels. (2) AFlow : We introduce AFlow , an MCTS-based method that automatically discovers effective workflows across multiple domains with minimal human intervention. (3) Extensive Evaluation : We evaluate AFlow on six benchmark datasets: HumanEval, MBPP, MATH, GSM8K, HotPotQA, and DROP. AFlow outperforms manually designed methods by 5.7% and surpasses existing automated approaches by 19.5%. Notably, workflows generated by AFlow enable smaller LLMs to outperform larger models, offering better cost-performance efficiency, with significant implications for real-world applications. 

2 Related Work 
Agentic Workflow 
Agentic workflow and autonomous agents ( Zhuge et al., 2023 ; Hong et al., 2024a ; Zhang et al., 2024c ; Wang et al., 2023 ) represent two distinct paradigms of LLM application. The former completes tasks statically through predefined processes with multiple LLM invocations, while the latter solves problems dynamically through flexible autonomous decision-making. Compared to autonomous agents that require specific actions and decision patterns designed for the environment, agentic workflows can be constructed based on existing human domain experience and iterative refinement, offering higher potential for automated construction. 

Agentic workflows can be broadly categorized into general and domain-specific types. General workflows emphasize universal problem-solving approaches, such as ( Wei et al., 2022 ; Wang et al., 2022 ; Madaan et al., 2023 ; Wang et al., 2024a ) . Domain-specific workflows focus on building effective processes to solve domain-specific problems, such as code generation ( Hong et al., 2024b ; Ridnik et al., 2024 ; Zhong et al., 2024a ) , data analysis ( Xie et al., 2024b ; Ye et al., 2024 ; Li et al., 2024a ; Zhou et al., 2023 ) , mathematics ( Zhong et al., 2024b ; Xu et al., 2024 ) , question answering ( Nori et al., 2023 ; Zhou et al., 2024a ) . Existing work has manually discovered numerous effective agentic workflows, but it’s challenging to exhaust various tasks across different domains, further highlighting the importance of automated workflow generation and optimization. 
Automated Agentic Optimization 
Recent work aims to automate the design of agentic workflows, categorized into three types: automated prompt optimization, hyperparameter optimization, and automated workflow optimization. Prompt optimization ( Fernando et al., 2024 ; Yüksekgönül et al., 2024 ; Yang et al., 2024a ; Khattab et al., 2024 ) uses LLMs to optimize prompts within fixed workflows. Hyperparameter optimization ( Saad-Falcon et al., 2024 ) focuses on optimizing predefined parameters. While these approaches improve performance, they are limited in generalization to new tasks and often require moderate human effort for task-specific designs. 

Automated workflow optimization ( Li et al., 2024b ; Zhou et al., 2024b ; Zhuge et al., 2024 ; Hu et al., 2024 ) aims to optimize entire workflow structures, offering more potential for fully automated generation. Recent works explore diverse representations and methods. GPTSwarm ( Zhuge et al., 2024 ) uses graph structures with reinforcement learning, but struggles to represent workflows with conditional states due to graph structure limitations. ADAS ( Hu et al., 2024 ) utilizes code structures to represent workflows and stores historical workflows in a linear list structure, aligning closely with our goals. However, it is constrained by the efficiency of its search algorithm as it relies on overly simplistic representations of experiences in the searching process, making it challenging to discover effective workflows. 

AFlow also uses code to represent workflows, but goes further by providing a more fundamental structure called named node. This structure encompasses various LLM invocation parameters, allowing for more detailed workflow representation. We also introduce operators that implement predefined node combination functions. Simultaneously, AFlow employs a specially designed MCTS algorithm for automated workflow optimization, leveraging the tree-structured experience and execution feedback to efficiently discover effective workflows. 

3 Preliminary 

In this section, we will first formulate the automated agentic workflows generation problem in Section 3.1 and then discuss design considerations of our AFlow in Section 3.2 . For the core concept of this section, we provide an example explanation in Figure 2 . 
Figure 2: The example of node, operator, and edge. We demonstrate the optional parameters for Nodes, the structure of some Operators, and common representations of Edges. 
3.1 Problem Formulation 
Agentic Workflow 
We define an agentic workflow W as a series of LLM-invoking nodes connected by edges to define the exection orders, denoted as 𝒩 = { N 1 , N 2 , … , N i ​ … } \mathcal{N}=\{N_{1},N_{2},\ldots,N_{i}\ldots\} . Each node N i N_{i} represents a specific operation performed by an LLM and is characterized by the following parameters. The code abstraction of the node is shown in Appendix A.2 . 

• 
Model M M : The specific language model invoked at node N i N_{i} . 

• 
Prompt P P : The input or task description provided to the model at each node. 

• 
Temperature τ \tau : A parameter controlling the randomness of the LLM’s output at node N i N_{i} . 

• 
Output format F F : The format in which the model’s output is structured ( e.g. , xml, json, markdown, raw). The node in workflow should provide different output formats, inspired by the Tam et al. (2024) . 

Edge E E represent abstract structures defining node relationships, governing the sequence of execution. The edge E E can be represented via various structures, such as: 

• 
Graph Zhuge et al. (2024) : A flexible structure representing hierarchical, sequential, or parallel relationships between nodes, allowing for complex branching workflows. 

• 
Neural Network ( Liu et al., 2023 ) : A structure that can represent complex, non-linear relationships between nodes, allowing for adaptive and learnable workflows based on input and feedback. 

• 
Code ( Hu et al., 2024 ) : A comprehensive representation that can express linear sequences, conditional logic, loops, and incorporate graph or network structures, offering the most precise control over workflow execution for LLMs. 

While graph structures can represent workflow relationships, they require complex extensions (e.g., Petri nets, BPMN) beyond basic DAGs to naturally express parallel execution and conditional logic. Neural networks enable adaptive transitions but lack precise control over workflow execution. In contrast, code representation inherently supports all these relationships through standard programming constructs. Therefore, we adopt code as our primary edge structure to maximize expressivity. 
Automated Workflow Optimization 
Given a task T T and an evaluation function G G , the goal of workflow optimization is to discover a workflow W W that maximizes G ⁡ ( W , T ) G(W,T) . This can be formulated as a search process where an algorithm A A explores the search space 𝒮 \mathcal{S} to determine the optimal workflow configuration. The search space 𝒮 \mathcal{S} for a workflow optimization problem encompasses all possible configurations of node parameters and edge structures: 
𝒮 = { ( 𝒩 , E ) ∣ E ∈ ℰ } , \mathcal{S}=\{(\mathcal{N},E)\mid E\in\mathcal{E}\}, 
where 𝒩 = { N ( M , τ , P , F ) ∣ M ∈ ℳ , τ ∈ [ 0 , 1 ] , P ∈ 𝒫 , F ∈ ℱ } \mathcal{N}=\{N(M,\tau,P,F)\mid M\in\mathcal{M},\tau\in[0,1],P\in\mathcal{P},F\in\mathcal{F}\} , with ℳ , 𝒫 , ℱ , ℰ \mathcal{M},\mathcal{P},\mathcal{F},\mathcal{E} representing the sets of possible language models, prompts, output formats, and edge configurations, respectively. 

With this formulation, the workflow optimization problem can be expressed as: 
W \displaystyle W = A ⁡ ( 𝒮 , G , T ) , \displaystyle=A(\mathcal{S},G,T), W ∗ \displaystyle W^{*} = arg ​ max W ∈ 𝒮 ⁡ G ​ ( W , T ) , \displaystyle=\argmax_{W\in\mathcal{S}}G(W,T), 
where A A is the search algorithm that explores the search space 𝒮 \mathcal{S} , and W ∗ W^{*} is the optimal workflow configuration that maximizes the evaluation function G G for the given task T T . 

3.2 AFlow Overview 
Limitations of Previous Methods 
Previous approaches Yüksekgönül et al. (2024) ; Khattab et al. (2024) ; Zhuge et al. (2024) to workflow optimization have primarily been constrained by the limited scope of their search spaces, based on problem definition in Section 3.1 . Another related work, ADAS ( Hu et al., 2024 ) , searches in a larger space comprising a combination of prompts N ⁡ ( P , T ) N(P,T) and edges E E , but fails to discover effective workflows due to the efficiency limitations of its linear heuristic search algorithm. 
Figure 3: Overall AFlow framework : By setting a search space composed of nodes with only prompt parameters flexible, a given operator set, and a code representing edge, AFlow performs an MCTS-based search within this space. Through a variant of MCTS designed for workflow optimization, AFlow iteratively executes a cycle of Soft Mixed Probability Selection, LLM-Based Expansion, Execution Evaluation, and Experience Backpropagation until reaching the maximum number of iterations or meeting convergence criteria. Formulation 
To address the limitations of previous methods, we propose AFlow , a novel framework that leverages Large Language Models (LLMs) as optimizers within a variant of Monte Carlo Tree Search (MCTS) to search for optimal workflows. As discussed in Section 3.1 , edges can be represented in both graphs and code. To ensure AFLOW can explore the full range of possible agentic workflows, we represent nodes N and edges E through code. Specifically, as shown in Figure 3 , AFlow uses a variant of MCTS to iteratively explore the workflow search space, evaluate different configurations, and backpropagate experiences to refine the workflow optimization process. 

To enhance search efficiency in practice, we simplify the search space by fixing key parameters such as the model M M , temperature τ \tau , and format F F . This simplification allows AFlow to focus its search primarily on the code-represented edges E E and prompts. To navigate this still vast search space effectively, we introduce the concept of Operators . These Operators encapsulate common agentic operations (e.g., Ensemble, Review, Revise) by combining N N and E E into unified interfaces, thereby enabling more efficient utilization by AFlow . By employing these Operators, we achieve more efficient search and streamlined workflow generation. 

Formally, given a set of Operators 𝒪 \mathcal{O} that represents predefined node combinations, and an edge space ℰ \mathcal{E} represented through code, the optimization problem can be formalized as: 
𝒮 AFlow = { ( P 1 , … , P n , E , O 1 , … , O n ) ∣ P i ∈ 𝒫 , E ∈ ℰ , O i ∈ 𝒪 } \mathcal{S}_{\text{AFlow}}=\{(P_{1},\ldots,P_{n},E,O_{1},\ldots,O_{n})\mid P_{i}\in\mathcal{P},E\in\mathcal{E},O_{i}\in\mathcal{O}\}\\ (1) W ∗ = AFlow ​ ( 𝒮 AFlow , G , T ) W^{*}=\text{{\sc AFlow} }(\mathcal{S}_{\text{AFlow}},G,T) (2) Tasks Scope and Operations 
In this paper, we focus on applying AFlow to reasoning tasks with numerical evaluation functions. We extract common operations from existing literature and define them as part of the operator set 𝒪 \mathcal{O} . These operations include: (1) Generate, (2) Format, (3) Review and Revise Madaan et al. (2023) , (4) Ensemble Wang et al. (2022) , (5) Test Zhong et al. (2024a) , (6) Programmer, and (7) Custom as the default operator for basic node construction. The operator set 𝒪 \mathcal{O} can be easily expanded to enhance search efficiency for various tasks. Even without any predefined operators, AFlow can construct different workflow nodes using the basic Custom operator. The efficiency comparison between these approaches is detailed in Section 5.2 . For a comprehensive understanding of the operators, we provide their detailed structures in Appendix A.4 . 

4 The Design Details of AFlow 

The core concept of AFlow is to employ Large Language Models (LLMs) as optimizers within a Monte Carlo Tree Search (MCTS) variant to discover effective workflows. In our MCTS structure, each tree node represents a complete workflow rather than individual LLM-invoking node , enabling the discovery of universal solutions for classes of problems. The search process operates through an iterative cycle of soft mixed probability selection, LLM-based optimization expansion, execution evaluation, and experience backpropagation until reaching maximum iterations or convergence criteria. A simplified illustration is shown in Figure 3 , with detailed algorithm process and theoretical analysis presented in Appendix A.6 and Appendix G , respectively. 

Existing workflow optimization methods iteratively use past workflow structures to prompt LLMs to discover new structures. However, due to information loss during accumulation (as input tokens increase), this approach struggles to guide LLMs towards specific performance metrics. Combined with the vast search space of code, this reduces search efficiency. Our key idea is to leverage the tree structure of MCTS to preserve workflow-based exploration experiences in N m ​ a ​ x N_{max} rounds workflow optimization. When a workflow is revisited, we accurately reuse past successful experiences and avoid failures, enabling effective workflow generation and improving search efficiency. To prevent local optima, we introduce a special selection mechanism allowing generation from a blank template at any round. Next, we will introduce the complete process of AFlow , as shown in Algorithm 1 . 
Algorithm 1 Algorithm of AFlow : Detailed implementation 1: Evaluator G G , Dataset D D , Operators 𝒪 \mathcal{O} 2: Optimized Workflow W ∗ W^{*} 3: Initialize W 0 W_{0} , split D D into D V D_{V} and D T D_{T} 4: W ∗ ← W 0 W^{*}\leftarrow W_{0} 5: for i ​ t ​ e ​ r ​ a ​ t ​ i ​ o ​ n ← 1 iteration\leftarrow 1 to N m ​ a ​ x N_{max} do 6: w ​ o ​ r ​ k ​ f ​ l ​ o ​ w ← workflow\leftarrow Select(tree) ⊳ \triangleright Using soft mixed probability strategy 7: c ​ h ​ i ​ l ​ d . w ​ o ​ r ​ k ​ f ​ l ​ o ​ w ← child.workflow\leftarrow Expand( w ​ o ​ r ​ k ​ f ​ l ​ o ​ w workflow , 𝒪 \mathcal{O} ) ⊳ \triangleright LLM-based expansion 8: s ​ c ​ o ​ r ​ e ← score\leftarrow Evaluate( c ​ h ​ i ​ l ​ d . w ​ o ​ r ​ k ​ f ​ l ​ o ​ w child.workflow , G G , D V D_{V} ) ⊳ \triangleright Multiple runs for robustness 9: Backpropagate( c ​ h ​ i ​ l ​ d . w ​ o ​ r ​ k ​ f ​ l ​ o ​ w child.workflow , s ​ c ​ o ​ r ​ e score ) ⊳ \triangleright Update experience and scores 10: Update W ∗ W^{*} if improved 11: if ConvergenceCriteriaMet() then break 12: end if 13: end for 14: return W ∗ W^{*} 
Initialization AFlow begins with a template workflow W 0 W_{0} , which provides a framework for invoking nodes and operators. The code template, detailed in Appendix A.3 , allows the LLM optimizer to complete workflow simply by completing call functions. Prior to initiating the search process, we randomly partition the dataset into a validation set (20%) and a test set (80%), with the random seed fixed at 42. To optimize computational efficiency, AFlow then executes the blank template five times on the validation dataset. From these executions, we select a subset of problems that exhibit high variance in scores, which becomes the final validation set. 

Selection Our algorithm forms the initial workflow by evaluating an empty workflow on the validation set. And then continuously select workflows based on a soft mixed probability selection strategy. cWe propose this strategy for workflow optimization: combining uniform and score-based weighted probability distributions to select from top-k workflows and the initial workflow, where including the initial workflow ensures persistent exploration capability while avoiding local optima. The formula for this selection strategy is as follows: 
P mixed ​ ( i ) = λ ⋅ 1 n + ( 1 − λ ) ⋅ exp ⁡ ( α ⋅ ( s i − s max ) ) ∑ j = 1 n exp ⁡ ( α ⋅ ( s j − s max ) ) , P_{\text{mixed}}(i)=\lambda\cdot\frac{1}{n}+(1-\lambda)\cdot\frac{\exp(\alpha\cdot(s_{i}-s_{\text{max}}))}{\sum_{j=1}^{n}\exp(\alpha\cdot(s_{j}-s_{\text{max}}))}, (3) 
where n is the number of workflows , s i is workflow i ’s score , s max is the maximum score , α ( 0.4 ) controls score influence, and λ ( 0.2 ) balances exploration and exploitation . \text{where }n\text{ is the number of workflows},s_{i}\text{ is workflow }i\text{'s score},s_{\text{max}}\text{ is the maximum score},\alpha\text{ }(0.4)\\ \text{ controls }\text{score influence, and }\lambda\text{ }(0.2)\text{ balances exploration and exploitation}. 

Expansion In the expansion phase, we employ an LLM as an optimizer to create new workflows and the optimize prompt is illustrated in Appendix A.1 . The optimizer leverages the selected workflow’s experience to generate new prompts or modify node connections by altering code, resulting in new workflows. Specifically, to maximally uncover insights from past iterations, the experience includes all modifications and their corresponding improvements or failures on the selected workflow, along with precise logs of predictions and expected output. 

Evaluation AFlow directly executes workflows to get feedback due to explicit evaluation functions in reasoning tasks. We test each generated workflow 5 times on the validation set, computing mean and standard deviation. While this increases per-iteration cost, it provides more accurate feedback for the optimizer. This precision enhances search efficiency, ultimately reducing the number of iterations required to reach an effective solution. 

Backpropagation After execution, we record: (1) the workflow’s performance, (2) the optimizer’s modification of its parent workflow, and (3) optimization success relative to its parent. This information is stored in experience and propagated back to the parent workflow, while the performance score is added to the global record for selection. 

Terminal Condition We implement early stopping to reduce unnecessary execution costs: the process terminates if the top-k average score shows no improvement for n n consecutive rounds, or after N N total rounds otherwise. See Appendix A.6 for algorithmic details. 

5 Experiments 

5.1 Experimental Setup 

Datasets We utilized six public benchmarks for our experiments. Following established practices ( Saad-Falcon et al., 2024 ; Hu et al., 2024 ) in workflow optimization, we divide the data into validation and test sets using a 1:4 ratio. Specifically, we use the full datasets for GSM8K ( Cobbe et al., 2021 ) , HumanEval ( Chen et al., 2021 ) , and MBPP ( Austin et al., 2021 ) . For HotpotQA ( Yang et al., 2018 ) and DROP ( Dua et al., 2019 ) , we randomly select 1,000 samples each, in line with ( Hu et al., 2024 ; Shinn et al., 2023 ) . For the MATH ( Hendrycks et al., 2021 ) dataset, we follow ( Hong et al., 2024a ) in selecting 617 problems from four typical problem types (Combinatorics & Probability, Number Theory, Pre-algebra, Pre-calculus) at difficulty level 5. 

Baselines We compare workflow discovered by AFlow against manually designed methods for LLMs, including IO (direct LLM invocation), Chain-of-Thought ( Wei et al., 2022 ) , Self Consistency CoT (5 answers) ( Wang et al., 2022 ) , MultiPersona Debate ( Wang et al., 2024a ) , Self-Refine (max 3 iteration rounds) ( Madaan et al., 2023 ) , and MedPrompt (3 answers and 5 votes) ( Nori et al., 2023 ) . We also compared against workflow designed by automated workflow optimization method ADAS ( Hu et al., 2024 ) . 

Implementation Details AFlow utilizes different models for optimization and execution. We employ Claude-3.5-sonnet ( Anthropic, 2024 ) as the optimizer and use models: DeepSeek-V2.5 ( Deepseek, 2024 ) , GPT-4o-mini-0718 ( OpenAI, 2024b ) , Claude-3.5-sonnet-0620 ( Anthropic, 2024 ) , GPT-4o-0513 ( OpenAI, 2024a ) ) as executors. All models are accessed via APIs. We set the temperature to 1 for DeepSeek-V2.5 and to 0 for the other models. We set iteration rounds to 20 for AFlow . For ADAS, we use Claude-3.5-sonnet as the optimizer and GPT-4o-mini as the executor, with the iteration rounds set to 30. 

Metrics. For GSM8K and MATH l ​ v ​ 5 ∗ {}_{lv5^{*}} , we report the Solve Rate (%) as the primary metric. For HumanEval and MBPP, we report the pass@1 metric as presented in ( Chen et al., 2021 ) to assess code accuracy. For HotpotQA and DROP, we report the F1 Score. Additionally, for all datasets, we calculate the cost by tracking token usage to construct a pareto front, visually demonstrating the performance-cost trade-offs between different methods. 

5.2 Experimental Results and Analysis 

Main Results The main experimental results, as shown in Table 1 , demonstrate the effectiveness of AFlow . Workflows optimized by AFlow outperform all manually designed methods by an average of 5.7% and surpass contemporary automatic workflow optimization work by 19.5%. Across six datasets in QA, Code, and Math domains, AFlow achieves an average performance of 80.3%, marking the capability and usability of this method. Notably, compared to similar works, AFlow performed better on more challenging tasks, improving over ADAS on MATH l ​ v ​ 5 ∗ {}_{lv5^{*}} and MBPP tasks by 57%, showcasing the robustness of the model on complex datasets. 
Table 1: Comparison of performance between manually designed methods and workflow generated by automated workflow optimization methods in QA, code, and Math scenarios. All methods are executed with GPT-4o-mini on divided test set, and we tested it three times and reported it on the average. Method Benchmarks Avg. HotpotQA DROP HumanEval MBPP GSM8K MATH IO (GPT-4o-mini) 68.1 68.3 87.0 71.8 92.7 48.6 72.8 CoT ( Wei et al., 2022 ) 67.9 78.5 88.6 71.8 92.4 48.8 74.7 CoT SC (5-shot) ( Wang et al., 2022 ) 68.9 78.8 91.6 73.6 92.7 50.4 76.0 MedPrompt ( Nori et al., 2023 ) 68.3 78.0 91.6 73.6 90.0 50.0 75.3 MultiPersona ( Wang et al., 2024a ) 69.2 74.4 89.3 73.6 92.8 50.8 75.1 Self Refine ( Madaan et al., 2023 ) 60.8 70.2 87.8 69.8 89.6 46.1 70.7 ADAS ( Hu et al., 2024 ) 64.5 76.6 82.4 53.4 90.8 35.4 67.2 Ours 73.5 80.6 94.7 83.4 93.5 56.2 80.3 Table 2: Comparison of performance between manually designed methods and workflows generated by AFlow with two executor LLM: GPT-4o-mini (“Ours”) and DeepSeek-V2.5 (“Ours*”). All workflows are tested thrice on the humaneval test set, with average results reported. “MP” denotes “MedPrompt” ( Nori et al., 2023 ) , and “MPD” denotes “MultiPersona Debate” ( Wang et al., 2024a ) . The results demonstrate that workflows obtained through AFlow exhibit strong transferability. Model Methods IO CoT CoT SC MP MPD SR Ours Ours ∗ GPT-4o-mini 87.0 88.6 91.6 91.6 89.3 87.8 94.7 90.8 DeepSeek-V2.5 88.6 89.3 88.6 88.6 89.3 90.0 93.9 94.7 GPT-4o 93.9 93.1 94.7 93.9 94.7 91.6 96.2 95.4 Claude-3.5-sonnet 90.8 92.4 93.9 91.6 90.8 89.3 95.4 94.7 
To explore whether the workflow searched by AFlow is model-agnostic, we use GPT-4o-mini and DeepSeek-V2.5 as execution LLMs to search effective workflows with different structures, with the results illustrated in Table 2 . When applying these workflows to other models, the vast majority demonstrate stronger performance than the baseline, showcasing the generalizability of the workflows discovered by AFlow . Simultaneously, we observe that the workflow identified using DeepSeek-V2.5 performs notably weaker on GPT-4o-mini compared to the workflow found using GPT-4o-mini itself. This suggests that different language models require different workflows to achieve their optimal performance. 
Figure 4: The cost refers to the total expense of executing the divided HumanEval test set. AFlow (execution model) refers to workflows found by AFlow using the execution model to obtain feedback. The colors in the legend represent the LLM used to execute each workflow in test dataset. The specific numerical values for this Figure can be found in Appendix D . 
Cost Analysis We demonstrate the comparison of performance and cost between the baselines and the top three workflows found by AFlow using GPT-4o-mini and DeepSeek-V2.5 as execution LLMs. The comparison is made across four models with different capabilities and price points. Results demonstrate that AFlow can identify workflows that allow weaker models to outperform stronger models on the pareto front of cost-effectiveness. This breakthrough effectively removes barriers to the widespread application of agentic workflows across various domains. By automating the design of effective agentic workflows, AFlow eliminates the human labor costs previously required. Moreover, the ability to achieve superior performance at lower costs compared to stronger models opens up further possibilities for widespread adoption. 
Figure 5: (A) Comparison of highest performance curves on GSM8K for both validation and test sets generated by AFlow with and without operators. Compared to other datasets, GSM8K has a larger data volume, meaning that the same percentage improvement represents a greater increase in correctly solved samples, avoiding fluctuations in improvement due to small data size that could affect comparisons; (B): The code for the best-performing workflow discovered by AFlow on the GSM8K dataset. 
Ablation Study We introduce operators as human-designed effort to enhance search efficiency. An ablation study on GSM8K (Figure 5 ) shows that operators help AFlow discover better workflows more efficiently, achieving incremental improvements. Notably, even without operators, AFlow maintains strong performance (93.1%), surpassing manual designs. Notably, AFlow autonomously develops ensemble-like structures without operators, demonstrating its capability for independent workflow design and marking a significant step towards full automation. Details is shown in Appendix B . 
Figure 6: Tree-structured iteration process of AFlow on GSM8K: We highlight the path from the initial round (round 1) to the best-performing workflow, reporting the score for each node and its modification from the previous node. The purple sections in the prompts on both sides represent the main prompt modifications in this iteration. 
Case Study AFlow demonstrates a clear iteration process, as shown in Figure 6 , illustrating how it evolves from a blank template (containing only a single Node without prompts) to the structure presented in Figure 5 (B). In each iteration, AFlow employs a single-step modification, meaning it either adds one operator (rounds 2, 3) or makes a targeted modification to a prompt (rounds 8, 10). Among the unsuccessful exploration rounds, AFlow introduced a custom review node that directly modified answers generated through complex processes without additional reasoning (round 5), which decreased accuracy. In round 14, AFlow attempted to rephrase the problem but overly focused on “discount” information, leading to a decrease in accuracy. This iteration process showcases how tree-based search allows AFlow to further optimize known paths while retaining the ability to explore new ones. On the MBPP dataset, AFlow discovered structures similar to current manually designed workflows, such as test generation and execution by LLMs as seen in Ridnik et al. (2024) . The workflow and more discovered results are presented in Appendix B and a complete optimization process is presented in Appendix C . 

6 Conclusion 

This paper has introduced AFlow , a novel framework for automated workflow optimization. We have comprehensively formulated the automated workflow optimization problem, establishing a foundational structure for future research. AFlow has leveraged Monte Carlo Tree Search and code-represented workflows to navigate the vast search space of possible workflows efficiently. Our experiments across six benchmarks demonstrate the effectiveness of AFlow , which has outperformed manually designed methods and existing automated optimization approaches. Ablation studies have shown that AFlow can autonomously discover effective structures, even without predefined operators. Importantly, AFlow has enabled weaker models to outperform stronger ones on the Pareto front of cost-effectiveness. We further discuss the potential applications of AFlow across diverse domains in Appendix F , potentially revolutionizing the adoption of agentic workflows across various domains. These results have highlighted AFlow ’s potential for enhancing LLMs’ problem-solving capabilities while optimizing computational costs. 

Acknowledgements 

This paper is supported by NSF of China (62402409), Guangzhou Municipality Big Data Intelligence Key Lab (2023A03J0012), Guangdong Basic and Applied Basic Research Foundation (2023A1515110545), Guangzhou Basic and Applied Basic Research Foundation (2025A04J3935), and Guangzhou-HKUST(GZ) Joint Funding Program (2025A03J3714). 

References 

Anthropic (2024) Anthropic. Introducing claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet , 2024. 

Austin et al. (2021) Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. CoRR , abs/2108.07732, 2021. 

Chen et al. (2021) Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. CoRR , abs/2107.03374, 2021. 

Cobbe et al. (2021) Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 , 2021. 

Dai et al. (2024) Yanqi Dai, Huanran Hu, Lei Wang, Shengjie Jin, Xu Chen, and Zhiwu Lu. Mmrole: A comprehensive framework for developing and evaluating multimodal role-playing agents. arXiv preprint arXiv:2408.04203 , 2024. 

Deepseek (2024) Deepseek. DeepSeek-V2.5. https://huggingface.co/deepseek-ai/DeepSeek-V2.5 , 2024. 

Dua et al. (2019) Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. DROP: A reading comprehension benchmark requiring discrete reasoning over paragraphs. In NAACL-HLT (1) , pp. 2368–2378. Association for Computational Linguistics, 2019. 

Fernando et al. (2024) Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, Simon Osindero, and Tim Rocktäschel. Promptbreeder: Self-referential self-improvement via prompt evolution. In ICML . OpenReview.net, 2024. 

Hendrycks et al. (2021) Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2) , 2021. 

Hong et al. (2024a) Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, Binhao Wu, Danyang Li, Jiaqi Chen, Jiayi Zhang, Jinlin Wang, Li Zhang, Lingyao Zhang, Min Yang, Mingchen Zhuge, Taicheng Guo, Tuo Zhou, Wei Tao, Wenyi Wang, Xiangru Tang, Xiangtao Lu, Xiawu Zheng, Xinbing Liang, Yaying Fei, Yuheng Cheng, Zongze Xu, and Chenglin Wu. Data interpreter: An LLM agent for data science. CoRR , abs/2402.18679, 2024a. 

Hong et al. (2024b) Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. Metagpt: Meta programming for A multi-agent collaborative framework. In ICLR . OpenReview.net, 2024b. 

Hu et al. (2024) Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. arXiv preprint arXiv:2408.08435 , 2024. 

Khattab et al. (2024) Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language model calls into state-of-the-art pipelines. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024 . OpenReview.net, 2024. 

Li et al. (2024a) Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, and Nan Tang. The dawn of natural language to SQL: are we fully ready? Proc. VLDB Endow. , 17(11):3318–3331, 2024a. 

Li et al. (2024b) Zelong Li, Shuyuan Xu, Kai Mei, Wenyue Hua, Balaji Rama, Om Raheja, Hao Wang, He Zhu, and Yongfeng Zhang. Autoflow: Automated workflow generation for large language model agents. CoRR , abs/2407.12821, 2024b. 

Liu et al. (2024) Xinyu Liu, Shuyu Shen, Boyan Li, Peixian Ma, Runzhi Jiang, Yuyu Luo, Yuxin Zhang, Ju Fan, Guoliang Li, and Nan Tang. A survey of NL2SQL with large language models: Where are we, and where are we going? CoRR , abs/2408.05109, 2024. 

Liu et al. (2023) Zijun Liu, Yanzhe Zhang, Peng Li, Yang Liu, and Diyi Yang. Dynamic llm-agent network: An llm-agent collaboration framework with agent team optimization. arXiv preprint arXiv:2310.02170 , 2023. 

Luo et al. (2018) Yuyu Luo, Xuedi Qin, Nan Tang, and Guoliang Li. Deepeye: Towards automatic data visualization. In ICDE , pp. 101–112. IEEE Computer Society, 2018. 

Madaan et al. (2023) Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. In Thirty-seventh Conference on Neural Information Processing Systems , 2023. 

Niu et al. (2025) Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, and Tongliang Liu. Flow: Modularized agentic workflow automation, 2025. URL https://arxiv.org/abs/2501.07834 . 

Nori et al. (2023) Harsha Nori, Yin Tat Lee, Sheng Zhang, Dean Carignan, Richard Edgar, Nicolò Fusi, Nicholas King, Jonathan Larson, Yuanzhi Li, Weishung Liu, Renqian Luo, Scott Mayer McKinney, Robert Osazuwa Ness, Hoifung Poon, Tao Qin, Naoto Usuyama, Chris White, and Eric Horvitz. Can generalist foundation models outcompete special-purpose tuning? case study in medicine. CoRR , abs/2311.16452, 2023. 

OpenAI (2024a) OpenAI. Hello gpt-4o. https://openai.com/index/hello-gpt-4o/ , 2024a. 

OpenAI (2024b) OpenAI. GPT-4o mini: Advancing cost-efficient intelligence. https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/ , 2024b. 

Qiao et al. (2025) Shuofei Qiao, Runnan Fang, Zhisong Qiu, Xiaobin Wang, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. Benchmarking agentic workflow generation, 2025. URL https://arxiv.org/abs/2410.07869 . 

Ridnik et al. (2024) Tal Ridnik, Dedy Kredo, and Itamar Friedman. Code generation with alphacodium: From prompt engineering to flow engineering. CoRR , abs/2401.08500, 2024. 

Saad-Falcon et al. (2024) Jon Saad-Falcon, Adrian Gamarra Lafuente, Shlok Natarajan, Nahum Maru, Hristo Todorov, Etash Guha, E. Kelly Buchanan, Mayee Chen, Neel Guha, Christopher Ré, and Azalia Mirhoseini. Archon: An architecture search framework for inference-time techniques. arXiv preprint arXiv:2409.15254 , 2024. 

Shinn et al. (2023) Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In NeurIPS , 2023. 

Song et al. (2023) Chan Hee Song, Brian M Sadler, Jiaman Wu, Wei-Lun Chao, Clayton Washington, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV) , pp. 2986–2997. IEEE Computer Society, 2023. 

Sun et al. (2023) Hongda Sun, Weikai Xu, Wei Liu, Jian Luan, Bin Wang, Shuo Shang, Ji-Rong Wen, and Rui Yan. From indeterminacy to determinacy: Augmenting logical reasoning capabilities with large language models. arXiv preprint arXiv:2310.18659 , 2023. 

Sun et al. (2024) Yiyou Sun, Junjie Hu, Wei Cheng, and Haifeng Chen. Chatbot meets pipeline: Augment large language model with definite finite automaton. arXiv preprint arXiv:2402.04411 , 2024. 

Tam et al. (2024) Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, and Yun-Nung Chen. Let me speak freely? A study on the impact of format restrictions on performance of large language models. CoRR , abs/2408.02442, 2024. 

Tang et al. (2024) Nan Tang, Chenyu Yang, Ju Fan, Lei Cao, Yuyu Luo, and Alon Y. Halevy. Verifai: Verified generative AI. In CIDR . www.cidrdb.org, 2024. 

Wang et al. (2023) Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 , 2023. 

Wang et al. (2022) Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations , 2022. 

Wang et al. (2024a) Zhenhailong Wang, Shaoguang Mao, Wenshan Wu, Tao Ge, Furu Wei, and Heng Ji. Unleashing the emergent cognitive synergy in large language models: A task-solving agent through multi-persona self-collaboration. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , pp. 257–279, 2024a. 

Wang et al. (2024b) Zilong Wang, Hao Zhang, Chun-Liang Li, Julian Martin Eisenschlos, Vincent Perot, Zifeng Wang, Lesly Miculicich, Yasuhisa Fujii, Jingbo Shang, Chen-Yu Lee, et al. Chain-of-table: Evolving tables in the reasoning chain for table understanding. In The Twelfth International Conference on Learning Representations , 2024b. 

Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems , 35:24824–24837, 2022. 

Xie et al. (2024a) Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. Travelplanner: A benchmark for real-world planning with language agents. In Forty-first International Conference on Machine Learning , 2024a. 

Xie et al. (2024b) Yupeng Xie, Yuyu Luo, Guoliang Li, and Nan Tang. Haichart: Human and AI paired visualization system. Proc. VLDB Endow. , 17(11):3178–3191, 2024b. 

Xu et al. (2024) Yiheng Xu, SU Hongjin, Chen Xing, Boyu Mi, Qian Liu, Weijia Shi, Binyuan Hui, Fan Zhou, Yitao Liu, Tianbao Xie, et al. Lemur: Harmonizing natural language and code for language agents. In The Twelfth International Conference on Learning Representations , 2024. 

Yang et al. (2024a) Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen. Large language models as optimizers. In ICLR . OpenReview.net, 2024a. 

Yang et al. (2024b) Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E Gonzalez, and Bin Cui. Buffer of thoughts: Thought-augmented reasoning with large language models. arXiv preprint arXiv:2406.04271 , 2024b. 

Yang et al. (2018) Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In EMNLP , pp. 2369–2380. Association for Computational Linguistics, 2018. 

Ye et al. (2024) Yilin Ye, Jianing Hao, Yihan Hou, Zhan Wang, Shishi Xiao, Yuyu Luo, and Wei Zeng. Generative AI for visualization: State of the art and future directions. Vis. Informatics , 8(1):43–66, 2024. 

Yu et al. (2023) Junchi Yu, Ran He, and Zhitao Ying. Thought propagation: An analogical approach to complex reasoning with large language models. In The Twelfth International Conference on Learning Representations , 2023. 

Yüksekgönül et al. (2024) Mert Yüksekgönül, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. Textgrad: Automatic ”differentiation” via text. CoRR , abs/2406.07496, 2024. 

Zhang et al. (2024a) Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, and Tianlong Chen. Cut the crap: An economical communication pipeline for llm-based multi-agent systems. arXiv preprint arXiv:2410.02506 , 2024a. 

Zhang et al. (2024b) Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, and Dawei Cheng. G-designer: Architecting multi-agent communication topologies via graph neural networks. arXiv preprint arXiv:2410.11782 , 2024b. 

Zhang et al. (2024c) Jiayi Zhang, Chuang Zhao, Yihan Zhao, Zhaoyang Yu, Ming He, and Jianping Fan. Mobileexperts: A dynamic tool-enabled agent team in mobile devices. CoRR , abs/2407.03913, 2024c. 

Zheng et al. (2023) Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems , 36:46595–46623, 2023. 

Zhong et al. (2024a) Li Zhong, Zilong Wang, and Jingbo Shang. Debug like a human: A large language model debugger via verifying runtime execution step by step. In ACL (Findings) , pp. 851–870. Association for Computational Linguistics, 2024a. 

Zhong et al. (2024b) Qihuang Zhong, Kang Wang, Ziyang Xu, Juhua Liu, Liang Ding, Bo Du, and Dacheng Tao. Achieving¿ 97% on gsm8k: Deeply understanding the problems makes llms perfect reasoners. arXiv preprint arXiv:2404.14963 , 2024b. 

Zhou et al. (2024a) Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In Forty-first International Conference on Machine Learning , 2024a. 

Zhou et al. (2024b) Wangchunshu Zhou, Yixin Ou, Shengwei Ding, Long Li, Jialong Wu, Tiannan Wang, Jiamin Chen, Shuai Wang, Xiaohua Xu, Ningyu Zhang, Huajun Chen, and Yuchen Eleanor Jiang. Symbolic learning enables self-evolving agents. CoRR , abs/2406.18532, 2024b. 

Zhou et al. (2023) Xuanhe Zhou, Guoliang Li, and Zhiyuan Liu. Llm as dba. arXiv preprint arXiv:2308.05481 , 2023. 

Zhu et al. (2024) Yizhang Zhu, Shiyin Du, Boyan Li, Yuyu Luo, and Nan Tang. Are large language models good statisticians? In NeurIPS , 2024. 

Zhuge et al. (2023) Mingchen Zhuge, Haozhe Liu, Francesco Faccio, Dylan R Ashley, Róbert Csordás, Anand Gopalakrishnan, Abdullah Hamdi, Hasan Abed Al Kader Hammoud, Vincent Herrmann, Kazuki Irie, et al. Mindstorms in natural language-based societies of mind. arXiv preprint arXiv:2305.17066 , 2023. 

Zhuge et al. (2024) Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. Gptswarm: Language agents as optimizable graphs. In Forty-first International Conference on Machine Learning , 2024. 

Appendix A Appendix 

A.1 LLM Based Expansion: Prompt for LLM Optimizer 

A.2 Basic Structure of Node 

A.3 Basic Structure of Workflow 

A.4 Operators 

Providing predefined operators can effectively enhance the search efficiency of AFlow . We implement six common operator structures, including: Generate (Contextual, Code), Format, Review & Revise, Ensemble, Test, and Programmer. For the Test Operator, we use the public test dataset of the dataset as test data. For datasets like MBPP that don’t provide a public test dataset, we follow the setting in Zhong et al. (2024a) where we use the first test case of each problem as public test data. 

A.5 Mapping workflow from Formulation to Code 

In this example, 

• 
self.custom is the interface for building nodes, through which the Optimizer can generate/modify its prompts. 

• 
self.test and self.sc ensemble are interfaces for using Operators (In this example, this workflow only use 2 operators). 

• 
Edge in AFlow are represented through code, controlling the flow of all input/output variables between Nodes and Operators to form a complete workflow. Given this definition, the traditional concept of a ’node having two outgoing edges’ does not apply to this formulation. 

A.6 MCTS Algorithm of AFlow . 
Algorithm 1 Detailed Explanation of the AFlow Algorithm 1: Initial Workflow W 0 W_{0} , Evaluator G G , Dataset D D , Number of rounds N N , Operators 𝒪 \mathcal{O} , Top k k k , Early stopping rounds n n 2: Optimal Workflow W ∗ W^{*} 3: Initialize r ​ e ​ s ​ u ​ l ​ t ​ s ← ∅ results\leftarrow\emptyset , e ​ x ​ p ​ e ​ r ​ i ​ e ​ n ​ c ​ e ​ s ← ∅ experiences\leftarrow\emptyset , N ← 20 N\leftarrow 20 , k ← 3 k\leftarrow 3 , n ← 5 n\leftarrow 5 4: D V , D T ← D_{V},D_{T}\leftarrow RandomSplit( D D , 0.2, 0.8) ⊳ \triangleright Split dataset: 20% for validation, 80% for training 5: s ​ c ​ o ​ r ​ e ​ s ← scores\leftarrow Execute( W 0 W_{0} , G G , D V D_{V} ) 6: D V ← D_{V}\leftarrow SelectHighVarianceInstances( D V D_{V} , s ​ c ​ o ​ r ​ e ​ s scores , t ​ h ​ r ​ e ​ s ​ h ​ o ​ l ​ d threshold ) ⊳ \triangleright Select instances 7: for r ​ o ​ u ​ n ​ d ← 1 round\leftarrow 1 to N N do 8: if r ​ o ​ u ​ n ​ d = 1 round=1 then 9: p ​ a ​ r ​ e ​ n ​ t ← W 0 parent\leftarrow W_{0} 10: else 11: p ​ a ​ r ​ e ​ n ​ t ← parent\leftarrow SelectParent( r ​ e ​ s ​ u ​ l ​ t ​ s results ) 12: end if 13: c ​ o ​ n ​ t ​ e ​ x ​ t ← context\leftarrow LoadContext( p ​ a ​ r ​ e ​ n ​ t parent , e ​ x ​ p ​ e ​ r ​ i ​ e ​ n ​ c ​ e ​ s experiences ) 14: W r ​ o ​ u ​ n ​ d , W_{round}, modification ← \leftarrow Optimizer( c ​ o ​ n ​ t ​ e ​ x ​ t context , 𝒪 \mathcal{O} ) 15: for i ← 1 i\leftarrow 1 to 5 5 do 16: s ​ c ​ o ​ r ​ e , c ​ o ​ s ​ t ← score,cost\leftarrow Executor( W r ​ o ​ u ​ n ​ d W_{round} , E E , D V D_{V} ) 17: r ​ e ​ s ​ u ​ l ​ t ​ s results .append( r ​ o ​ u ​ n ​ d round , s ​ c ​ o ​ r ​ e score , c ​ o ​ s ​ t cost ) 18: end for 19: a ​ v ​ g ​ S ​ c ​ o ​ r ​ e ← avgScore\leftarrow CalculateAverageScore( r ​ e ​ s ​ u ​ l ​ t ​ s results [ r ​ o ​ u ​ n ​ d round ]) 20: e ​ x ​ p ​ e ​ r ​ i ​ e ​ n ​ c ​ e ← experience\leftarrow CreateExperience( p ​ a ​ r ​ e ​ n ​ t parent , m ​ o ​ d ​ i ​ f ​ i ​ c ​ a ​ t ​ i ​ o ​ n modification , a ​ v ​ g ​ S ​ c ​ o ​ r ​ e avgScore ) 21: e ​ x ​ p ​ e ​ r ​ i ​ e ​ n ​ c ​ e ​ s experiences .append( e ​ x ​ p ​ e ​ r ​ i ​ e ​ n ​ c ​ e experience ) 22: if a ​ v ​ g ​ S ​ c ​ o ​ r ​ e > b ​ e ​ s ​ t ​ S ​ c ​ o ​ r ​ e avgScore>bestScore then 23: W ∗ ← W r ​ o ​ u ​ n ​ d W^{*}\leftarrow W_{round} 24: b ​ e ​ s ​ t ​ S ​ c ​ o ​ r ​ e ← a ​ v ​ g ​ S ​ c ​ o ​ r ​ e bestScore\leftarrow avgScore 25: end if 26: if The Top k k Workflows remains unchanged in n n rounds then ⊳ \triangleright Early stopping return W ∗ W^{*} 27: end if 28: end for 29: return W ∗ W^{*} 30: procedure SelectParent ( r ​ e ​ s ​ u ​ l ​ t ​ s results ) 31: s o r t e d _ r e s u l t s ← SortDescending ( r e s u l t s , key=lambda r: r.scores ) sorted\_results\leftarrow\text{SortDescending}(results,\text{key=lambda r: r.scores}) 32: t o p _ k _ r e s u l t s ← s o r t e d _ r e s u l t s [ : k ] top\_k\_results\leftarrow sorted\_results[:k] 33: s c o r e s ← [ r e s u l t . s c o r e s for r e s u l t in t o p _ k _ r e s u l t s ] scores\leftarrow[result.scores\text{ for }result\text{ in }top\_k\_results] 34: p ​ r ​ o ​ b ​ a ​ b ​ i ​ l ​ i ​ t ​ i ​ e ​ s ← CalculateMixedProbabilities ​ ( s ​ c ​ o ​ r ​ e ​ s ) probabilities\leftarrow\text{CalculateMixedProbabilities}(scores) 35: return SampleFromCategorical (probabilities) 36: end procedure 37: procedure CalculateMixedProbabilities ( s ​ c ​ o ​ r ​ e ​ s scores ) 38: n ← n\leftarrow length( s ​ c ​ o ​ r ​ e ​ s scores ), λ ← 0.4 \lambda\leftarrow 0.4 , α ← 0.2 \alpha\leftarrow 0.2 , s m ​ a ​ x ← max ⁡ ( s ​ c ​ o ​ r ​ e ​ s ) s_{max}\leftarrow\max(scores) 39: w i ← exp ⁡ ( α ⋅ ( s i − s m ​ a ​ x ) ) w_{i}\leftarrow\exp(\alpha\cdot(s_{i}-s_{max})) for i ∈ [ 1 , n ] i\in[1,n] 40: P s ​ c ​ o ​ r ​ e ← w i / ∑ j = 1 n w j P_{score}\leftarrow w_{i}/\sum_{j=1}^{n}w_{j} for i ∈ [ 1 , n ] i\in[1,n] 41: P u ​ n ​ i ​ f ​ o ​ r ​ m ← 1 / n P_{uniform}\leftarrow 1/n for i ∈ [ 1 , n ] i\in[1,n] 42: P m ​ i ​ x ​ e ​ d ← λ ⋅ P u ​ n ​ i ​ f ​ o ​ r ​ m + ( 1 − λ ) ⋅ P s ​ c ​ o ​ r ​ e P_{mixed}\leftarrow\lambda\cdot P_{uniform}+(1-\lambda)\cdot P_{score} 43: return P m ​ i ​ x ​ e ​ d P_{mixed} 44: end procedure 45: procedure Optimizer ( c ​ o ​ n ​ t ​ e ​ x ​ t context , O ​ p ​ e ​ r ​ a ​ t ​ o ​ r ​ s Operators ) 46: // LLM as Optimizer, generate new workflow and modification. 47: return n ​ e ​ w ​ W ​ o ​ r ​ k ​ f ​ l ​ o ​ w newWorkflow , m ​ o ​ d ​ i ​ f ​ i ​ c ​ a ​ t ​ i ​ o ​ n modification 48: end procedure 49: procedure Executor ( W W , e ​ v ​ a ​ l ​ u ​ a ​ t ​ o ​ r evaluator , d ​ a ​ t ​ a ​ s ​ e ​ t dataset ) 50: // LLM as Executor, execute workflow on dataset and return score and cost 51: return s ​ c ​ o ​ r ​ e score , c ​ o ​ s ​ t cost 52: end procedure 
Appendix B Case Study 

B.1 Case Study of AFlow 

AFlow demonstrates its ability to reduce human effort by evolving from an empty workflow to a solution highly similar to manually designed workflows like Ridnik et al. (2024) in the code generation scenario. This showcases AFlow ’s capability to generate efficient workflows comparable to expert designs with minimal human intervention. 

This optimal workflow generated for the MATH task showcases the model’s ability to generate complex, task-specific solutions from task-agnostic initial settings. It combines programmatic solutions with various reasoning strategies, culminating in an ensemble selection process, and spontaneously formats the answer into the required form. This adaptation demonstrates the model’s flexibility in tailoring workflows to different problem domains, while maintaining sophisticated problem-solving structures. 

The optimal workflow generated for the MBPP task simply combines operators with an ingenious FIX-CODE PROMPT, achieving the optimal workflow in the iteration at the fourteenth round. Although this workflow is simple, its score is extremely high and stable, demonstrating AFlow ’s potential to find the optimal cost-performance balance. 

The optimal workflow generated for the HotpotQA task demonstrates the effectiveness of execution feedback. Apart from logical reasoning, another factor affecting QA problem scores is effective formatting. AFlow can effectively identify the correct format and automatically perform formatting through learning from execution feedback, showcasing the efficacy of this design. 

In the ablation study, where predefined operators were deliberately removed, AFlow surprisingly developed this simplified yet effective workflow. Most notably, it independently evolved an ensemble-like operator, mirroring a key aspect of the optimal workflow. This emergence of a multi-solution generation and selection process, despite reduced guidance, highlights AFlow ’s inherent tendency towards robust problem-solving strategies. The spontaneous development of this ensemble approach in a constrained environment underscores AFlow ’s ability to identify and implement ef
