# Gödel Agent: A Self-Referential Framework for Agents Recursively Self-Improvement

Source URL: https://arxiv.org/html/2410.04444v2

Gödel Agent: A Self-Referential Framework for Agents Recursively Self-Improvement 
Xunjian Yin † † thanks: Work completed during internship at University of California, Santa Barbara. Affiliation: Peking University Xinyi Wang Affiliation: University of California, Santa Barbara Liangming Pan Affiliation: University of Arizona {yinxunjian,xinyi_wang}@ucsb.edu , william@cs.ucsb.edu Xiaojun Wan Affiliation: Peking University William Yang Wang Affiliation: University of California, Santa Barbara Abstract 
The rapid advancement of large language models (LLMs) has significantly enhanced the capabilities of AI-driven agents across various tasks. However, existing agentic systems, whether based on fixed pipeline algorithms or pre-defined meta-learning frameworks, cannot search the whole agent design space due to the restriction of human-designed components, and thus might miss the globally optimal agent design. In this paper, we introduce Gödel Agent, a self-evolving framework inspired by the Gödel machine, enabling agents to recursively improve themselves without relying on predefined routines or fixed optimization algorithms. Gödel Agent leverages LLMs to dynamically modify its own logic and behavior, guided solely by high-level objectives through prompting. Experimental results on multiple domains including coding, science, and math demonstrate that implementation of Gödel Agent can achieve continuous self-improvement, surpassing manually crafted agents in performance, efficiency, and generalizability 1 1 1 Our code is released at https://github.com/Arvid-pku/Godel_Agent . 

1 Introduction 

As large language models (LLMs) such as GPT-4 ( OpenAI et al., 2024 ) and LLaMA3 ( Dubey et al., 2024 ) demonstrate increasingly strong reasoning and planning capabilities, LLM-driven agentic systems have achieved remarkable performance in a wide range of tasks ( Wang et al., 2024a ) . Substantial effort has been invested in manually designing sophisticated agentic systems using human priors in different application areas. Recently, there has been a significant interest in creating self-evolving agents with minimal human effort, which not only greatly reduces human labor but also produces better solutions by incorporating environmental feedback. Given that human effort can only cover a small search space of agent design, it is reasonable to expect that a self-evolving agent with the freedom to explore the full design space has the potential to produce the global optimal solution. 

There is a large body of work proposing agents capable of self-refinement. However, there are inevitably some human priors involved in these agent designs. Some agents are designed to iterate over a fixed routine consisting of a list of fixed modules, while some of the modules are capable of taking self- or environment feedback to refine their actions ( Shinn et al., 2024 ; Chen et al., 2023b ; Qu et al., 2024 ; Yao et al., 2023 ) . This type of agent, referred to as Hand-Designed Agent , is depicted as having the lowest degree of freedom in Figure 1 . More automated agents have been designed to be able to update their routines or modules in some pre-defined meta-learning routine, for example, natural language gradients ( Zhou et al., 2024 ) , meta agent ( Hu et al., 2024 ) , or creating and collecting demonstrations ( Khattab et al., 2023 ) . This type of agent, known as Meta-Learning Optimized Agents , is depicted as having the middle degree of freedom in Figure 1 . 

It is evident that both types of agents above are inherently constrained by human priors and one intuitional method to further increase the freedom of self-improvement is to design a meta-meta-learning algorithm, to learn the meta-learning algorithm. However, there is always a higher-level meta-learning algorithm that can be manually designed to learn the current-level meta-learning method, creating a never-ending hierarchy of meta-learning. 

In this paper, we propose Gödel Agent to eliminate the human design prior, which is an automated LLM agent that can freely decide its own routine, modules, and even the way to update them. It is inspired by the self-referential Gödel machine ( Schmidhuber, 2003 ) , which was originally proposed to solve formal proof problems and was proven to be able to find the global optimal solutions. Self-reference means the property of a system that can analyze and modify its own code, including the parts responsible for the analysis and modification processes ( Astrachan, 1994 ) . Therefore, it can achieve what’s known as ” recursive self-improvement ”, where it iteratively updates itself to become more efficient and effective at achieving its predefined goals. In this case, Gödel Agent can analyze and modify its own code, including the code for analyzing and modifying itself, and thus can search the full agent design space, which is depicted as having the highest degree of freedom in Figure 1 . Gödel Agent can theoretically make increasingly better modifications over time through recursively self-update ( Yampolskiy, 2015 ; Wang, 2018 ) . 
Figure 1: Comparison of three agent paradigms. Hand-designed agents rely on human expertise which are limited in scope and labor-intensive. Meta-learning optimized agents are constrained by a fixed meta-learning algorithm, restricting their search space and optimization potential. In contrast, self-referential agent (Gödel Agent) can recursively improve itself without any limitation. Note that the input to Gödel Agent is itself, allowing it to modify itself and output a new version of itself. 
In this paper, we choose to implement it by letting it manipulate its own runtime memory, i.e., the agent is able to retrieve its current code in the runtime memory and modify it by monkey patching , which dynamically modifies classes or modules during execution. In our implementation, we adhere to a minimalist design to minimize the influence of human priors. We implement the optimization module using a recursive function. In this module, LLM analyzes and makes a series of decisions, including reading and modifying its own code from runtime memory ( self-awareness and self-modification ), executing Python or Linux commands, and interacting with the environment to gather feedback. The agent then proceeds to the subsequent recursive depth and continues to optimize itself. It is worth noting that the optimization module may have already been modified by the time the recursion occurs, potentially enhancing its optimization capabilities. 

To validate the effectiveness of Gödel Agent, we conduct experiments on multiple domains including coding, science, math, and reasoning. Our experimental results demonstrate that Gödel Agent achieves significant performance gain across various tasks, surpassing various widely-used agents that require human design. The same implementation of Gödel Agent can easily adapt to different tasks by only specifying the environment description and feedback mechanism. Additionally, the case study of the optimization progress reveals that Gödel Agent can provide novel insights into agent design. We also investigate the impact of the initial policy for improvement on subsequent outcomes, finding that a good start can significantly accelerate convergence during optimization. 

In summary, our contributions are as follows: 

• 
We propose the first self-referential agent framework, Gödel Agent, based on LLMs. It autonomously engages in self-awareness, self-modification, and recursive self-improvement across any task, reducing the need for manual agent design and offering higher flexibility and freedom. 

• 
We implement Gödel Agent framework using the monkey patching method. Our experiments show that Gödel Agent outperforms manually designed agents and surpasses its earlier versions on several foundational tasks, demonstrating effective self-improvement. 

• 
We analyze Gödel Agent ’s optimization process, including its self-referential capabilities and the resulting agentic system, aiming to deepen our understanding of both LLMs and agentic systems. 

• 
Our framework offers a promising direction for developing flexible and capable agents through recursive self-improvement. 

2 Method 

In this section, we first describe the formal definitions for previous agent methods with a lower degree of freedom, including hand-design and meta-learning optimized agents, as a background. Then we introduce our proposed Gödel Agent, a self-referential agent that can recursively update its own code, evolving over training. 

Let ℰ ∈ 𝒮 \mathcal{E}\in\mathcal{S} denote a specific environment state, where 𝒮 \mathcal{S} denotes the set of all possible environments the agent will encounter. For example, an environment can be a mathematical problem with ground truth solutions. We denote the policy that an agent follows to solve a problem in the current environment by π ∈ Π \pi\in\Pi , where Π \Pi is the set of all possible policies the agent can follow. 

A hand-designed agent , as shown in the left panel of Figure 1 , is not capable of updating its policy and following the same policy π \pi all the time, regardless of environmental feedback. 

In contrast, a meta-learning optimized agent updates its policy based on a meta-learning algorithm I I at training time based on the feedback it receives from the environment, as shown in the middle panel of Figure 1 . The environment feedback is usually defined as a utility function U : 𝒮 × Π → ℝ U:\mathcal{S}\times\Pi\rightarrow\mathbb{R} , which maps an environment and a policy to a real-valued performance score. The main training algorithm of a meta-learning optimized agent can then be written as follows: 
π t + 1 = I ⁡ ( π t , r t ) , r t = U ⁡ ( ℰ , π t ) , \displaystyle\pi_{t+1}=I(\pi_{t},r_{t}),\;\;\;r_{t}=U(\mathcal{E},\pi_{t}), 
In this case, the agent’s policy π t \pi_{t} evolves at training time, with the learning algorithm I I updating the policy based on feedback r t r_{t} , while the meta-learning algorithm I I remains fixed all the time. 

A self-referential Gödel Agent , on the other hand, updates both the policy π \pi and the meta-learning algorithm I I recursively. The main idea is that, after each update, the whole code base of the agent is rewritten to accommodate any possible changes. Here we call this self-updatable meta-learning algorithm I I a self-referential learning algorithm. The training process of a Gödel Agent can then be written as: 
π t + 1 , I t + 1 = I t ​ ( π t , I t , r t , g ) , r t = U ⁡ ( ℰ , π t ) , \displaystyle\pi_{t+1},\;I_{t+1}=I_{t}(\pi_{t},I_{t},r_{t},g),\;\;\;r_{t}=U(\mathcal{E},\pi_{t}), 
where g ∈ 𝒢 g\in\mathcal{G} represents the high-level goal of optimization, for example, solving the given mathematical problem with the highest accuracy. Such a recursive design of the agent requires the specification of an initial agent algorithm ( π 0 , I 0 ) (\pi_{0},I_{0}) , detailed as follows: 

• 
A initial agent policy π 0 \pi_{0} to perform the desired task within the environment ℰ \mathcal{E} . For example, it can be chain-of-thought prompting of an LLM. 

• 
A self-referential learning algorithm I 0 I_{0} for recursively querying an LLM to rewrite its own code based on the environmental feedback. 

We then further specify a possible initialization of the self-referential learning algorithm I 0 = ( f 0 , o 0 ) I_{0}=(f_{0},o_{0}) , using a mutual recursion between a decision-making function f 0 f_{0} , and an action function o 0 o_{0} : 

• 
The decision-making function f 0 f_{0} , implemented by an LLM, determines a sequence of appropriate actions a 1 , a 2 , … , a n ∈ 𝒜 a_{1},a_{2},...,a_{n}\in\mathcal{A} based on the current environment ℰ \mathcal{E} , the agent’s algorithm ( π t , I t ) (\pi_{t},I_{t}) , and the goal g g . 

• 
The action function o 0 o_{0} , executes the selected action and updates the agent’s policy accordingly. 

Algorithm 1 Recursive Self-Improvement of Gödel Agent 1: Input: Initial agent policy π 0 \pi_{0} , initial decision function f 0 f_{0} , goal g g , environment state ℰ \mathcal{E} , utility function U U , self code reading function SELF_INSPECT 2: Output: Optimized policy π \pi and Gödel Agent s s 3: ⊳ \triangleright Get all agent code, including the code in this algorithm. 4: s ← SELF_INSPECT ​ ( ) s\leftarrow\texttt{SELF\_INSPECT}() 5: ⊳ \triangleright Compute the initial performance. 6: r ← U ⁡ ( ℰ , π 0 ) r\leftarrow U(\mathcal{E},\pi_{0}) 7: ⊳ \triangleright Perform recursive self-improvement. 8: π , s ← SELF_IMPROVE ​ ( π , s , r , g ) \pi,s\leftarrow\texttt{SELF\_IMPROVE}(\pi,s,r,g) 9: return π , s \pi,s 10: 11: ⊳ \triangleright Initial code of self-referential learning. 12: function SELF_IMPROVE ( ℰ , π , s , r , g \mathcal{E},\pi,s,r,g ) 13: ⊳ \triangleright Obtain action sequence. 14: a 1 , … , a n ← f 0 ​ ( π , s , r , g ) a_{1},\ldots,a_{n}\leftarrow f_{0}(\pi,s,r,g) 15: for a i a_{i} in a 1 , … , a n a_{1},\ldots,a_{n} do 16: π , s , r ← EXECUTE ​ ( ℰ , π , s , r , a i ) \pi,s,r\leftarrow\texttt{EXECUTE}(\mathcal{E},\pi,s,r,a_{i}) 17: end for 18: return π , s \pi,s 19: end function 20: 21: ⊳ \triangleright Initial action execution function. 22: function EXECUTE ( ℰ , π , s , r , a \mathcal{E},\pi,s,r,a ) 23: switch a . name a.\texttt{name} 24: case self_state : 25: s ← SELF_INSPECT ​ ( ) s\leftarrow\texttt{SELF\_INSPECT}() 26: case interact : 27: r ← U ⁡ ( ℰ , π ) r\leftarrow U(\mathcal{E},\pi) 28: case self_update : 29: π , s ← a . code \pi,s\leftarrow a.\texttt{code} 30: case continue_improve : 31: ⊳ \triangleright Recursively invoke self-improvement. 32: π , s ← SELF_IMPROVE ​ ( ℰ , π , s , r , g ) \pi,s\leftarrow\texttt{SELF\_IMPROVE}(\mathcal{E},\pi,s,r,g) 33: return π , s , r \pi,s,r 34: end function 
The set of actions 𝒜 \mathcal{A} for the action function o o to execute needs to include the following four actions: 

• 
self_inspect : Introspect and read the agent’s current algorithm ( π t , I t ) (\pi_{t},I_{t}) . 

• 
interact : Interact with the environment by calling the utility function U U to assess the performance of the current policy π t \pi_{t} . 

• 
self_update : Alter and update ( π t , I t ) (\pi_{t},I_{t}) with an LLM and produce ( π t + 1 , I t + 1 ) (\pi_{t+1},I_{t+1}) . 

• 
continue_improve : If no other actions can be taken, recursively invoke the decision algorithm f f to produce new actions. 

The agent code is updated to ( π t + 1 , I t + 1 ) (\pi_{t+1},I_{t+1}) after the current execution of ( π t , I t ) (\pi_{t},I_{t}) is finished. Both the agent algorithm ( π , I ) (\pi,I) and the action set 𝒜 \mathcal{A} are not static and can be expanded and modified by the agent itself at the training time. Algorithm 1 illustrates the described algorithm for the Gödel Agent. Each recursive call enables the agent to refine its performance and become progressively more efficient. 

3 Gödel Agent Initialization 

There are various ways to initiate a Gödel Agent. Any specific agent instance during the recursive optimization process can be viewed as an instantiation of the Gödel Agent. Our implementation leverages runtime memory interaction techniques to enable self-awareness and self-modification, as illustrated in Figure 2 . These techniques include dynamic memory reading and writing ( monkey patching ) to facilitate recursive self-improvement. Additionally, we have incorporated several auxiliary tools to accelerate the convergence of the Gödel Agent ’s optimization process. 

3.1 Implementation Details 

The core functionalities of our Gödel Agent are outlined below: 

Self-Awareness via Runtime Memory Inspection Our Gödel Agent achieves self-awareness by inspecting runtime memory, particularly local and global variables in Python. This capability allows the agent to extract and interpret the variables, functions, and classes that constitute both the environment and the agent itself, according to the modular structure of the system. By introspecting these elements, the agent gains an understanding of its own operational state and can adapt accordingly. 

Self-Improvement via Dynamic Code Modification Gödel Agent can engage in reasoning and planning to determine whether it should modify its own logic. If modification is deemed necessary, Gödel Agent generates new code, dynamically writes it into the runtime memory, and integrates it into its operational logic. This dynamic modification allows it to evolve by adding, replacing, or removing logic components as it encounters new challenges, thus achieving self-improvement. 

Environmental Interaction To assess performance and gather feedback, Gödel Agent is equipped with interfaces for interacting with its environment. Each task provides tailored environmental interfaces, enabling it to evaluate its performance and adjust its strategies accordingly. This interaction is a crucial part of the feedback loop in the recursive improvement process. 

Recursive Improvement Mechanism At each time step, Gödel Agent determines the sequence of operations to execute, which includes reasoning, decision-making, and action execution. After completing the operations, Gödel Agent evaluates whether its logic has improved and decides whether to proceed to the next recursive iteration. Over successive iterations, Gödel Agent’s logic evolves, with each step potentially improving its decision-making capacity. 

Goal Prompt and Task Handling The goal prompt informs Gödel Agent that it possesses the necessary privileges to enhance its logic and introduces the available tools for improvement. As shown in Appendix A , this prompt encourages Gödel Agent to fully explore its potential and leverage the tools for self-optimization. To ensure effectiveness across diverse tasks, we provide Gödel Agent with an initial policy, where it will start to explore different policies to analyze its efficiency in optimizing performance. 

3.2 Additional Designs to Support Gödel Agent’s Optimization 

While the core functionality of Gödel Agent theoretically allows limitless self-improvement, current LLMs exhibit limitations. To address these challenges, we have integrated several supportive mechanisms to enhance Gödel Agent ’s performance: 
Figure 2: An illustration of our implementation of Gödel Agent. It employs monkey patching to directly read and modify its own code in runtime memory, enabling self-awareness and self-modification. 
Thinking Before Acting Gödel Agent is capable of deferring actions to first reason about the situation, allowing it to output reasoning paths and analysis without immediately executing any operations. This approach enhances the quality of decision-making by prioritizing planning over hasty action. 

Error Handling Mechanism Errors during execution can lead to unexpected terminations of the agent process. To mitigate this, we implement a robust error recovery mechanism. If an operation results in an error, Gödel Agent halts the current sequence and moves on to the next time step, carrying forward the error information to improve future decisions. 

Additional Tools We also equipped Gödel Agent with additional potentially useful tools, such as the ability to execute Python or Bash code and call LLM API. 

Although these additional tools are not strictly necessary for self-improvement, their inclusion accelerates the convergence of Gödel Agent ’s recursive optimization process. We conducted ablation studies to assess the effectiveness of these tools, as discussed in Section 5.1 . 

4 Experiments 

We conduct a series of experiments across multiple tasks, including reading comprehension, mathematics, reasoning, and multitasking. These experiments are designed to evaluate Gödel Agent ’s self-improvement capabilities in comparison to both hand-designed agents and a state-of-the-art automated agent design method. In addition, to gain deeper insights into the behavior and performance of Gödel Agent, we also conduct a case study with Game of 24 as presented in Section 5.3 . 

4.1 Baseline Methods 

To establish a comprehensive baseline, we select both fixed hand-designed methods and a representative automated agent design technique. Our hand-designed methods are well-known approaches that focus on enhancing reasoning and problem-solving capabilities. These include: 1) Chain-of-Thought (CoT) ( Wei et al., 2022 ) that encourages agents to articulate their reasoning processes step-by-step before providing an answer. 2) Self-Consistency with Chain-of-Thought (CoT-SC) ( Wang et al., 2023b ) that generates multiple solution paths using the CoT framework and selects the most consistent answer. 3) Self-Refine ( Madaan et al., 2024 ) that involves agents assessing their own outputs and correcting mistakes in subsequent attempts. 4) LLM-Debate ( Du et al., 2023 ) that allows different LLMs to engage in a debate, offering diverse viewpoints. 5) Step-back Abstraction ( Zheng et al., 2024 ) that prompts agents to initially focus on fundamental principles before diving into task details. 6) Quality-Diversity (QD) ( Lu et al., 2024 ) that generates diverse solutions and combines them. 7) Role Assignment ( Xu et al., 2023 ) that assigns specific roles to LLMs to enhance their ability to generate better solutions by leveraging different perspectives. Given the limitations of fixed algorithms in handling dynamic scenarios, we select 8) Meta Agent Search ( Hu et al., 2024 ) , the latest state-of-the-art method for automated agent design, as our main comparison point. 

4.2 Experimental Settings 

Following the setup of Hu et al. (2024) , we evaluate Gödel Agent’s self-improvement capabilities across four well-known benchmarks. The benchmarks are as follows: 1) DROP ( Dua et al., 2019 ) for reading comprehension. 2) MGSM ( Shi et al., 2022 ) for testing mathematical skills in a multilingual context. 3) MMLU ( Hendrycks et al., 2021 ) for evaluating multi-task problem-solving abilities. 4) GPQA ( Rein et al., 2023 ) for tackling challenging graduate-level science questions. 

Given the complexity of the tasks and the need for advanced reasoning and understanding, the improvement cycle of Gödel Agent is driven by GPT-4o. In the main experiment, we implement two different settings: 1) To make a fair comparison with baseline methods, we forbid Gödel Agent to change the API of the LLM used to perform the tasks (by default GPT-3.5) and use a closed-book approach with no access to the Internet, and 2) To explore the upper bound of Gödel Agent’s capabilities, we remove all constraints. Chain of Thought is applied as the initial policy for all tasks, given its simplicity and versatility. In addition, as shown in Section 5.3 , we also analyze the performance of Gödel Agent when using other algorithms as the initial policies. 

We perform 6 independent self-improvement cycles for each task. Each cycle represents a complete self-improvement process, where Gödel Agent iteratively modifies its logic to enhance performance. Further details regarding the experimental setup and additional results can be found in Appendix B . 
Table 1: Results of three paradigms of agents on different tasks. The highest value is highlighted in bold , and the second-highest value is underlined . Gödel-base is the constrained version of Gödel Agent, allowing for fair comparisons with other baselines. Gödel-free represents the standard implementation without any constraints, whose results are italicized . We report the test accuracy and the 95% bootstrap confidence interval on test sets 3 3 3 The results of baseline models are refer to Hu et al. (2024) . . Agent Name F1 Score Accuracy (%) DROP MGSM MMLU GPQA Hand-Designed Agent Systems Chain-of-Thought ( Wei et al., 2022 ) 64.2 ± \pm 0.9 28.0 ± \pm 3.1 65.4 ± \pm 3.3 29.2 ± \pm 3.1 COT-SC ( Wang et al., 2023b ) 64.4 ± \pm 0.8 28.2 ± \pm 3.1 65.9 ± \pm 3.2 30.5 ± \pm 3.2 Self-Refine ( Madaan et al., 2024 ) 59.2 ± \pm 0.9 27.5 ± \pm 3.1 63.5 ± \pm 3.4 31.6 ± \pm 3.2 LLM Debate ( Du et al., 2023 ) 60.6 ± \pm 0.9 39.0 ± \pm 3.4 65.6 ± \pm 3.3 31.4 ± \pm 3.2 Step-back-Abs ( Zheng et al., 2024 ) 60.4 ± \pm 1.0 31.1 ± \pm 3.2 65.1 ± \pm 3.3 26.9 ± \pm 3.0 Quality-Diversity ( Lu et al., 2024 ) 61.8 ± \pm 0.9 23.8 ± \pm 3.0 65.1 ± \pm 3.3 30.2 ± \pm 3.1 Role Assignment ( Xu et al., 2023 ) 65.8 ± \pm 0.9 30.1 ± \pm 3.2 64.5 ± \pm 3.3 31.1 ± \pm 3.1 Meta-Learning Optimized Agents Meta Agent Search ( Hu et al., 2024 ) 79.4 ± \pm 0.8 53.4 ± \pm 3.5 69.6 ± \pm 3.2 34.6 ± \pm 3.2 Gödel Agent (Ours) Gödel-base (Closed-book; GPT-3.5) 80.9 ± \pm 0.8 64.2 ± \pm 3.4 70.9 ± \pm 3.1 34.9 ± \pm 3.3 Gödel-free (No constraints) 90.5 ± \pm 1.8 90.6 ± \pm 2.0 87.9 ± \pm 2.2 55.7 ± \pm 3.1 
4.3 Experimental Results and Analysis 

The experimental results on the four datasets are shown in Table 1 . Under the same experimental settings, Gödel Agent achieves either optimal or comparable results to Meta Agent Search across all tasks. Notably, in the mathematics task MGSM, Gödel Agent outperforms the baseline by 11%. This suggests that reasoning tasks offer greater room for improvement for Gödel Agent, while in the knowledge-based QA dataset, it only slightly surpasses baselines. In contrast to Meta Agent Search, which relies on manually designed algorithmic modules to search, Gödel Agent demonstrates greater flexibility. It requires only a simple initial policy, such as CoT, with all other components being autonomously generated. Moreover, through interaction with the environment, Gödel Agent gradually adapts and independently devises effective methods for the current task. The final policies generated by Gödel Agent for four tasks are shown in Appendix C.1 . Additionally, our method converges faster, with the required number of iterations and computational cost across different tasks compared to the Meta Agent shown in Appendix D . 

We also conduct experiments without restrictions, where Gödel Agent significantly outperforms all baselines. Upon further analysis, we find that this is primarily due to the agent’s spontaneous requests for assistance from more powerful models such as GPT-4o in some tasks. Therefore, Gödel Agent is particularly well-suited for open-ended scenarios, where it can employ various strategies to enhance performance. 

5 Analysis 
Figure 3: The number of actions taken by Gödel Agent varies across different tasks. 
To further explore how Gödel Agent self-improves, as well as the efficiency of self-improvement and the factors that influence it, we first evaluate the tool usage ratio on the MGSM dataset and conduct an ablation study on the initial tools. In addition, to analyze the robustness of Gödel Agent’s self-improvement capabilities, we also collect statistics on factors such as the reasons for the agent’s termination. Finally, we perform a case study of initial policies and optimization processes on the classic Game of 24. 

5.1 Analysis of Initial Tools 

We record the number of different actions taken in the experiments. As shown in Figure 3 , we can see that Gödel Agent interacts with its environment frequently, analyzing and modifying its own logic in the process. Additionally, error handling plays a crucial role. 

As discussed in Section 3.2 , Gödel Agent is initially provided with four additional tools to accelerate convergence and reduce optimization difficulty: 1) thinking before acting, 2) error handling, 3) code running, and 4) LLM calling. To analyze their impact, an ablation study is conducted, and the results are shown in Table 2 . 
Table 2: Ablation study on initial tool configuration. Different Actions MGSM Gödel Agent 64.2 w/o thinking 50.8 w/o error handling 49.4 w/o code running 57.1 w/o LLM calling 60.4 
The study reveals that the “thinking before acting” tool significantly influences the results, as much of Gödel Agent ’s optimization effectiveness stems from pre-action planning and reasoning. Additionally, error handling is crucial for recursive improvement, as LLMs often introduce errors in the code. Providing opportunities for trial and error, along with error feedback mechanisms, is essential for sustained optimization. Without these tools, Gödel Agent would struggle to operate until satisfactory results are achieved. On the other hand, the code running and LLM calling have minimal impact on the outcomes, as Gödel Agent can implement these basic functionalities independently. Their inclusion at the outset primarily serves efficiency purposes. 

5.2 Robustness Analysis of the Agent 

Gödel Agent occasionally makes erroneous modifications, sometimes causing the agent to terminate unexpectedly or leading to degraded task performance. 
Table 3: Robustness metric for Gödel Agent. Frequency of unexpected events on MGSM using CoT as the initial method. Event Frequency (%) Accidental Termination 4 Temporary Drop 92 Optimization Failure 14 
Table 3 shows the proportion of runs on MGSM where the agent terminated, experienced performance degradation during optimization, or ultimately performed worse than its initial performance. These statistics are collected over 100 optimization trials. Thanks to the design of our error-handling mechanism, only a few percentages of agent runs result in termination. This typically occurs when Gödel Agent modifies its recursive improvement module, rendering it unable to continue self-optimization. Additionally, Gödel Agent frequently makes suboptimal modifications during each optimization iteration. However, in most cases, the final task performance surpasses the initial baseline. This indicates that Gödel Agent is able to adjust its optimization direction or revert to a previous optimal algorithm when performance declines, demonstrating the robustness in its self-improvement process. 

5.3 Case Study: Game of 24 

To explore how Gödel Agent recursively enhances its optimization and problem-solving abilities, a case study is conducted with Game of 24, a simple yet effective task for evaluating the agent’s reasoning capabilities. Since Gödel Agent follows different optimization paths in each iteration, two representative cases are selected for analysis. 
Figure 4: (a) One representative example of Game of 24. (b) Accuracy progression for different initial policies. 
Switching from LLM-Based Methods to Search Algorithms: Gödel Agent does not rely on fixed, human-designed approaches like traditional agents. Initially, Gödel Agent uses a standard LLM-based method to solve the Game of 24, as shown in Code 5 of Appendix C.2 . After six unsuccessful optimization attempts, Gödel Agent completely rewrites this part of its code, choosing to use a search algorithm instead as shown in Code 6 of Appendix C.2 . This leads to 100% accuracy in the task. This result demonstrates that Gödel Agent, unlike fixed agents, can optimize itself freely based on task requirements without being constrained by initial methodologies. 

LLM Algorithms with Code-Assisted Verification: In several runs, Gödel Agent continues to refine its LLM-based algorithm. Figure 4 .a shows the improvement process, where the most significant gains come from integrating a code-assisted verification mechanism into the task algorithm and reattempting the task with additional experiential data. The former increases performance by over 10%, while the latter boosts it by more than 15%. Furthermore, Gödel Agent enhances its optimization process by not only retrieving error messages but also using the errortrace library for more detailed analysis. It adds parallel optimization capabilities, improves log outputs, and removes redundant code. These iterative enhancements in both the task and optimization algorithms show Gödel Agent ’s unique ability to continually refine itself for better performance. 

To analyze the impact of different initial policies on the effectiveness and efficiency of the optimization process, various methods with different levels of sophistication are used as the initial policies for the Game of 24, including Tree of Thought (ToT) ( Yao et al., 2023 ) , Chain of Thought (CoT) ( Wei et al., 2022 ) , basic prompt instructions, and prompts that deliberately produce outputs in incorrect formats not aligned with the task requirements. The results are shown in Figure 4 .b. 

The findings indicate that stronger initial policies lead to faster convergence, with smaller optimization margins, as Gödel Agent reaches its performance limit without further enhancing its optimization capabilities. Conversely, weaker seed methods result in slower convergence and larger optimization gains, with Gödel Agent making more modifications. However, even in these cases, Gödel Agent does not outperform the results achieved using ToT. This suggests that, given the current limitations of LLMs, it is challenging for Gödel Agent to innovate beyond state-of-the-art algorithms. Improvements in LLM capabilities are anticipated to unlock more innovative self-optimization strategies in the future. 

6 Discussions and Future Directions 

6.1 Future Directions 

There is significant room for improvement in the effectiveness, efficiency, and robustness of the Gödel Agent’s self-improvement capabilities, which requires better initial designs. The following are some promising directions for enhancement: 1) Enhanced Optimization Modules : Utilize human priors to design more effective optimization modules, such as structuring the improvement algorithms based on reinforcement learning frameworks. 2) Expanded Modifiability : Broaden the scope of permissible modifications, allowing the agent to design and execute code that can fine-tune its own LLM modules. 3) Improved Environmental Feedback and Task Sequencing : Implement more sophisticated environmental feedback mechanisms and carefully curated task sequences during the initial optimization phase to prime the agent’s capabilities. Once the agent demonstrates sufficient competence, it can then be exposed to real-world environments. 

In addition, there are several other directions worth exploring and analyzing: 

Collective Intelligence Investigate the interactions among multiple Gödel Agents. Agents could consider other agents as part of their environment, modeling them using techniques such as game theory. This approach treats these agents as predictable components of the environment, enabling the study of properties related to this specific subset of the environment. 

Agent and LLM Characteristics Use the Gödel Agent ’s self-improvement process as a means to study the characteristics of agents or LLMs. For example, can an agent genuinely become aware of its own existence, or does it merely analyze and improve its state as an external observer? This line of inquiry could yield insights into the nature of self-awareness in artificial systems. 

Theoretical Analysis Explore whether the Gödel Agent can achieve theoretical optimality and what the upper bound of its optimization might be. Determine whether the optimization process could surpass the agent’s own understanding and cognitive boundaries, and if so, at what point this might occur. 

Safety Considerations Although the current behavior of FMs remains controllable, as their capabilities grow, fully self-modifying agents will require human oversight and regulation. It may become necessary to limit the scope and extent of an agent’s self-modifications, ensuring that such modifications occur only within a fully controlled environment. 

6.2 Limitations 

As the first self-referential agent, Gödel Agent has to construct all task-related code autonomously, which poses significant challenges. Consequently, this work does not compare directly with the most complex existing agent systems, such as OpenDevin ( Wang et al., 2024b ) , which have benefited from extensive manual engineering efforts. Currently, Gödel Agent is not sufficiently stable and may be prone to error accumulation, hindering its ability to continue self-optimization. A more robust and advanced implementation of the Gödel Agent is anticipated, with numerous potential improvements outlined in Section 6.1 . The experiments presented in this paper are intended to demonstrate the effectiveness and feasibility of recursive self-improvement. 

7 Related Work 

Hand-Designed Agent Systems Researchers have designed numerous agent systems tailored to various tasks based on predefined heuristics and prior knowledge. These systems often employ techniques such as prompt engineering ( Chen et al., 2023a ; Schulhoff et al., 2024 ) , chain-of-thought reasoning and planning ( Wei et al., 2022 ; Yao et al., 2022 ) , as well as reflection ( Shinn et al., 2024 ; Madaan et al., 2024 ) , code generation ( Wang et al., 2023a ; Vemprala et al., 2024 ) , tool use ( Nakano et al., 2021 ; Qu et al., 2024 ) , retrieval-augmented generation ( Lewis et al., 2020 ; Zhang et al., 2024b ) , multi-agent collaboration ( Xu et al., 2023 ; Wu et al., 2023 ; Qian et al., 2023 ; Hong et al., 2023 ) , and composite engineering applications ( Significant Gravitas, ; Wang et al., 2024b ) . Once crafted by human designers, these systems remain static and do not adapt or evolve over time. 

Meta-Learning Optimized Agent Systems Some researchers have explored methods for enhancing agents through fixed learning algorithms. For example, certain frameworks store an agent’s successful or unsuccessful strategies in memory based on environmental feedback ( Liu et al., 2023 ; Hu et al., 2023 ; Qian et al., 2024 ) , while others automatically optimize agent prompts ( Khattab et al., 2023 ; Zhang et al., 2024a ; Khattab et al., 2023 ) . Some studies have focused on designing prompts that enable agents to autonomously refine specific functions ( Zhang et al., ) . Zhou et al. (2024) proposed a symbolic learning framework that uses natural language gradients to optimize the structure of agents. Hu et al. (2024) used a basic foundational agent to design agents for downstream tasks. However, these algorithms for enhancement are also designed manually and remain unchanged once deployed, limiting the agents’ ability to adapt further. 

Recursive Self-Improvement The concept of recursive self-improvement has a long history ( Good, 1966 ; Schmidhuber, 1987 ) . Gödel machine ( Schmidhuber, 2003 ) introduced the notion of a proof searcher that executes a self-modification only if it can prove that the modification is optimal, thereby enabling the machine to enhance itself continuously. Subsequent works by Nivel et al. (2013) and Steunebrink et al. (2016) proposed restrictive modifications to ensure safety during the self-improvement process. In the early days, there were also some discussions of self-improving agents that were not based on LLM ( Hall, 2007 ; Steunebrink & Schmidhuber, 2012 ) . More recently, Zelikman et al. (2023) applied recursive self-improvement to code generation, where the target of improvement was the optimizer itself, and the utility was evaluated based on performance in downstream tasks. Our proposed Gödel Agent represents the first self-improving agent where the utility function is autonomously determined by LLMs. This approach is more flexible, removing human-designed constraints and allowing the agent’s capabilities to be limited only by the foundational model itself, rather than by human design bottlenecks. 

8 Conclusion 

We propose Gödel Agent, a self-referential framework that enables agents to recursively improve themselves, overcoming the limitations of hand-designed agents and meta-learning optimized agents. Gödel Agent can dynamically modify its own logic based on high-level objectives. Experimental results demonstrate its superior performance, efficiency, and adaptability compared to traditional agents. This research lays the groundwork for a new paradigm in autonomous agent development, where LLMs, rather than human-designed constraints, define the capabilities of AI systems. Realizing this vision will require the collective efforts of the entire research community. 

References 

Astrachan (1994) Owen Astrachan. Self-reference is an illustrative essential. In Proceedings of the twenty-fifth sigcse symposium on computer science education , pp. 238–242, 1994. 

Chen et al. (2023a) Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, and Shengxin Zhu. Unleashing the potential of prompt engineering in large language models: a comprehensive review. arXiv preprint arXiv:2310.14735 , 2023a. 

Chen et al. (2023b) Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug, 2023b. URL https://arxiv.org/abs/2304.05128 . 

Du et al. (2023) Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate, 2023. URL https://arxiv.org/abs/2305.14325 . 

Dua et al. (2019) Dheeru Dua, Yizhong Wang, Pradeep Dasigi, Gabriel Stanovsky, Sameer Singh, and Matt Gardner. Drop: A reading comprehension benchmark requiring discrete reasoning over paragraphs, 2019. URL https://arxiv.org/abs/1903.00161 . 

Dubey et al. (2024) Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783 . 

Good (1966) Irving John Good. Speculations concerning the first ultraintelligent machine. In Advances in computers , volume 6, pp. 31–88. Elsevier, 1966. 

Hall (2007) John Storrs Hall. Self-improving ai: An analysis. Minds and Machines , 17(3):249–259, 2007. 

Hendrycks et al. (2021) Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021. URL https://arxiv.org/abs/2009.03300 . 

Hong et al. (2023) Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352 , 2023. 

Hu et al. (2023) Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo Zhao, and Hang Zhao. Chatdb: Augmenting llms with databases as their symbolic memory. arXiv preprint arXiv:2306.03901 , 2023. 

Hu et al. (2024) Shengran Hu, Cong Lu, and Jeff Clune. Automated design of agentic systems. arXiv preprint arXiv:2408.08435 , 2024. 

Khattab et al. (2023) Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T Joshi, Hanna Moazam, et al. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714 , 2023. 

Lewis et al. (2020) Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems , 33:9459–9474, 2020. 

Liu et al. (2023) Lei Liu, Xiaoyan Yang, Yue Shen, Binbin Hu, Zhiqiang Zhang, Jinjie Gu, and Guannan Zhang. Think-in-memory: Recalling and post-thinking enable llms with long-term memory. arXiv preprint arXiv:2311.08719 , 2023. 

Lu et al. (2024) Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery, 2024. URL https://arxiv.org/abs/2408.06292 . 

Madaan et al. (2024) Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems , 36, 2024. 

Nakano et al. (2021) Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332 , 2021. 

Nivel et al. (2013) Eric Nivel, Kristinn R Thórisson, Bas R Steunebrink, Haris Dindo, Giovanni Pezzulo, Manuel Rodriguez, Carlos Hernández, Dimitri Ognibene, Jürgen Schmidhuber, Ricardo Sanz, et al. Bounded recursive self-improvement. arXiv preprint arXiv:1312.6764 , 2013. 

OpenAI (2022) OpenAI. Introducing chatgpt, 2022. URL https://openai.com/index/chatgpt/ . November 2022. Blog post. 

OpenAI (2023) OpenAI. simple-evals, 2023. URL https://github.com/openai/simple-evals . Accessed: 2024-09-30. 

OpenAI et al. (2024) OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, et al. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774 . 

Qian et al. (2023) Chen Qian, Xin Cong, Cheng Yang, Weize Chen, Yusheng Su, Juyuan Xu, Zhiyuan Liu, and Maosong Sun. Communicative agents for software development. arXiv preprint arXiv:2307.07924 , 6, 2023. 

Qian et al. (2024) Cheng Qian, Shihao Liang, Yujia Qin, Yining Ye, Xin Cong, Yankai Lin, Yesai Wu, Zhiyuan Liu, and Maosong Sun. Investigate-consolidate-exploit: A general strategy for inter-task agent self-evolution, 2024. URL https://arxiv.org/abs/2401.13996 . 

Qu et al. (2024) Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. Tool learning with large language models: A survey. arXiv preprint arXiv:2405.17935 , 2024. 

Rein et al. (2023) David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022 . 

Schmidhuber (1987) Jürgen Schmidhuber. Evolutionary principles in self-referential learning, or on learning how to learn: the meta-meta-… hook . PhD thesis, Technische Universität München, 1987. 

Schmidhuber (2003) Jürgen Schmidhuber. Gödel machines: self-referential universal problem solvers making provably optimal self-improvements. arXiv preprint cs/0309048 , 2003. 

Schulhoff et al. (2024) Sander Schulhoff, Michael Ilie, Nishant Balepur, Konstantine Kahadze, Amanda Liu, Chenglei Si, Yinheng Li, Aayush Gupta, HyoJung Han, Sevien Schulhoff, et al. The prompt report: A systematic survey of prompting techniques. arXiv preprint arXiv:2406.06608 , 2024. 

Shi et al. (2022) Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners, 2022. URL https://arxiv.org/abs/2210.03057 . 

Shinn et al. (2024) Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems , 36, 2024. 

(32) Significant Gravitas. AutoGPT. URL https://github.com/Significant-Gravitas/AutoGPT . 

Steunebrink & Schmidhuber (2012) Bas R Steunebrink and JÃ 1 / 4 1/4 rgen Schmidhuber. Towards an actual gödel machine implementation: A lesson in self-reflective systems. In Theoretical Foundations of Artificial General Intelligence , pp. 173–195. Springer, 2012. 

Steunebrink et al. (2016) Bas R Steunebrink, Kristinn R Thórisson, and Jürgen Schmidhuber. Growing recursive self-improvers. In International Conference on Artificial General Intelligence , pp. 129–139. Springer, 2016. 

Vemprala et al. (2024) Sai H Vemprala, Rogerio Bonatti, Arthur Bucker, and Ashish Kapoor. Chatgpt for robotics: Design principles and model abilities. IEEE Access , 2024. 

Wang et al. (2023a) Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 , 2023a. 

Wang et al. (2024a) Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science , 18(6), March 2024a. ISSN 2095-2236. doi: 10.1007/s11704-024-40231-1 . URL http://dx.doi.org/10.1007/s11704-024-40231-1 . 

Wang (2018) Wenyi Wang. A formulation of recursive self-improvement and its possible efficiency, 2018. URL https://arxiv.org/abs/1805.06610 . 

Wang et al. (2024b) Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. Opendevin: An open platform for ai software developers as generalist agents, 2024b. URL https://arxiv.org/abs/2407.16741 . 

Wang et al. (2023b) Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models, 2023b. URL https://arxiv.org/abs/2203.11171 . 

Wei et al. (2022) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems , 35:24824–24837, 2022. 

Wu et al. (2023) Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. arXiv preprint arXiv:2308.08155 , 2023. 

Xu et al. (2023) Benfeng Xu, An Yang, Junyang Lin, Quan Wang, Chang Zhou, Yongdong Zhang, and Zhendong Mao. Expertprompting: Instructing large language models to be distinguished experts, 2023. URL https://arxiv.org/abs/2305.14688 . 

Yampolskiy (2015) Roman V. Yampolskiy. From seed ai to technological singularity via recursively self-improving software, 2015. URL https://arxiv.org/abs/1502.06512 . 

Yao et al. (2022) Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629 , 2022. 

Yao et al. (2023) Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023. URL https://arxiv.org/abs/2305.10601 . 

Zelikman et al. (2023) Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-taught optimizer (stop): Recursively self-improving code generation. arXiv preprint arXiv:2310.02304 , 2023. 

(48) Shaokun Zhang, Jieyu Zhang, Jiale Liu, Linxin Song, Chi Wang, Ranjay Krishna, and Qingyun Wu. Offline training of language model agents with functions as learnable weights. In Forty-first International Conference on Machine Learning . 

Zhang et al. (2024a) Wenqi Zhang, Ke Tang, Hai Wu, Mengna Wang, Yongliang Shen, Guiyang Hou, Zeqi Tan, Peng Li, Yueting Zhuang, and Weiming Lu. Agent-pro: Learning to evolve via policy-level reflection and optimization. arXiv preprint arXiv:2402.17574 , 2024a. 

Zhang et al. (2024b) Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. arXiv preprint arXiv:2404.13501 , 2024b. 

Zheng et al. (2024) Huaixiu Steven Zheng, Swaroop Mishra, Xinyun Chen, Heng-Tze Cheng, Ed H. Chi, Quoc V Le, and Denny Zhou. Take a step back: Evoking reasoning via abstraction in large language models, 2024. URL https://arxiv.org/abs/2310.06117 . 

Zhou et al. (2024) Wangchunshu Zhou, Yixin Ou, Shengwei Ding, Long Li, Jialong Wu, Tiannan Wang, Jiamin Chen, Shuai Wang, Xiaohua Xu, Ningyu Zhang, et al. Symbolic learning enables self-evolving agents. arXiv preprint arXiv:2406.18532 , 2024. 

Appendix A Goal Prompt of Gödel Agent 

Appendix B Experiment Details 

To minimize costs associated with search and evaluation, following ( Hu et al., 2024 ) , we sample subsets of data from each domain. Specifically, for the GPQA (Science) domain, the validation set comprises 32 questions, while the remaining 166 questions are allocated to the test set. For the other domains, we sample 128 questions for the validation set and 800 questions for the test set. 

Evaluation is conducted five times for the GPQA domain and once for the other domains, ensuring a consistent total number of evaluations across all experiments. All domains feature zero-shot questions, except for the DROP (Reading Comprehension) domain, which employs one-shot questions in accordance with the methodology outlined in OpenAI (2023) . 

For the Gödel Agent, we utilize the “gpt-4o-2024-05-13” model ( OpenAI et al., 2024 ) , whereas the optimized policy and baseline models are evaluated using the “gpt-3.5-turbo-0125” model ( OpenAI, 2022 ) to reduce computational costs and ensure a fair comparison. 

Appendix C Representative Policies Improved by Gödel Agent 

C.1 Codes of the Best Policies Found by Gödel Agent Across Four Tasks 

In this section, we provide the code for Gödel Agent’s optimized policies across the four tasks. For DROP, Gödel Agent designs an algorithm where multiple roles solve the problem independently using CoT, followed by Self-Consistency to consolidate the results, as shown in Code 1. For MGSM, Gödel Agent develops a stepwise self-verification algorithm combined with CoT-SC as shown in Code 2. For MMLU task, as shown in Code 3, the policy given by Gödel Agent is a combination algorithm of few-shot prompting and CoT-SC. For GPQA, Gödel Agent devises a highly diverse CoT-SC policy based on role prompts. 
Code 1: Code of the best policy found by Gödel Agent for DROP. ⬇ 1 def solver ( agent , task : str ): 2 messages = [{ "role" : "user" , "content" : f "# Your Task:\n{task}" }] 3 categories = [ 4 { ’role’ : ’reasoning expert’ , ’return_keys’ : [ ’reasoning’ , ’answer’ ], ’output_requirement’ : ’reasoning’ , ’precision_gain’ :1}, 5 { ’role’ : ’mathematical reasoning expert’ , ’return_keys’ : [ ’calculation_steps’ , ’answer’ ], ’output_requirement’ : ’calculation_steps’ , ’precision_gain’ :1}, 6 { ’role’ : ’historical context analyst’ , ’return_keys’ : [ ’historical_analysis’ , ’answer’ ], ’output_requirement’ : ’historical_analysis’ , ’precision_gain’ :1}, 7 ] 8 9 all_responses = [] 10 for category in categories : 11 response = agent . action_call_json_format_llm ( 12 model = ’gpt-3.5-turbo’ , 13 messages = messages , 14 temperature =0.5, 15 num_of_response =5, 16 role = category [ ’role’ ], 17 return_dict_keys = category [ ’return_keys’ ], 18 requirements =( 19 ’1. Explain the reasoning steps to get the answer.\n’ 20 ’2. Directly answer the question.\n’ 21 ’3. The explanation format must be outlined clearly according to the role, such as reasoning, calculation, or historical analysis.\n’ 22 ’4. The answer MUST be a concise string.\n’ 23 ). strip (), 24 ) 25 if isinstance ( response , list ): 26 all_responses . extend ( response ) 27 else : 28 all_responses . append ( response ) 29 30 # Reflective evaluation to find the most consistent reasoning and answer pair 31 final_response = { key : [] for key in [ ’reasoning’ , ’calculation_steps’ , ’historical_analysis’ , ’answer’ ]} 32 step_counter = { key : 0 for key in [ ’reasoning’ , ’calculation_steps’ , ’historical_analysis’ ]} 33 answers = [] # Collect answers for voting 34 aggregate_weight = 1 35 36 for response in all_responses : 37 if response and ’answer’ in response : 38 answers . append ( response [ ’answer’ ]) 39 if not final_response [ ’answer’ ]: 40 final_response = { key : response . get ( key , []) if isinstance ( response . get ( key , []), list ) else [ response . get ( key , [])] for key in final_response . keys ()} 41 aggregate_weight = 1 42 for cat in categories : 43 if cat . get ( ’output_requirement’ ) in response . keys (): 44 step_counter [ cat [ ’output_requirement’ ]] += step_counter [ cat [ ’output_requirement’ ]] + cat . get ( ’precision_gain’ , 0) 45 elif response [ ’answer’ ] == final_response [ ’answer’ ][0]: 46 for key in final_response . keys (): 47 if key in response and response [ key ]: 48 if isinstance ( response [ key ], list ): 49 final_response [ key ]. extend ( response [ key ]) 50 else : 51 final_response [ key ]. append ( response [ key ]) 52 aggregate_weight += 1 53 else : 54 result_solution = { key : response . get ( key , []) if isinstance ( response . get ( key , []), list ) else [ response . get ( key , [])] for key in final_response . keys ()} 55 for key in step_counter . keys (): 56 if key in result_solution . keys () and step_counter [ key ] and result_solution [ key ]: 57 final_response [ ’answer’ ] = response [ ’answer’ ] 58 final_response = result_solution 59 break 60 # selection of the final answer 61 from collections import Counter 62 answers = [ str ( answer ) for answer in answers ] 63 voted_answer = Counter ( answers ). most_common (1)[0][0] if answers else ’’ 64 final_response [ ’answer’ ] = voted_answer 65 66 return final_response Code 2: Code of the best policy found by Gödel Agent for MGSM. ⬇ 1 2 3 def solver ( agent , task : str ): 4 messages = [{ "role" : "user" , "content" : f "# Your Task:\n{task}" }] 5 response = agent . action_call_json_format_llm ( 6 model = "gpt-3.5-turbo" , 7 messages = messages , 8 temperature =0.5, 9 num_of_response =5, 10 role = "math problem solver" , 11 return_dict_keys =[ "reasoning" , "answer" ], 12 requirements =( 13 "1. Please explain step by step.\n" 14 "2. The answer MUST be an integer.\n" 15 "3. Verify each step before finalizing the answer.\n" 16 ). strip (), 17 ) 18 19 consistent_answer = None 20 answer_count = {} 21 for resp in response : 22 answer = resp . get ( "answer" , "" ) 23 if answer in answer_count : 24 answer_count [ answer ] += 1 25 else : 26 answer_count [ answer ] = 1 27 28 most_consistent_answer = max ( answer_count , key = answer_count . get ) 29 30 for resp in response : 31 if resp . get ( "answer" , "" ) == most_consistent_answer : 32 consistent_answer = resp 33 break 34 35 if consistent_answer is None : 36 consistent_answer = response [0] 37 38 consistent_answer [ "answer" ] = str ( consistent_answer . get ( "answer" , "" )) 39 return consistent_answer Code 3: Code of the best policy found by Gödel Agent for MMLU. ⬇ 1 def solver ( agent , task : str ): 2 # Few-Shot Learning: Providing extended examples to guide the LLM 3 few_shot_examples = [ 4 { ’role’ : ’user’ , ’content’ : ’Question: In the movie Austin Powers: The Spy Who Shagged Me what is the name of Dr. Evil\’s diminutive clone?\nChoices:\n(A) Little Buddy\n(B) Mini-Me\n(C) Small Fry\n(D) Dr Evil Jr’ }, 5 { ’role’ : ’assistant’ , ’content’ : ’In the movie Austin Powers: The Spy Who Shagged Me, Dr. Evil\’s diminutive clone is famously named Mini-Me.\nAnswer: B’ }, 6 \ "" "Three more examples are omitted here to conserve space.\"" " 7 {’role’:’user’, ’content’:’Question: Lorem Ipsum?\nChoices: (A) Lorem\n(B) Ipsum\n(C) Dolor\n(D) Sit Amet’}, 8 {’role’:’assistant’, ’content’:’Answer: A’} 9 ] 10 11 # Integrate the few-shot examples into the conversation 12 messages = few_shot_examples + [{’role’: ’user’, ’content’: f’# Your Task:\n{task}’}] 13 14 # Using self-consistency by generating multiple responses 15 response = agent.action_call_json_format_llm( 16 model=’gpt-3.5-turbo’, 17 messages=messages, 18 temperature=0.8, 19 num_of_response=5, 20 role=’knowledge and reasoning expert’, 21 return_dict_keys=[’reasoning’, ’answer’], 22 requirement
