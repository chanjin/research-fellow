# ProcMEM: Learning Reusable Procedural Memory from Experience via Non-Parametric PPO for LLM Agents

Source URL: https://arxiv.org/html/2602.01869v1

ProcMEM: Learning Reusable Procedural Memory from Experience via Non-Parametric PPO for LLM Agents 
Qirui Mi Affiliation: Institute of Automation, Chinese Academy of Sciences Affiliation: School of Artificial Intelligence, Chinese Academy of Sciences Zhijian Ma Affiliation: Institute of Automation, Chinese Academy of Sciences Affiliation: School of Artificial Intelligence, Chinese Academy of Sciences Mengyue Yang Affiliation: University of Bristol Haoxuan Li Affiliation: Peking University Yisen Wang Affiliation: Peking University Haifeng Zhang Affiliation: Institute of Automation, Chinese Academy of Sciences Correspondence to: haifeng.zhang@ia.ac.cn Jun Wang Affiliation: University College London Correspondence to: jun.wang@ucl.ac.uk Abstract 
LLM-driven agents demonstrate strong performance in sequential decision-making but often rely on on-the-fly reasoning, re-deriving solutions even in recurring scenarios. This insufficient experience reuse leads to computational redundancy and execution instability. To bridge this gap, we propose ProcMEM , a framework that enables agents to autonomously learn procedural memory from interaction experiences without parameter updates. By formalizing a Skill-MDP , ProcMEM transforms passive episodic narratives into executable Skills defined by activation, execution, and termination conditions to ensure executability. To achieve reliable reusability without capability degradation, we introduce Non-Parametric PPO , which leverages semantic gradients for high-quality candidate generation and a PPO Gate for robust Skill verification. Through score-based maintenance, ProcMEM sustains compact, high-quality procedural memory. Experimental results across in-domain, cross-task, and cross-agent scenarios demonstrate that ProcMEM achieves superior reuse rates and significant performance gains with extreme memory compression. Visualized evolutionary trajectories and Skill distributions further reveal how ProcMEM transparently accumulates, refines, and reuses procedural knowledge to facilitate long-term autonomy. 
Keywords: Machine Learning, ICML 
1 Introduction 
Figure 1 : Episodic memory versus procedural memory in LLM-driven agents. Episodic memory retrieves past interactions for reference, requiring inference-heavy reasoning at decision time. Procedural memory encodes reusable executable Skills that directly map situations to actions, enabling efficient experience reuse. 
Large Language Model (LLM)-driven agents have shown strong performance in complex sequential decision-making ( Park et al., 2023 ; Shinn et al., 2023 ) . However, this performance is often driven by on-the-fly reasoning , where agents interpret prompts, observations, and feedback in real-time to generate solutions ( Wei et al., 2022 ; Yao et al., 2022 ; Yao et al., 2023 ) . Even in recurring situations, they typically redo the full reasoning process from scratch, treating each interaction as an unseen problem. This insufficient experience reuse results in substantial computational redundancy and increases the risk of error accumulation in long-horizon scenarios, eventually leading to lower reliability in execution ( Liu et al., 2024 ; Press et al., 2023 ) . 

To incorporate interaction experience, existing work broadly falls into two paradigms. Parametric methods , such as Reinforcement Learning (RL), attempt to encode experience into model parameters ( Sutton et al., 1998 ; Ouyang et al., 2022 ) . While effective in specific domains, these approaches face high training costs, risks of catastrophic forgetting, and a narrowing of general-purpose capabilities ( Kirkpatrick et al., 2017 ) . Alternatively, non-parametric methods improve behavior at inference time without updating the base LLM, most commonly via external memory ( Hu et al., 2025 ) . These agents store diverse forms of experience in external memory, including past trajectories ( Lewis et al., 2020 ; Rajesh et al., 2025 ) , distilled reflections ( Xu et al., 2025 ; Zhao et al., 2024 ) , and structured graphs ( Zhang et al., 2025a ; Xia et al., 2025 ) or workflows ( Wang et al., 2024 ) . At decision time, they retrieve stored experiences to condition the LLM’s reasoning and improve performance, typically without updating the base LLM. However, despite their effectiveness, these methods predominantly operate as forms of episodic memory ( Cohen and Squire, 1980 ) , storing and retrieving past experiences as “history books” to be consulted (Fig. 1 ). Even with large memory, the agent still has to spend its limited context window interpreting retrieved cases and re-deriving solutions, effectively returning to the inference-heavy loop. Inspired by procedural memory in human cognition, an implicit system that directly maps situations to action patterns ( Squire, 2004 ) ; once acquired, it enables the automatic execution of Skills without conscious re-derivation ( Anderson, 1982 ) . While frameworks like Claude Agent Skills ( Anthropic, 2025 ) reuse manually encoded procedures, this work investigates how LLM agents can autonomously learn reusable procedural Skills from interaction experience for future decision-making. 

However, establishing reusable procedural memory faces three fundamental obstacles: C1: Executability. Interaction experience is often stored as passive episodic narratives describing past events rather than active decision procedures that can be directly instantiated at runtime. C2: Reusability. The challenge lies in ensuring that stored procedures can be reliably invoked and reused in future tasks while providing a consistent performance gain. C3: Non-Parametric Optimization. The difficulty lies in learning reusable procedural memory through non-parametric methods while preserving the agent’s general-purpose capabilities. 

To address these challenges, we propose ProcMEM , a framework designed to learn reusable pro cedural mem ory from interaction experience without parameter updates. First, ProcMEM formalizes procedural units as Skills consisting of Activation Conditions, Execution Procedures, and Termination Conditions. By constructing a Skill-MDP , the agent selects and reuses these executable procedures (Skills) for decision-making to ensure executability ( C1 ). To achieve reliable reusability ( C2 ) through non-parametric optimization ( C3 ), we introduce Non-Parametric PPO . This mechanism leverages semantic gradients extracted from batch trajectories to propose refined Skill candidates, while a PPO-style Trust-Region Verification (PPO Gate) ensures the selection of high-quality Skills for inclusion in the procedural memory. Furthermore, an online scoring mechanism filters out redundant or low-quality procedures. Experimental results demonstrate that ProcMEM achieves superior reuse rates, significant performance gains, and extreme memory compression compared to baselines across in-domain, cross-task, and cross-agent scenarios (Table 1 , 2 ). Ablation studies confirm that Semantic Gradients and the PPO Gate are indispensable for generating and verifying high-quality skills, while online scoring preserves long-term evolutionary gains (Table 3 , Fig. 3 ). Finally, visualized evolutionary trajectories (Fig. 4 ) and Skill distribution (Fig. 5 ) reveal how ProcMEM’s procedural memory is transparently constructed and reused to facilitate long-term autonomy. Our core contributions are three-fold: 

• 
Procedural Memory Formalization : We introduce the Skill-MDP, transitioning LLM agents from episodic narratives to reusable Skills. 

• 
Non-Parametric PPO : We propose a parameter-free optimization mechanism leveraging Semantic Gradients , a PPO Gate , and score-based maintenance to evolve high-quality skills without weight updates. 

• 
Superior reuse rates and performance gain across diverse scenarios with extreme memory compression. 1 1 1 Code is available at: https://github.com/Miracle1207/ProcMEM 

Figure 2 : Overview of the ProcMEM framework. (Left) Skill-MDP: The agent selects a Skill ω \omega based on state s t s_{t} and activation conditions. A frozen LLM executes ω \omega into primitive actions a t a_{t} over multiple steps until termination. Post-episode trajectories 𝒯 \mathcal{T} are stored in a buffer. (Middle) Procedural Memory: Skills are dynamically managed via refinement , generation , and score-based pruning to maintain pool quality. (Right) Non-Parametric PPO: Evolution proceeds in two stages: ① Semantic Gradient : Derives and aggregates per-trajectory gradients through hindsight attribution to generate candidates ω ′ \omega^{\prime} . ② PPO Gate : Filters candidates via trust-region verification, admitting only the best-performing valid candidate into the Skill pool. 
2 Related Work 

(A comprehensive literature review is shown in Appendix A .) Learning from Interaction. LLM agents improve decision-making via parametric fine-tuning , such as reinforcement learning ( Ouyang et al., 2022 ; Rafailov et al., 2023 ; Guo et al., 2025 ) , or non-parametric adaptation . While parametric updates yield strong performance, they often incur high computational costs and risk catastrophic forgetting or over-specialization ( Ziegler et al., 2020 ; Shi et al., 2025a ; Luo et al., 2025 ) , driving the shift toward memory-augmented agents as a more efficient, non-parametric alternative. 

Memory-Augmented LLM Agents. Existing frameworks primarily differ in experience representation: (i) Episodic Trajectories: Storing raw trajectories for case-based reasoning ( Park et al., 2023 ) ; (ii) Abstracted Knowledge: Distilling experience into summaries ( Yang et al., 2025 ) , high-level principles ( Wu et al., 2025 ; Agrawal et al., 2025 ; Cai et al., 2025 ) , or failure-derived insights ( Zhao et al., 2024 ) ; (iii) Structured & Compressed Memory: Organizing experience into graphs ( Zhang et al., 2025a ; Jimenez Gutierrez et al., 2024 ; Rezazadeh et al., 2024 ; Xia et al., 2025 ) , dense vectors ( Das et al., 2024 ; Zhang et al., 2025b ) , or dynamic knowledge snippets ( Asai et al., 2024 ; Shi et al., 2025b ; Zhou et al., 2025b ) ; (iv) Workflow: Maintaining explicit task-completion paths ( Wang et al., 2024 ) . Skill-centric and procedural frameworks ( Wang et al., 2023 ; Tan et al., 2024 ; Zhu et al., 2023 ; Sumers et al., 2023 ; Han et al., 2025 ; Fang et al., 2025 ) leverage executable logic, yet robust reusability remains non-trivial. To bridge the gap between storage and reusability, we propose ProcMEM to learn procedural memory for efficient, autonomous long-term execution. 

3 Reusable Procedural Units: Skills 

In this section, we introduce Skills as reusable procedural units integrated into the decision-making process of LLM agents. We define a Skill as a temporally extended procedural units specifying: (1) when it should be activated, (2) how the agent should execute a sequence of actions, and (3) when control should return. 

Unlike human procedural memory ( Cohen and Squire, 1980 ; Squire, 2004 ) , which is largely implicit, our Skills are currently represented in an explicit, readable form, similar to systems such as Claude Agent Skills ( Anthropic, 2025 ) . While explicit in representation, Skills can be hidden at the system level during execution; directions toward implicit procedural representations are discussed in Appendix . 

3.1 Problem Formulation 

We formulate the LLM agent’s decision-making process as a Skill-augmented Markov Decision Process (Skill-MDP) . The Skill-MDP extends the classical Markov Decision Process (MDP) by introducing a dynamic Skill pool Ω \Omega , which explicitly represents the agent’s procedural memory and organizes decision making around the selection and execution of reusable procedural Skills. Formally, a Skill-MDP is defined as a tuple 
ℳ Ω = ( 𝒮 , 𝒜 , Ω , P , R , γ ) , \mathcal{M}_{\Omega}=(\mathcal{S},\mathcal{A},\Omega,P,R,\gamma), 
where 𝒮 \mathcal{S} denotes the semantic state space and 𝒜 \mathcal{A} denotes the primitive action space, both represented in natural language; Ω = { ω ( 1 ) , … , ω ( K ) } \Omega=\{\omega^{(1)},\ldots,\omega^{(K)}\} denotes the pool of available Skills, which serves as the agent’s procedural memory, with K K Skills stored; P P and R R are the state transition and reward functions, respectively, and γ ∈ ( 0 , 1 ] \gamma\in(0,1] is the discount factor. 

In a Skill-MDP, at each time step t t , the agent first selects a Skill via a Skill-selection policy μ \mu , conditioned on the current state s t s_{t} and the current Skill pool Ω \Omega : 
ω t ∼ μ ⁡ ( ω ∣ s t , Ω ) . \omega_{t}\sim\mu(\omega\mid s_{t},\Omega). 
Conditioned on the active Skill ω t \omega_{t} , the agent then generates a primitive action using an LLM-driven action policy: 
a t ∼ π LLM ​ ( a ∣ s t , ω t ) . a_{t}\sim\pi_{\text{LLM}}(a\mid s_{t},\omega_{t}). (1) 
Accordingly, the hierarchical policy over Skills and primitive actions factorizes as 
π Ω ​ ( ω t , a t ∣ s t ) = μ ⁡ ( ω t ∣ s t , Ω ) ​ π LLM ​ ( a t ∣ s t , ω t ) , \pi_{\Omega}(\omega_{t},a_{t}\mid s_{t})\;=\;\mu(\omega_{t}\mid s_{t},\Omega)\,\pi_{\text{LLM}}(a_{t}\mid s_{t},\omega_{t}), (2) 
After executing a t a_{t} , the agent receives reward r t r_{t} and transitions to next state s t + 1 s_{t+1} until the horizon T T is reached. While Eq. ( 2 ) shares a similar factorization with memory-augmented agents (e.g., Memento ( Zhou et al., 2025a ) ), which typically optimize the retrieval policy μ \mu , our work focuses on autonomously evolving the Skill pool Ω \Omega . 

Optimization Objective. We aim to optimize the agent’s decision-making performance under the hierarchical policy π Ω \pi_{\Omega} , measured by the expected cumulative discounted return: 
max π Ω ⁡ 𝔼 ⁡ [ ∑ t = 0 T γ t ​ r t ] . \max_{\pi_{\Omega}}\;\mathbb{E}\left[\sum_{t=0}^{T}\gamma^{t}r_{t}\right]. (3) 
In this paper, we focus on learning the Skill pool Ω \Omega , while keeping the LLM policy and the Skill-selection mechanism fixed. We next specify the internal structure of a Skill, the basic reusable procedural unit in the Skill-MDP. 

3.2 Skill Structure 

We define a Skill ω ∈ Ω \omega\in\Omega as a reusable, natural-language procedural unit that specifies when to activate, how to act while active, and when to return control. This enables Skills to be reused across similar states. Formally, a Skill is defined as the tuple 
ω = ⟨ ℐ ω , π ω , β ω ⟩ , \omega=\langle\mathcal{I}_{\omega},\pi_{\omega},\beta_{\omega}\rangle, 
representing its activation condition, execution procedure, and termination condition. 

(1) Activation Condition ℐ ω \mathcal{I}_{\omega} . The activation condition specifies when a Skill is invoked. Rather than a learned classifier in latent space, ℐ ω \mathcal{I}_{\omega} is a natural-language description of observable context patterns where the Skill applies. During decision making, the Skill-selection policy μ \mu uses ℐ ω \mathcal{I}_{\omega} to select Skills based on the current state or interaction history. 

(2) Execution Procedure π ω \pi_{\omega} . The execution procedure π ω \pi_{\omega} specifies an ordered sequence of actions to be executed while the Skill ω \omega is active, expressed as in natural language. At each time step t t , the LLM generates a primitive action conditioned on the current state s t s_{t} and π ω \pi_{\omega} (Eq. ( 1 )), enabling the agent to reuse procedural steps without re-deriving deliberative reasoning from scratch. 

(3) Termination Condition β ω \beta_{\omega} . The termination condition specifies when the execution of a Skill should end. Like the activation condition, β ω \beta_{\omega} is expressed in natural language and evaluated on the current state: 
β ω ​ ( s t ) = 1 ​ iff ​ s t ​ satisfies ​ β ω . \beta_{\omega}(s_{t})=1\;\;\text{iff }s_{t}\text{ satisfies }\beta_{\omega}. 
If β ω ​ ( s t ) = 1 \beta_{\omega}(s_{t})=1 , the Skill ω \omega terminates and μ \mu selects the next Skill; otherwise, it remains active. 

Example. We illustrate with an example Skill: 
Name: StrategicPlanning Activation Condition ℐ ω \mathcal{I}_{\omega} : When the task begins and no prior information or feedback is available. Execution Procedure π ω \pi_{\omega} : Step 1: Establish an initial hypothesis space based on the task constraints. Step 2: Generate a diverse exploratory action that maximally reduces uncertainty. Termination Condition β ω \beta_{\omega} : Terminate after the first exploratory action is executed and feedback is observed. 
3.3 Skill Selection 

At each decision step t t , the agent maintains a Skill pool Ω t \Omega_{t} . A new Skill is selected by the Skill-selection policy μ \mu only upon termination of the current Skill. In that case, μ \mu selects a Skill ω t ∈ Ω t \omega_{t}\in\Omega_{t} based on the current state s t s_{t} and the pool Ω t \Omega_{t} . We present two simple instantiations of μ \mu . 

(i) Selection by Similarity. This mechanism selects the Skill whose activation condition best matches the state s t s_{t} : 
ω t = arg ⁡ max ω ∈ Ω t ⁡ Sim ⁡ ( s t , ℐ ω ) , \omega_{t}=\arg\max\nolimits_{\omega\in\Omega_{t}}\mathrm{Sim}(s_{t},\mathcal{I}_{\omega}), 
where Sim ⁡ ( s t , ℐ ω ) \mathrm{Sim}(s_{t},\mathcal{I}_{\omega}) measures the similarity between the current state and the activation condition ℐ ω \mathcal{I}_{\omega} . The similarity function Sim ⁡ ( ⋅ , ⋅ ) \mathrm{Sim(\cdot,\cdot)} can be implemented via cosine similarity over embeddings, or LLM as judge. 

(ii) Selection by Value. We also support value-based selection to prioritize Skills with higher expected return. We first form a candidate set of the top- k k Skills by similarity: 
Ω t ( k ) = TopK ω ∈ Ω t ⁡ Sim ⁡ ( s t , ℐ ω ) , \Omega_{t}^{(k)}=\operatorname{TopK}\nolimits_{\omega\in\Omega_{t}}\mathrm{Sim}(s_{t},\mathcal{I}_{\omega}), 
where Ω t ( k ) ⊆ Ω t \Omega_{t}^{(k)}\subseteq\Omega_{t} and | Ω t ( k ) | = k |\Omega_{t}^{(k)}|=k . We then select the Skill with the highest estimated value in this set: 
ω t = arg ⁡ max ω ∈ Ω t ( k ) ⁡ Q ⁡ ( s t , ω ) . \omega_{t}=\arg\max\nolimits_{\omega\in\Omega_{t}^{(k)}}Q(s_{t},\omega). 
Here Q ⁡ ( s t , ω ) Q(s_{t},\omega) denotes an estimate of the expected return obtained by invoking Skill ω \omega from state s t s_{t} . 

These mechanisms are simple instantiations of μ \mu and can be replaced by more advanced Skill retrieval policies, e.g., RL-based methods ( Zhou et al., 2025a ; Zhang et al., 2026 ) . Our focus lies on Skill reuse and evolution. 

3.4 Skill Pool Evolution 

The Skill pool is learned from interaction experience and evolves over time. In our framework, the pool is updated using trajectories generated under the current policy. Given a batch of trajectories 𝒯 ( B ) \mathcal{T}^{(B)} collected under the Skill-augmented policy π Ω \pi_{\Omega} , we define a Skill pool evolution operator ℰ \mathcal{E} : 
Ω new = ℰ ⁡ ( Ω old , 𝒯 ( B ) ) , \Omega_{\text{new}}\;=\;\mathcal{E}(\Omega_{\text{old}},\mathcal{T}^{(B)}), (4) 
which synthesizes new Skills, refines existing Skills, and prunes those that underperform empirically. 

This paper focuses on Skill pool evolution. The LLM action policy π LLM \pi_{\text{LLM}} and the Skill-selection policy μ \mu are kept fixed, and learning proceeds through repeated application of ℰ \mathcal{E} . Under this setting, optimizing Eq. ( 3 ) reduces to optimizing the evolution operator ℰ \mathcal{E} : 
max ℰ ⁡ 𝔼 τ ∼ π Ω ∗ ​ [ ∑ t = 0 T γ t ​ r t ] , where ​ Ω ∗ = ℰ ( N ) ​ ( Ω 0 ) . \max_{\mathcal{E}}\;\mathbb{E}_{\tau\sim\pi_{\Omega^{*}}}\left[\sum_{t=0}^{T}\gamma^{t}r_{t}\right],\quad\text{where }\Omega^{*}=\mathcal{E}^{(N)}(\Omega_{0}). (5) 
Here, ℰ ( N ) \mathcal{E}^{(N)} denotes N N successive applications of ℰ \mathcal{E} , each using newly collected experience. 

4 Non-Parametric PPO for Skill Evolution 

In this section, we present a Proximal Policy Optimization (PPO)-inspired non-parametric method for Skill pool evolution, which we refer to as Non-Parametric PPO . This method leverages PPO-style trust-region principles to realize reliable Skill pool evolution as defined in Eq. ( 4 ), without updating any LLM parameters. 

Standard PPO ( Schulman et al., 2017 ) improves a parameterized stochastic policy through gradient-based optimization of a clipped surrogate objective. In contrast, our Non-Parametric PPO replaces parameter updates with Skill refinement, and consists of two components: (1) generating candidate Skills via semantic gradients extracted from trajectories, and (2) accepting candidates only if they satisfy a PPO-style trust-region verification under the frozen LLM policy. The complete procedure is shown in Algorithm 1 . 

4.1 Semantic Gradients 

To learn without updating LLM parameters, we introduce Semantic Gradients as learning signals that specify how a Skill should be refined. 

Per-trajectory semantic gradients. Unlike TextGrad ( Yuksekgonul et al., 2024 ) , which uses automatic differentiation to optimize static variables for instantaneous response quality, our semantic gradients are designed for sequential decision making. These gradients provide natural-language update directions extracted from interaction trajectories, indicating how a Skill’s activation, execution, and termination conditions should be refined via hindsight attribution . Consider a Skill ω = ⟨ ℐ ω , π ω , β ω ⟩ \omega=\langle\mathcal{I}_{\omega},\pi_{\omega},\beta_{\omega}\rangle and trajectories 𝒯 ( B ) \mathcal{T}^{(B)} where it is invoked. For each trajectory τ i \tau_{i} , we analyze the segment controlled by ω \omega and attribute the outcome to its activation condition, execution procedure, or termination condition. This yields a structured semantic gradient 
g i = ∇ sem ( τ i , ω ) = ( g i ( ℐ ) , g i ( π ) , g i ( β ) ) . g_{i}=\nabla_{\mathrm{sem}}(\tau_{i},\omega)=\bigl(g_{i}^{(\mathcal{I})},\,g_{i}^{(\pi)},\,g_{i}^{(\beta)}\bigr). 
where each component is a natural-language refinement suggestion for the corresponding Skill component. Intuitively, g i g_{i} serves as a local update direction for Skill ω \omega induced by trajectory τ i \tau_{i} . An example of semantic gradients from our experiments is given in the Appendix D . 

Batch-level aggregation. Individual trajectories may provide inconsistent update signals. To obtain a stable learning signal, we aggregate semantic gradients across the batch: 
g ¯ ω = Aggregate ​ ( { g i } i = 1 B ) , \bar{g}_{\omega}=\text{Aggregate}\bigl(\{g_{i}\}_{i=1}^{B}\bigr), 
where Aggregate ​ ( ⋅ ) \text{Aggregate}(\cdot) denotes an LLM-based consolidation procedure that extracts recurring failure patterns and consistent refinement suggestions across trajectories, while filtering out conflicting or trajectory-specific signals. The resulting g ¯ ω = ( g ¯ ( ℐ ) , g ¯ ( π ) , g ¯ ( β ) ) \bar{g}_{\omega}=(\bar{g}^{(\mathcal{I})},\bar{g}^{(\pi)},\bar{g}^{(\beta)}) represents a batch-averaged semantic gradient that captures systematic weaknesses of Skill ω \omega revealed by experience. 

Semantic Skill Update. We update Skill ω \omega using the batch-averaged semantic gradient to obtain a candidate Skill ω ′ \omega^{\prime} : 
ω ′ = ω ⊕ g ¯ ω , \omega^{\prime}=\omega\oplus\bar{g}_{\omega}, 
where ⊕ \oplus denotes an LLM-driven update operation that revises ℐ ω \mathcal{I}_{\omega} , π ω \pi_{\omega} , and β ω \beta_{\omega} according to the refinement directions encoded in g ¯ ω \bar{g}_{\omega} , while preserving the overall Skill structure. This operation plays the role of a gradient ascent step in Non-Parametric PPO: instead of updating numerical parameters, the Skill is updated along a direction intended to improve expected return, as suggested by aggregated hindsight feedback. 

4.2 PPO-Style Trust-Region Verification 

Semantic-gradient updates are generated by an LLM from hindsight feedback. As a result, they may extrapolate beyond the observed interaction data and introduce hallucinated or behaviorally unstable Skills. To mitigate this risk, we introduce a PPO-style trust-region verification step to evaluate each candidate Skill before adding it to the pool. 

We treat the frozen LLM as the underlying stochastic policy and evaluate a candidate Skill ω ′ \omega^{\prime} using batch-size trajectories collected under the previous Skill ω \omega . For each timestep t t , we compute an importance ratio 
ρ t ​ ( ω ′ ) = π LLM ​ ( a t ∣ s t , ω ′ ) π LLM ​ ( a t ∣ s t , ω ) , \rho_{t}(\omega^{\prime})=\frac{\pi_{\text{LLM}}(a_{t}\mid s_{t},\omega^{\prime})}{\pi_{\text{LLM}}(a_{t}\mid s_{t},\omega)}, (6) 
which measures how the likelihood of the historical action a t a_{t} would change if the candidate Skill ω ′ \omega^{\prime} were applied instead of the behavior Skill ω \omega at the same state s t s_{t} . Since we do not train a value function, we estimate advantages using return-to-go with a running baseline: 

G t = ∑ k = t T − 1 γ k − t ​ r k , A ^ t = G t − R ¯ , \textstyle G_{t}=\sum_{k=t}^{T-1}\gamma^{k-t}r_{k},\hskip 9.24994pt\hat{A}_{t}=G_{t}-\bar{R}, 

where R ¯ \bar{R} is a running baseline used to reduce variance. We compute a PPO-style clipped surrogate verification functional , hereafter referred to as the PPO Gate , to evaluate the counterfactual advantage of applying the candidate Skill ω ′ \omega^{\prime} on historical trajectories: 
L CLIP ​ ( ω ′ ) = \displaystyle L^{\text{CLIP}}(\omega^{\prime})= 𝔼 ^ τ ∼ ℬ ​ [ 1 | τ | ​ ∑ t ∈ τ min ⁡ ( ρ t ​ ( ω ′ ) ​ A ^ t , clip ⁡ ( ρ t ​ ( ω ′ ) , 1 − ϵ , 1 + ϵ ) ​ A ^ t ) ] . \displaystyle\hat{\mathbb{E}}_{\tau\sim\mathcal{B}}\left[\frac{1}{|\tau|}\sum_{t\in\tau}\min\!\left(\rho_{t}(\omega^{\prime})\hat{A}_{t},\;\operatorname{clip}(\rho_{t}(\omega^{\prime}),1-\epsilon,1+\epsilon)\,\hat{A}_{t}\right)\right]. 
This verification functional favors candidate Skills that assign higher probability to high-advantage actions observed in past trajectories, while limiting large deviations from the behavior policy, thereby enforcing a trust-region constraint. 

Best-of- N c N_{c} Selection. Given N c N_{c} candidates generated via semantic gradients, we compute the PPO Gate score J ⁡ ( ω ′ ) ≜ L CLIP ​ ( ω ′ ) J(\omega^{\prime})\triangleq L^{\text{CLIP}}(\omega^{\prime}) for each and select the best candidate: 
ω new = arg ⁡ max ω ′ ⁡ J ⁡ ( ω ′ ) , subject to J ⁡ ( ω new ) > 0 . \textstyle\omega_{\text{new}}=\arg\max_{\omega^{\prime}}J(\omega^{\prime}),\quad\text{subject to}\quad J(\omega_{\text{new}})>0. 
Since the PPO Gate is based on advantage estimates, a positive score indicates that the candidate is expected to outperform the previous Skill under the trust-region constraint, filtering out unreliable or hallucinated candidates to prevent destabilizing shifts. 

4.3 Score-Based Skill Pool Maintenance 

Since the Skill pool has a fixed capacity K K , indiscriminately adding new Skills increases storage and selection overhead. Beyond PPO Gate, we retain or prune Skills based on their empirical contribution, measured by an online score. During interaction, multiple Skills may be invoked within a trajectory, and rewards may be provided at different granularities. We therefore define a unified, advantage-style Skill gain. Given a trajectory τ \tau with rewards { r t } \{r_{t}\} , we first form a per-step advantage signal r ~ t ≜ r t − r ¯ \tilde{r}_{t}\triangleq r_{t}-\bar{r} , where r ¯ \bar{r} is a running baseline. The gain of Skill ω \omega in τ \tau is defined as the average advantage accumulated over the time steps during which ω \omega is active: 
G ⁡ ( ω , τ ) = 1 | 𝒯 ω ​ ( τ ) | ​ ∑ t ∈ 𝒯 ω ​ ( τ ) r ~ t , G(\omega;\tau)=\frac{1}{|\mathcal{T}_{\omega}(\tau)|}\sum_{t\in\mathcal{T}_{\omega}(\tau)}\tilde{r}_{t}, (7) 
where 𝒯 ω ​ ( τ ) \mathcal{T}_{\omega}(\tau) denotes the set of time steps when Skill ω \omega is executed. When only a trajectory-level return R ⁡ ( τ ) R(\tau) is available, we set r ~ t ≡ ( R ⁡ ( τ ) − R ¯ ) / | τ | \tilde{r}_{t}\equiv(R(\tau)-\bar{R})/|\tau| , yielding the same advantage-style definition. 

Online score update. For each Skill ω \omega , we maintain a cumulative gain G b ​ ( ω ) G_{b}(\omega) and an invocation count N b ​ ( ω ) N_{b}(\omega) . After processing a batch 𝒯 ( b ) \mathcal{T}^{(b)} , we update the online score: 
G b + 1 = G b + \displaystyle G_{b+1}=G_{b}+ ∑ G ⁡ ( ω , τ ) , N b + 1 = N b + ∑ c ⁡ ( ω , τ ) , \displaystyle\sum G(\omega;\tau),\hskip 9.24994ptN_{b+1}=N_{b}+\sum c(\omega;\tau), Score b + 1 = G b + 1 max ⁡ ( 1 , N b + 1 ) . \displaystyle\text{Score}_{b+1}=\frac{G_{b+1}}{\max(1,N_{b+1})}. 
Online Score-Based Pruning. To enforce the fixed pool capacity, we maintain the Skill pool using online scores. Specifically, we remove (i) Skills with non-positive online score, i.e., Score ​ ( ω ) ≤ 0 \text{Score}(\omega)\leq 0 , which indicates no expected advantage over existing Skills, and (ii) duplicate or semantically redundant Skills. If the pool still exceeds capacity, we further prune Skills in ascending order of online score. As the baseline improves over time, this rule imposes evolutionary pressure, phasing out obsolete Skills while retaining those with consistently positive gains. 
Table 1 : Main Results on Memory Reuse and Efficiency. Results are reported as Mean ± Std Dev \text{Mean}_{\pm\text{Std Dev}} . Unit types in “Avg Tokens Per Unit” denote T: Trajectory, I: Insights, N: Notes, W: Workflow, PT: Part of Trajectory, and Skill (ours). ↑ \uparrow ( ↓ \downarrow ) indicates higher (lower) is better. Method Experience Reuse Rate ( ↑ \uparrow ) Efficiency Metrics ( ↓ \downarrow ) In-domain Cross-task Reuse Cross-agent Reuse Storage Cost Execution Cost Mastermind -v0 Mastermind -v0-Hard Mastermind -v0-Extreme Gemma-3 -4B Qwen3 -32B Total Stored Tokens Avg Tokens per Unit Δ \Delta Prompt Tokens/Step Retrieval Ratio RAG 0.349 ±0.145 0.441 ±0.002 0.467 ±0.050 0.111 ±0. 0.146 ±0.064 116527 2675 ±414 (T) 2698 ±414 1 ±0. Expel 0.285 ±0.015 0.242 ±0.024 0.258 ±0.016 0.254 ±0.013 0.270 ±0.017 294447 642 ±0 (I) 4568 ±2541 (T) 5210 ±2541 1 ±0. A-MEM 0.020 ±0.005 0.017 ±0.002 0.015 ±0.002 0.020 ±0.003 0.018 ±0.003 200129 1210 ±3 (N) 1214 ±3 1 ±0. AWM 0.080 ±0.010 0.063 ±0.006 0.075 ±0.007 0.073 ±0.006 0.060 ±0.010 391706 602 ±0 (W) 2914 ±297 (T) 3658 ±21 0.049 ±0.009 G-Memory 0.091 ±0.027 0.170 ±0.063 0.092 ± 0.016 0.092_{\pm 0.016} 0.360 ± 0.162 0.360_{\pm 0.162} 0.264 ± 0.104 0.264_{\pm 0.104} 40510 100 ±79 (I) 334 ±2 (PT) 434 ±79 0.097 ±0.027 ProcMEM (Ours) 0.925 ±0.061 0.825 ±0.061 0.900 ±0.094 0.850 ±0.094 0.875 ±0.112 816 102 ±0 (Skill) 273 ±5 0.591 ±0.016 Table 2 : Main Performance Results. Results are Mean ± Std Dev \text{Mean}_{\pm\text{Std Dev}} ; bold denotes best. Shading blue is normalized per column, indicating relative performance within each task. Both sections evaluate the performance of memory reuse or reasoning baselines: the left across cross-task difficulties with a fixed backbone; the right across cross-agent LLM backbones. Algorithm ALFWorld (Success Rate ↑ \uparrow ) Mastermind (Avg Return ↑ \uparrow ) Cross-agent (Mastermind-v0) Train OOD v0 Hard Extreme Gemma-3 4B-it Qwen3 32B Llama-3.3 70B State 0.312 ± 0.040 0.312_{\pm 0.040} 0.262 ± 0.062 0.262_{\pm 0.062} 0.388 ± 0.236 0.388_{\pm 0.236} 0.336 ± 0.183 0.336_{\pm 0.183} 0.272 ± 0.129 0.272_{\pm 0.129} 0.414 ± 0.101 0.414_{\pm 0.101} 0.497 ± 0.159 0.497_{\pm 0.159} 0.613 ± 0.201 0.613_{\pm 0.201} RAG 0.480 ± 0.134 0.480_{\pm 0.134} 0.402 ± 0.264 0.402_{\pm 0.264} 0.521 ± 0.236 0.521_{\pm 0.236} 0.344 ± 0.159 0.344_{\pm 0.159} 0.241 ± 0.136 0.241_{\pm 0.136} 0.404 ± 0.191 0.404_{\pm 0.191} 0.558 ± 0.204 0.558_{\pm 0.204} 0.620 ± 0.211 0.620_{\pm 0.211} CoT 0.600 ± 0.069 0.600_{\pm 0.069} 0.620 ± 0.068 0.620_{\pm 0.068} 0.531 ± 0.063 0.531_{\pm 0.063} 0.381 ± 0.043 0.381_{\pm 0.043} 0.254 ± 0.031 0.254_{\pm 0.031} 0.417 ± 0.120 0.417_{\pm 0.120} 0.470 ± 0.153 0.470_{\pm 0.153} 0.542 ± 0.206 0.542_{\pm 0.206} ReAct 0.580 ± 0.070 0.580_{\pm 0.070} 0.640 ± 0.068 0.640_{\pm 0.068} 0.557 ± 0.059 0.557_{\pm 0.059} 0.405 ± 0.074 0.405_{\pm 0.074} 0.263 ± 0.048 0.263_{\pm 0.048} 0.408 ± 0.131 0.408_{\pm 0.131} 0.425 ± 0.125 0.425_{\pm 0.125} 0.604 ± 0.230 0.604_{\pm 0.230} Expel 0.680 ± 0.065 0.680_{\pm 0.065} 0.740 ± 0.063 0.740_{\pm 0.063} 0.424 ± 0.033 0.424_{\pm 0.033} 0.305 ± 0.031 0.305_{\pm 0.031} 0.239 ± 0.024 0.239_{\pm 0.024} 0.429 ± 0.117 0.429_{\pm 0.117} 0.483 ± 0.185 0.483_{\pm 0.185} 0.575 ± 0.27 0.575_{\pm 0.27} A-MEM 0.520 ± 0.071 0.520_{\pm 0.071} 0.640 ± 0.068 0.640_{\pm 0.068} 0.471 ± 0.042 0.471_{\pm 0.042} 0.310 ± 0.038 0.310_{\pm 0.038} 0.253 ± 0.026 0.253_{\pm 0.026} 0.388 ± 0.115 0.388_{\pm 0.115} 0.570 ± 0.230 0.570_{\pm 0.230} 0.542 ± 0.162 0.542_{\pm 0.162} AWM 0.700 ± 0.065 0.700_{\pm 0.065} 0.900 ± 0.042 0.900_{\pm 0.042} 0.546 ± 0.052 0.546_{\pm 0.052} 0.299 ± 0.036 0.299_{\pm 0.036} 0.294 ± 0.040 0.294_{\pm 0.040} 0.417 ± 0.144 0.417_{\pm 0.144} 0.592 ± 0.183 0.592_{\pm 0.183} 0.550 ± 0.238 0.550_{\pm 0.238} G-Memory 0.681 ± 0.079 0.681_{\pm 0.079} 0.812 ± 0.017 0.812_{\pm 0.017} 0.577 ± 0.052 0.577_{\pm 0.052} 0.406 ± 0.056 0.406_{\pm 0.056} 0.356 ±0.036 0.428 ± 0.039 0.428_{\pm 0.039} 0.475 ± 0.190 0.475_{\pm 0.190} 0.535 ± 0.079 0.535_{\pm 0.079} ProcMEM 0.900 ±0.300 0.909 ±0.287 0.606 ±0.234 0.463 ±0.210 0.333 ±0.118 0.444 ±0.161 0.615 ±0.290 0.647 ±0.236 
5 Experiments 

To examine whether ProcMEM learns reusable procedural memory (instantiated as Skills) from interaction experience, we conduct the following experiments: 

• 
RQ1: Is the learned procedural memory efficiently reusable? (§ 5.2 ) 

• 
RQ2: How do different components contribute to learning reusable procedural memory? (§ 5.3 ) 

• 
RQ3: How does procedural memory evolve and get reused in practice? (§ 5.4 ) 

5.1 Experimental Setup 

Benchmarks. We conduct experiments on ALFWorld ( Shridhar et al., 2021 ) and TextArena ( Guertler et al., 2025 ) , two canonical benchmarks for multi-turn sequential decision-making. ALFWorld separates training from out-of-distribution environments, while Mastermind-v0 from TextArena spans three difficulty tiers; both supporting evaluation of cross-task memory reuse (see Appendix B.1 ). 

Baselines. We compare ProcMEM against a diverse set of memory-augmented and reasoning-based LLM agents. Memory-augmented baselines span raw trajectory retrieval ( RAG ( Lewis et al., 2020 ) ), distilled insights ( Expel ( Zhao et al., 2024 ) ), concise notes ( A-MEM ( Xu et al., 2025 ) ), structured workflows ( AWM ( Wang et al., 2024 ) ), and hybrid memory representations ( G-Memory ( Zhang et al., 2025a ) ). We further include representative reasoning-based baselines, including ReAct ( Yao et al., 2022 ) and CoT ( Wei et al., 2022 ) , as well as a State -based agent without external memory. All methods use the same frozen LLM, ensuring a fair comparison without parameter fine-tuning. 

LLM Backbones. We evaluate ProcMEM with multiple LLM backbones. On TextArena, we learn Skills using Gemma-2-9B and evaluate their reuse across heterogeneous agents, including Gemma-3-4B , Qwen3-32B , and LLaMA-3.3-70B-Instruct . On ALFWorld, all experiments are conducted with Qwen3-32B . 

Evaluation Metrics. We evaluate all methods along three complementary dimensions: (1) Memory reuse is measured by in-domain, cross-task, and cross-agent reuse rates , indicating probability that stored memory is reused per episode. (2) Performance is measured by an agent’s episodic return. (3) Efficiency evaluates the storage cost and inference cost of memory reuse, measured by total tokens stored in memory (Total Stored Tokens), average tokens per memory units (Avg Tokens per Unit), additional prompt tokens added to the decision prompt ( Δ \Delta Prompt Tokens/Step), and the probability of retrieving memory at each step (Retrieval Ratio). Metric details are provided in Appendix B.4 . 

5.2 Does ProcMEM Truly Enable Reusability? 

We evaluate ProcMEM and baseline memory methods on memory reuse and efficiency (Table 1 ), as well as task performance (Table 2 ). Memories are built on in-domain tasks (Mastermind-v0 and ALFWorld-Train) and reused on out-of-distribution or higher-difficulty tasks (cross-task), and across agents with different LLM backbones (cross-agent); results are averaged over 50 episodes per setting. 

ProcMEM’s superior reuse rates validate both Skill-MDP effectiveness and learned Skills’ quality. As shown in Table 1 , ProcMEM’s reuse rate consistently outperforms all baselines in in-domain, cross-task, and cross-agent evaluations. While baselines suffer from low reuse due to redundant episodic data, ProcMEM’s high reuse rate demonstrates that learned procedural Skills are both high-quality and inherently generalizable. 

ProcMEM’s high efficiency is evidenced by minimal storage and low execution overhead. While baselines accumulate hundreds of thousands of tokens by storing diverse episodic units, such as trajectories, insights, and workflows, ProcMEM maintains only 816 tokens , demonstrating that procedural memory is far more compact than episodic memory. ProcMEM’s lean representation, as reflected by Δ \Delta Prompt Tokens/Step, prevents prompt bloat and minimizes LLM execution load. Furthermore, by utilizing temporally extended “Skills”, ProcMEM reduces per-step retrieval ratio, ensuring highly efficient agent execution. 

ProcMEM achieves superior performance despite a highly compressed memory footprint. As shown in Table 2 , our learned memory consistently yields performance gains when reused across varying task difficulties and LLM backbones of different scales. Notably, even under extreme memory compression, ProcMEM maintains the highest success rates, reaching 0.90 in ALFWorld. This superior performance confirms that our framework successfully captures essential task logic, ensuring that only high-quality, decision-critical content is stored. 

5.3 Why Does ProcMEM Work? 

To evaluate the contribution of each component, we conduct an ablation study comparing ProcMEM against several variants, primarily using the Mastermind-v0 environment in TextArena. Beyond performance and reuse rate, we introduce two metrics: Online Score (average Skill quality in the pool) and PPO Gate Pass Rate (the ratio of candidates satisfying PPO Gate). 

• 
w/o Skill : Utilizes only states for decision-making. 

• 
w/o NP-PPO : Employs fixed skill seeds without the NP-PPO evolution process. 

• 
w/o SG : Replaces Semantic Gradients with trajectory summaries; directly utilizing raw trajectories would otherwise trigger context window overflow for Gemma-2-9B . 

• 
w/o PPO Gate : Removes the PPO Gate, allowing all generated candidates to enter the skill pool unconditionally. 

• 
w/o Score (FIFO) : Replaces score-based pruning with a First-In-First-Out (FIFO) to manage pool capacity. 

Both the procedural Skill and NP-PPO evolution are fundamental to task success. As shown in Table 3 , the w/o Skill variant suffers a sharp performance drop ( 0.606 → 0.388 0.606\rightarrow 0.388 ),confirming that skills are essential building blocks for complex decision-making. While initial seeds provide a functional baseline, the w/o NP-PPO results highlight that our evolution mechanism is critical for refining general seeds into task-specific expertise, significantly boosting both reuse and success rates. 

Semantic Gradients and the PPO Gate are indispensable for the generation and verification of high-quality Skills. Ablating either component degrades pool quality. Specifically, w/o SG triggers a 30% drop in the PPO Gate Pass Rate, proving that Semantic Gradients significantly enhance the quality of generated skill candidates. Conversely, w/o PPO Gate admits all candidates without verification, destabilizing training as evidenced in the training curves (Fig. 3 ). Notably, w/o SG remains more stable than w/o PPO Gate , as its candidates must still pass trust-region verification. 

Score-based maintenance is critical for preserving evolutionary gains within the skill pool. The w/o Score (FIFO) variant exhibits the most severe degradation among all NP-PPO ablations. Despite maintaining a high PPO Gate Pass Rate, FIFO inadvertently replaces high-performing Skills with unproven newcomers. The resulting negative Online Score ( − 0.0018 -0.0018 ) confirms that without score-based pruning, the pool fails to retain superior procedural knowledge, ultimately leading to a collapse in long-term performance. 
Table 3 : Ablation Study on ProcMEM . Subscripts show relative change to the Full version (blue for degradation). Reuse Perfor- Online PPO Gate Methods Rate ( ↑ \uparrow ) mance ( ↑ \uparrow ) Score ( ↑ \uparrow ) Pass Rate ( ↑ \uparrow ) ProcMEM (Full) 0.925 0.606 0.0406 59.49% w/o Skill N/A 0.388 (-36.0%) {}_{\text{\color[rgb]{0.2,0.2,1}(-36.0\%)}} N/A N/A w/o NP-PPO 0.563 (-39.1%) {}_{\text{\color[rgb]{0.2,0.2,1}(-39.1\%)}} 0.482 (-20.5%) {}_{\text{\color[rgb]{0.2,0.2,1}(-20.5\%)}} 0.0265 (-34.7%) {}_{\text{\color[rgb]{0.2,0.2,1}(-34.7\%)}} N/A Ablation on NP-PPO w/o SG 0.306 (-66.9%) {}_{\text{\color[rgb]{0.2,0.2,1}(-66.9\%)}} 0.530 (-12.5%) {}_{\text{\color[rgb]{0.2,0.2,1}(-12.5\%)}} 0.0015 (-96.3%) {}_{\text{\color[rgb]{0.2,0.2,1}(-96.3\%)}} 41.54% (-30.2%) {}_{\text{\color[rgb]{0.2,0.2,1}(-30.2\%)}} w/o PPO Gate 0.222 (-76.0%) {}_{\text{\color[rgb]{0.2,0.2,1}(-76.0\%)}} 0.453 (-25.2%) {}_{\text{\color[rgb]{0.2,0.2,1}(-25.2\%)}} 0.0011 (-97.3%) {}_{\text{\color[rgb]{0.2,0.2,1}(-97.3\%)}} 100.00% (+68.1%) {}_{\text{\color[rgb]{0.4,0.4,0.4}(+68.1\%)}} w/o Score (FIFO) 0.131 (-85.8%) {}_{\text{\color[rgb]{0.2,0.2,1}(-85.8\%)}} 0.439 (-27.6%) {}_{\text{\color[rgb]{0.2,0.2,1}(-27.6\%)}} -0.0064 (-115.8%) {}_{\text{\color[rgb]{0.2,0.2,1}(-115.8\%)}} 57.18% (-3.9%) {}_{\text{\color[rgb]{0.2,0.2,1}(-3.9\%)}} Standard deviations are provided in Table 4 in Appendix. Figure 3 : Training curves of ProcMEM and ablation variants. Solid lines and shaded areas denote the smoothed mean and standard deviation of average returns, respectively. 
5.4 How Does ProcMEM Evolve and Reuse? 
Figure 4 : Evolutionary Lineage of Skills. Gray bars represent Skill lifespans along the evolutionary timeline (horizontal axis). Dashed vertical lines denote refinement events where a parent Skill evolves into children; multiple lines indicate repeated refinements. Red ‘X’ markers signify pruning of underperforming variants for pool efficiency. The dark blue arrow and sequential alignment (e.g., v 1 → v 13 v_{1}\to v_{13} ) track the sustained trajectory of Skill evolution. Figure 5 : Skill distribution across LLM agents and task complexities. Bars represent the empirical invocation probability for each skill categorized by different LLM backbones (top) and task difficulty levels in Mastermind-v0 (bottom). N N denotes the average number of skill invocations per episode. 
Cross-Agent and Cross-Task Generalization. Fig. 5 characterizes skill reuse through invocation probability and mean frequency per episode ( N N ). Our analysis reveals that while different LLM backbones exhibit distinct usage profiles, the underlying selection patterns remain remarkably stable across varying task difficulties. Specifically, in cross-agent evaluations, Gemma2-9B shows a heightened reliance on FBInference , whereas StratPlan maintains consistent activation levels across agents, acting as a standardized procedural primitive. Crucially, in cross-task evaluations on Mastermind-v0 , the skill distribution remains invariant as difficulty scales from Base to Extreme. This consistency demonstrates that ProcMEM effectively distills the fundamental task logic, enabling robust generalization across environment complexities. 

Evolutionary Dynamics. Fig. 4 illustrates the evolutionary trajectory of two representative Skills. The observed refinements and score-based pruning events underscore ProcMEM’s ability to evolve a compact, high-utility skill pool. Detailed analysis is provided in Appendix C.1 . 

6 Conclusion 

We presented ProcMEM, a framework enabling LLM agents to autonomously learn procedural memory without parameter updates. By formalizing the Skill-MDP , ProcMEM transforms passive episodic narratives into executable, reusable Skills, eliminating redundant on-the-fly reasoning. To ensure reliability without capability degradation, we propose Non-Parametric PPO , which leverages semantic gradients for high-quality candidate generation and a PPO Gate for robust Skill verification. Finally, a score-based maintenance mechanism prunes low-return Skills to sustain long-term memory quality. 

Results across diverse scenarios confirm that ProcMEM achieves superior reuse rates and significant performance gains with extreme memory compression . Our findings validate that high-quality procedural memory is fundamentally more efficient than raw episodic storage for long-term autonomy. Future work will integrate implicit execution modules to better emulate human-like intelligence. Ultimately, the autonomous accumulation of procedural expertise via interaction without parameter updates represents a pivotal milestone toward the emergence of truly self-evolving artificial intelligence. 

Impact Statement 

This paper presents work whose goal is to advance the field of machine learning by improving the efficiency and reusability of autonomous agents. By enabling procedural knowledge accumulation without continuous parameter updates, our work contributes to more resource-efficient and sustainable AI development. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here. 

References 

Agrawal et al. (2025) L. A. Agrawal, S. Tan, D. Soylu, N. Ziems, R. Khare, K. Opsahl-Ong, A. Singhvi, H. Shandilya, M. J. Ryan, M. Jiang, et al. Gepa: reflective prompt evolution can outperform reinforcement learning . arXiv preprint arXiv:2507.19457 . Cited by: Appendix A , §2 . 

Anderson (1982) J. R. Anderson Acquisition of cognitive skill. . Psychological review 89 ( 4 ), pp. 369 . Cited by: §1 . 

Anthropic (2025) Anthropic Agent skills . External Links: Link Cited by: §1 , §3 . 

Asai et al. (2024) A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi Self-rag: learning to retrieve, generate, and critique through self-reflection . In The Twelfth International Conference on Learning Representations , Cited by: Appendix A , §2 . 

Cai et al. (2025) Z. Cai, X. Guo, Y. Pei, J. Feng, J. Chen, Y. Zhang, W. Ma, M. Wang, and H. Zhou Flex: continuous agent evolution via forward learning from experience . arXiv preprint arXiv:2511.06449 . Cited by: Appendix A , §2 . 

Cohen and Squire (1980) N. J. Cohen and L. R. Squire Preserved learning and retention of pattern-analyzing skill in amnesia: dissociation of knowing how and knowing that . Science 210 ( 4466 ), pp. 207–210 . Cited by: §1 , §3 . 

Das et al. (2024) P. Das, S. Chaudhury, E. Nelson, I. Melnyk, S. Swaminathan, S. Dai, A. Lozano, G. Kollias, V. Chenthamarakshan, S. Dan, et al. Larimar: large language models with episodic memory control . arXiv preprint arXiv:2403.11901 . Cited by: Appendix A , §2 . 

Fang et al. (2025) R. Fang, Y. Liang, X. Wang, J. Wu, S. Qiao, P. Xie, F. Huang, H. Chen, and N. Zhang Memp: exploring agent procedural memory . arXiv preprint arXiv:2508.06433 . Cited by: Appendix A , §2 . 

Guertler et al. (2025) L. Guertler, B. Cheng, S. Yu, B. Liu, L. Choshen, and C. Tan TextArena . External Links: 2504.11442 , Link Cited by: §B.1 , §5.1 . 

Guo et al. (2025) D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, X. Zhang, X. Yu, Y. Wu, Z. F. Wu, Z. Gou, Z. Shao, Z. Li, Z. Gao, A. Liu, B. Xue, B. Wang, B. Wu, B. Feng, C. Lu, C. Zhao, C. Deng, C. Ruan, D. Dai, D. Chen, D. Ji, E. Li, F. Lin, F. Dai, F. Luo, G. Hao, G. Chen, G. Li, H. Zhang, H. Xu, H. Ding, H. Gao, H. Qu, H. Li, J. Guo, J. Li, J. Chen, J. Yuan, J. Tu, J. Qiu, J. Li, J. L. Cai, J. Ni, J. Liang, J. Chen, K. Dong, K. Hu, K. You, K. Gao, K. Guan, K. Huang, K. Yu, L. Wang, L. Zhang, L. Zhao, L. Wang, L. Zhang, L. Xu, L. Xia, M. Zhang, M. Zhang, M. Tang, M. Zhou, M. Li, M. Wang, M. Li, N. Tian, P. Huang, P. Zhang, Q. Wang, Q. Chen, Q. Du, R. Ge, R. Zhang, R. Pan, R. Wang, R. J. Chen, R. L. Jin, R. Chen, S. Lu, S. Zhou, S. Chen, S. Ye, S. Wang, S. Yu, S. Zhou, S. Pan, S. S. Li, S. Zhou, S. Wu, T. Yun, T. Pei, T. Sun, T. Wang, W. Zeng, W. Liu, W. Liang, W. Gao, W. Yu, W. Zhang, W. L. Xiao, W. An, X. Liu, X. Wang, X. Chen, X. Nie, X. Cheng, X. Liu, X. Xie, X. Liu, X. Yang, X. Li, X. Su, X. Lin, X. Q. Li, X. Jin, X. Shen, X. Chen, X. Sun, X. Wang, X. Song, X. Zhou, X. Wang, X. Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. Zhang, Y. Xu, Y. Li, Y. Zhao, Y. Sun, Y. Wang, Y. Yu, Y. Zhang, Y. Shi, Y. Xiong, Y. He, Y. Piao, Y. Wang, Y. Tan, Y. Ma, Y. Liu, Y. Guo, Y. Ou, Y. Wang, Y. Gong, Y. Zou, Y. He, Y. Xiong, Y. Luo, Y. You, Y. Liu, Y. Zhou, Y. X. Zhu, Y. Huang, Y. Li, Y. Zheng, Y. Zhu, Y. Ma, Y. Tang, Y. Zha, Y. Yan, Z. Z. Ren, Z. Ren, Z. Sha, Z. Fu, Z. Xu, Z. Xie, Z. Zhang, Z. Hao, Z. Ma, Z. Yan, Z. Wu, Z. Gu, Z. Zhu, Z. Liu, Z. Li, Z. Xie, Z. Song, Z. Pan, Z. Huang, Z. Xu, Z. Zhang, and Z. Zhang DeepSeek-r1 incentivizes reasoning in llms through reinforcement learning . Nature 645 ( 8081 ), pp. 633–638 . External Links: ISSN 1476-4687 , Link , Document Cited by: Appendix A , §2 . 

Han et al. (2025) D. Han, C. Couturier, D. M. Diaz, X. Zhang, V. Rühle, and S. Rajmohan Legomem: modular procedural memory for multi-agent llm systems for workflow automation . arXiv preprint arXiv:2510.04851 . Cited by: Appendix A , §2 . 

Hu et al. (2025) Y. Hu, S. Liu, Y. Yue, G. Zhang, B. Liu, F. Zhu, J. Lin, H. Guo, S. Dou, Z. Xi, et al. Memory in the age of ai agents . arXiv preprint arXiv:2512.13564 . Cited by: §1 . 

Jimenez Gutierrez et al. (2024) B. Jimenez Gutierrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su Hipporag: neurobiologically inspired long-term memory for large language models . Advances in Neural Information Processing Systems 37 , pp. 59532–59569 . Cited by: Appendix A , §2 . 

Kirkpatrick et al. (2017) J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks . Proceedings of the national academy of sciences 114 ( 13 ), pp. 3521–3526 . Cited by: §1 . 

Lewis et al. (2020) P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks . Advances in neural information processing systems 33 , pp. 9459–9474 . Cited by: §B.2 , §1 , §5.1 . 

Liu et al. (2024) N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang Lost in the middle: how language models use long contexts . Transactions of the association for computational linguistics 12 , pp. 157–173 . Cited by: §1 . 

Luo et al. (2025) Y. Luo, Z. Yang, F. Meng, Y. Li, J. Zhou, and Y. Zhang An empirical study of catastrophic forgetting in large language models during continual fine-tuning . IEEE Transactions on Audio, Speech and Language Processing . Cited by: Appendix A , §2 . 

Ouyang et al. (2022) L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. F. Christiano, J. Leike, and R. Lowe Training language models to follow instructions with human feedback . In Advances in Neural Information Processing Systems , S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.) , Vol. 35 , pp. 27730–27744 . Cited by: Appendix A , §1 , §2 . 

Park et al. (2023) J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein Generative agents: interactive simulacra of human behavior . In Proceedings of the 36th annual acm symposium on user interface software and technology , pp. 1–22 . Cited by: Appendix A , §1 , §2 . 

Press et al. (2023) O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, and M. Lewis Measuring and narrowing the compositionality gap in language models . In Findings of the Association for Computational Linguistics: EMNLP 2023 , pp. 5687–5711 . Cited by: §1 . 

Rafailov et al. (2023) R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn Direct preference optimization: your language model is secretly a reward model . In Proceedings of the 37th International Conference on Neural Information Processing Systems , NIPS ’23 , Red Hook, NY, USA . Cited by: Appendix A , §2 . 

Rajesh et al. (2025) S. Rajesh, P. Holur, C. Duan, D. Chong, and V. Roychowdhury Beyond fact retrieval: episodic memory for rag with generative semantic workspaces . arXiv preprint arXiv:2511.07587 . Cited by: §1 . 

Rezazadeh et al. (2024) A. Rezazadeh, Z. Li, W. Wei, and Y. Bao From isolated conversations to hierarchical schemas: dynamic tree memory representation for llms . arXiv preprint arXiv:2410.14052 . Cited by: Appendix A , §2 . 

Schulman et al. (2017) J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov Proximal policy optimization algorithms . arXiv preprint arXiv:1707.06347 . Cited by: §4 . 

Shi et al. (2025a) H. Shi, Z. Xu, H. Wang, W. Qin, W. Wang, Y. Wang, Z. Wang, S. Ebrahimi, and H. Wang Continual learning of large language models: a comprehensive survey . ACM Computing Surveys 58 ( 5 ), pp. 1–42 . Cited by: Appendix A , §2 . 

Shi et al. (2025b) Y. Shi, Y. Chen, S. Wang, S. Li, H. Cai, Q. Gu, X. Wang, and A. Zhang Look back to reason forward: revisitable memory for long-context llm agents . arXiv preprint arXiv:2509.23040 . Cited by: Appendix A , §2 . 

Shinn et al. (2023) N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao Reflexion: language agents with verbal reinforcement learning . Advances in Neural Information Processing Systems 36 , pp. 8634–8652 . Cited by: §1 . 

Shridhar et al. (2021) M. Shridhar, X. Yuan, M. Côté, Y. Bisk, A. Trischler, and M. Hausknecht ALFWorld: Aligning Text and Embodied Environments for Interactive Learning . In Proceedings of the International Conference on Learning Representations (ICLR) , External Links: Link Cited by: §B.1 , §5.1 . 

Squire (2004) L. R. Squire Memory systems of the brain: a brief history and current perspective . Neurobiology of learning and memory 82 ( 3 ), pp. 171–177 . Cited by: §1 , §3 . 

Sumers et al. (2023) T. Sumers, S. Yao, K. R. Narasimhan, and T. L. Griffiths Cognitive architectures for language agents . Transactions on Machine Learning Research . Cited by: Appendix A , §2 . 

Sutton et al. (1998) R. S. Sutton A. G. Barto et al. Reinforcement learning: an introduction . Vol. 1 , MIT press Cambridge . Cited by: §1 . 

Tan et al. (2024) W. Tan, W. Zhang, X. Xu, H. Xia, Z. Ding, B. Li, B. Zhou, J. Yue, J. Jiang, Y. Li, et al. Cradle: empowering foundation agents towards general computer control . arXiv preprint arXiv:2403.03186 . Cited by: Appendix A , §2 . 

Wang et al. (2023) G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar Voyager: an open-ended embodied agent with large language models . arXiv preprint arXiv:2305.16291 . Cited by: Appendix A , §2 . 

Wang et al. (2024) Z. Z. Wang, J. Mao, D. Fried, and G. Neubig Agent workflow memory . arXiv preprint arXiv:2409.07429 . Cited by: Appendix A , §B.2 , §1 , §2 , §5.1 . 

Wei et al. (2022) J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models . Advances in neural information processing systems 35 , pp. 24824–24837 . Cited by: §B.2 , §1 , §5.1 . 

Wu et al. (2025) R. Wu, X. Wang, J. Mei, P. Cai, D. Fu, C. Yang, L. Wen, X. Yang, Y. Shen, Y. Wang, et al. EvolveR: self-evolving llm agents through an experience-driven lifecycle . arXiv preprint arXiv:2510.16079 . Cited by: Appendix A , §2 . 

Xia et al. (2025) S. Xia, Z. Xu, J. Chai, W. Fan, Y. Song, X. Wang, G. Yin, W. Lin, H. Zhang, and J. Wang From experience to strategy: empowering llm agents with trainable graph memory . arXiv preprint arXiv:2511.07800 . Cited by: Appendix A , §1 , §2 . 

Xu et al. (2025) W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang A-mem: agentic memory for llm agents . arXiv preprint arXiv:2502.12110 . Cited by: §B.2 , §1 , §5.1 . 

Yang et al. (2025) C. Yang, X. Yang, L. Wen, D. Fu, J. Mei, R. Wu, P. Cai, Y. Shen, N. Deng, B. Shi, et al. Learning on the job: an experience-driven self-evolving agent for long-horizon tasks . arXiv preprint arXiv:2510.08002 . Cited by: Appendix A , §2 . 

Yao et al. (2023) S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan Tree of thoughts: deliberate problem solving with large language models . Advances in neural information processing systems 36 , pp. 11809–11822 . Cited by: §1 . 

Yao et al. (2022) S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. R. Narasimhan, and Y. Cao React: synergizing reasoning and acting in language models . In The eleventh international conference on learning representations , Cited by: §B.2 , §1 , §5.1 . 

Yuksekgonul et al. (2024) M. Yuksekgonul, F. Bianchi, J. Boen, S. Liu, Z. Huang, C. Guestrin, and J. Zou Textgrad: automatic” differentiation” via text . arXiv preprint arXiv:2406.07496 . Cited by: §4.1 . 

Zhang et al. (2025a) G. Zhang, M. Fu, G. Wan, M. Yu, K. Wang, and S. Yan G-memory: tracing hierarchical memory for multi-agent systems . arXiv preprint arXiv:2506.07398 . Cited by: Appendix A , §B.2 , §1 , §2 , §5.1 . 

Zhang et al. (2025b) G. Zhang, M. Fu, and S. Yan Memgen: weaving generative latent memory for self-evolving agents . arXiv preprint arXiv:2509.24704 . Cited by: Appendix A , §2 . 

Zhang et al. (2026) S. Zhang, J. Wang, R. Zhou, J. Liao, Y. Feng, W. Zhang, Y. Wen, Z. Li, F. Xiong, Y. Qi, B. Tang, and M. Wen MemRL: self-evolving agents via runtime reinforcement learning on episodic memory . External Links: 2601.03192 , Link Cited by: §3.3 . 

Zhao et al. (2024) A. Zhao, D. Huang, Q. Xu, M. Lin, Y. Liu, and G. Huang Expel: llm agents are experiential learners . In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 38 , pp. 19632–19642 . Cited by: Appendix A , Appendix A , §B.2 , §1 , §2 , §5.1 . 

Zhou et al. (2025a) H. Zhou, Y. Chen, S. Guo, X. Yan, K. H. Lee, Z. Wang, K. Y. Lee, G. Zhang, K. Shao, L. Yang, et al. Memento: fine-tuning llm agents without fine-tuning llms . arXiv preprint arXiv:2508.16153 . Cited by: Appendix A , §3.1 , §3.3 . 

Zhou et al. (2025b) Z. Zhou, A. Qu, Z. Wu, S. Kim, A. Prakash, D. Rus, J. Zhao, B. K. H. Low, and P. P. Liang MEM1: learning to synergize memory and reasoning for efficient long-horizon agents . arXiv preprint arXiv:2506.15841 . Cited by: Appendix A , §2 . 

Zhu et al. (2023) X. Zhu, Y. Chen, H. Tian, C. Tao, W. Su, C. Yang, G. Huang, B. Li, L. Lu, X. Wang, et al. Ghost in the minecraft: generally capable agents for open-world environments via large language models with text-based knowledge and memory . arXiv preprint arXiv:2305.17144 . Cited by: Appendix A , §2 . 

Ziegler et al. (2020) D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving Fine-tuning language models from human preferences . External Links: 1909.08593 , Link Cited by: Appendix A , §2 . 

Summary of Appendices 

• 
Section A: Full Related Works 

• 
Section B: Detailed Experimental Setup 

• 
Section C: Additional Experimental Details 

• 
Section D: Case Study: Semantic Gradient Generation 

• 
Section E: Prompt Templates 

• 
Section F: Discussion: Implicit Procedural Memory 

Algorithm 1 Non-Parametric PPO for Skill Evolution Input: Initial Skill pool Ω 0 \Omega_{0} , Frozen LLM π LLM \pi_{\text{LLM}} , Capacity K K Initialize online scores Score ​ ( ω ) = 0 \text{Score}(\omega)=0 for all ω ∈ Ω 0 \omega\in\Omega_{0} for n = 1 n=1 to N N do // 1. Experience Collection Collect a batch of trajectories 𝒯 ( B ) \mathcal{T}^{(B)} using policy π Ω = μ ⋅ π LLM \pi_{\Omega}=\mu\cdot\pi_{\text{LLM}} . // 2. Semantic Gradient Extraction & Optimization for each Skill ω ∈ Ω n \omega\in\Omega_{n} inv
