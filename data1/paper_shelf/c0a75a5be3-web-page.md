# ToolGen: Unified Tool Retrieval and Calling via Generation

Source URL: https://arxiv.org/html/2410.03439v3

ToolGen: Unified Tool Retrieval and Calling via Generation 
Renxi Wang Affiliation: LibrAI Affiliation: Mohamed bin Zayed University of Artificial Intelligence Email: renxi.wang@mbzuai.ac.ae Xudong Han Affiliation: LibrAI Affiliation: Mohamed bin Zayed University of Artificial Intelligence Email: xudong.han@mbzuai.ac.ae Lei Ji Affiliation: Microsoft Email: timothy.baldwin@mbzuai.ac.ae Shu Wang Affiliation: University of California, Los Angeles Email: haonan.li@mbzuai.ac.ae Timothy Baldwin Haonan Li Affiliation: LibrAI Affiliation: Mohamed bin Zayed University of Artificial Intelligence Affiliation: The University of Melbourne Email: leiji@microsoft.com Email: shuwang0712@ucla.edu Abstract 
As large language models (LLMs) advance, their inability to autonomously execute tasks by directly interacting with external tools remains a critical limitation. Traditional methods rely on inputting tool descriptions as context, which is constrained by context length and requires separate, often inefficient, retrieval mechanisms. We introduce ToolGen, a paradigm shift that integrates tool knowledge directly into the LLM’s parameters by representing each tool as a unique token. This enables the LLM to generate tool calls and arguments as part of its next token prediction capabilities, seamlessly blending tool invocation with language generation. Our framework allows the LLM to access and utilize a vast amount of tools with no additional retrieval step, significantly enhancing both performance and scalability. Experimental results with over 47,000 tools show that ToolGen not only achieves superior results in both tool retrieval and autonomous task completion but also sets the stage for a new era of AI agents that can adapt to tools across diverse domains. By fundamentally transforming tool retrieval into a generative process, ToolGen paves the way for more versatile, efficient, and autonomous AI systems. ToolGen enables end-to-end tool learning and opens opportunities for integration with other advanced techniques such as chain-of-thought and reinforcement learning, thereby expanding the practical capabilities of LLMs 1 1 1 Data and code are available at https://github.com/Reason-Wang/ToolGen . 

1 Introduction 

Large language models (LLMs) have demonstrated impressive capabilities as interactive systems, adept at processing external inputs, executing actions, and autonomously completing tasks ( Gravitas, 2023 ; Qin et al., 2023 ; Yao et al., 2023 ; Shinn et al., 2023 ; Wu et al., 2024a ; Liu et al., 2024 ; Wang et al., 2024b ; Wang et al., 2024c ) . Among the various methods enabling LLMs to interact with the world, tool calling via APIs has emerged as one of the most common and effective approaches. However, as the number of tools grows into the tens of thousands, existing methods for tool retrieval and execution struggle to scale efficiently. 

A common approach in real-world scenarios is to combine tool retrieval with tool execution, where a retrieval model first narrows down the relevant tools before passing them to the LLM for final selection and execution ( Qin et al., 2023 ; Patil et al., 2023 ) . While this combined method addresses the challenge of handling vast numbers of tools, it has notable limitations: retrieval models often rely on small encoders that fail to fully capture the semantics of complex tools and queries, and separating retrieval from execution introduces inefficiencies and potential misalignment between stages of task completion. 

Moreover, LLMs and their tokenizers are pretrained primarily on natural language data ( Brown et al., 2020 ; Touvron et al., 2023 ) , leaving them with limited intrinsic knowledge of tool-related functionalities. This gap in knowledge results in suboptimal performance, especially when the LLM must rely on retrieved tool descriptions for decision-making. 

In this study, we introduce ToolGen , a novel framework that integrates real-world tool knowledge directly into the LLM’s parameters and transforms tool retrieval and execution into a unified generation task. Specifically, ToolGen expands the LLM’s vocabulary with tool-specific virtual tokens and trains the model to generate these tokens within a conversational context, allowing the LLM to leverage its pre-existing knowledge more effectively for both retrieving and calling tools. 

Specifically, each tool is represented as a unique virtual token within the LLM’s vocabulary. Building upon a pretrained LLM, ToolGen’s training process consists of three stages: tool memorization, retrieval training, and agent training. In the tool memorization stage, the model associates each virtual tool token with its documentation. During retrieval training, the model learns to generate relevant tool tokens based on user queries. Finally, in end-to-end agent-tuning, the model is trained to act as an autonomous agent, generating plans and tools, and determining the appropriate parameters to complete tasks. By calling tools and receiving feedback from external environments, the model can handle user queries efficiently and integratively. Figure 1 shows comparison between ToolGen and traditional paradigms. 

We demonstrate ToolGen’s superiority in two scenarios: a tool retrieval task, where the model retrieves the correct tool for a given query, and an LLM-based agent task, where the model completes complex tasks involving real-world API calls. Leveraging a dataset of 47,000 real-world tools, ToolGen achieves performance comparable to the leading tool retrieval methods, but with significantly lower cost and greater efficiency. Additionally, it surpasses traditional tool learning paradigms, highlighting its potential for advancing more effective tool usage systems. 
Figure 1: Comparison between previous retrieval-based methods and our ToolGen. Previous methods use a retriever to retrieve relevant tools based on similarity matching, which are further put into prompts for LLMs to select. ToolGen can retrieve tools by generating tool tokens directly. ToolGen can also complete the task without relying on any external retriever. 
ToolGen represents a paradigm shift in tool interaction by merging retrieval and generation into a single, cohesive model. This innovation sets the stage for a new generation of AI agents capable of adapting to a vast array of tools across diverse domains. Additionally, ToolGen opens new opportunities for integrating advanced techniques like chain-of-thought reasoning and reinforcement learning with the ability to use tools in a unified generation way, expanding the capabilities of LLMs in real-world applications. 

In summary, our contributions are: 

• 
A novel framework, ToolGen, that integrates tool retrieval and execution into the LLM’s generative process using virtual tokens. 

• 
A three-stage training process that enables efficient and scalable tool retrieval and API calling within ToolGen. 

• 
Experimental validation demonstrates that ToolGen achieves comparable performance to current best tool retrieval methods with significantly less cost and higher efficiency and surpasses traditional tool learning paradigms across large-scale tool repositories. 

2 Related Work 

2.1 Tool Retrieval 

Tool retrieval is essential for LLM agents in real-world task execution, where tools are usually represented by their documentation. Traditional methods like sparse (e.g., BM25 ( Robertson et al., 2009 ) ) and dense retrieval (e.g., DPR ( Karpukhin et al., 2020 ) , ANCE ( Xiong et al., 2021 ) ) rely on large document indices and external modules, leading to inefficiencies and difficulty in optimizing in an end-to-end agent framework. Some work has explored alternative methods. For example, Chen et al. (2024b) rewrite queries and extract their intent, targeting unsupervised retrieval settings, though the results are not comparable to supervised approaches. Xu et al. (2024) propose a method that iteratively refines queries based on tool feedback, improving retrieval accuracy but increasing latency. 

Recently, generative retrieval has emerged as a promising new paradigm, wherein models directly generate relevant document identifiers rather than relying on traditional retrieval mechanisms ( Wang et al., 2022 ; Sun et al., 2023b ; Kishore et al., 2023b ; Mehta et al., 2023b ; Chen et al., 2023c ) . Motivated by this, ToolGen represents each tool as a unique token, allowing tool retrieval and calling to be framed as a generation task. Beyond simplifying retrieval, this design integrates smoothly with other LLM and LLM-based agent features like chain-of-thought reasoning ( Wei et al., 2023 ) and ReAct ( Yao et al., 2023 ) . By consolidating retrieval and task execution into a single LLM agent, it reduces latency and computational overhead, leading to more efficient and effective task completion. 

2.2 LLM-Agents with Tool Calling 

LLMs have shown strong potential in mastering tools for various tasks. However, most existing works focus on a limited set of actions ( Chen et al., 2023a ; Zeng et al., 2023 ; Yin et al., 2024 ; Wang et al., 2024a ) . For instance, Toolformer ( Schick et al., 2023 ) fine-tunes GPT-J to handle just five tools, such as calculators. While effective for narrow tasks, this approach struggles in real-world scenarios with vast action spaces. ToolBench ( Qin et al., 2023 ) expands the scope by introducing over 16,000 tools, highlighting the challenge of tool selection in complex environments. 

To perform tool selection, current methods often use a retriever-generator pipeline, where relevant tools are retrieved and then utilized by the LLM ( Patil et al., 2023 ; Qin et al., 2023 ) . In addition, TPTU ( Ruan et al., ) proposes a structured framework for LLM agents and evaluates their task planning and tool usage abilities. Furthermore, TPTU-v2 ( Kong et al., ; Kong et al., 2024 ) builds an LLM Finetuner to enhance agent performance with curated datasets and a demo selector to select relevant demonstrations. They set a flexible and superior paradigm compared to traditional retrieval-based paradigm. However, pipelined approaches face two major issues: error propagation from the retrieval step and the inability of LLMs to fully understand and use tools via simple prompting. 

To mitigate these issues, researchers have tried representing actions as tokens, converting action prediction into a generative task. For example, RT2 ( Brohan et al., 2023 ) generates tokens representing robot actions, and Self-RAG ( Asai et al., 2023 ) uses special tokens to decide when to retrieve documents. ToolkenGPT ( Hao et al., 2023 ) introduces tool-specific tokens to trigger tool usage, a concept closest to our approach. 

Our approach differs from ToolkenGPT in several ways. First, we focus on real-world tools that require flexible parameters for complex tasks (e.g., YouTube channel search), while ToolkenGPT is limited to simpler tools with fewer inputs (e.g., math functions with two numbers). Additionally, ToolkenGPT relies on few-shot prompting, whereas ToolGen incorporates tool knowledge directly into the LLM through full-parameter fine-tuning, enabling the model to retrieve and execute tasks autonomously. Finally, our experiments involve a much larger tool set—47,000 tools compared to ToolkenGPT’s 13–300. Detailed comparison and other related work can be found in Section A . 

3 ToolGen 

In this section, we first introduce the notations used throughout the paper. Then we detail the specific methods of ToolGen, including tool virtualization, tool memorization, retrieval training, and end-to-end agent tuning, as illustrated in Figure 2 . Lastly, we describe our inference approach. 
Figure 2: An illustration of ToolGen framework. In tool virtualization, tools are mapped into virtual tokens. In the following three-stage training, ToolGen first memorizes tools by predicting tool tokens based on their documentations. Then it learns to retrieve tools by predicting tool tokens from queries. Finally, pipeline data, i.e., trajectories, are used to finetune the retriever model from the last stage, resulting in the ToolGen Agent model. 
3.1 Preliminaries 

Given a user query q q , tool learning aims to resolve q q using tools from a large tool set D = { d 1 , d 2 , … , d N } D=\{d_{1},d_{2},\ldots,d_{N}\} , where | D | = N |D|=N is a large number, making it impractical to include all tools in D D in the LLM context. Therefore, current research typically uses a retriever R R to retrieve k k relevant tools from D D , denoted as D k , R = { d r 1 , d r 2 , … , d r k } = R ⁡ ( q , k , D ) D_{k,R}=\{d_{r_{1}},d_{r_{2}},\ldots,d_{r_{k}}\}=R(q,k,D) , where | D k , R | ≪ N |D_{k,R}|\ll N . The final prompt is then the concatenation of q q and D k , R D_{k,R} , denoted as P ​ r ​ o ​ m ​ p ​ t = [ q , D k , R ] Prompt=[q,D_{k,R}] . To complete a task (query), an LLM-based agent usually adopts a four-stage paradigm ( Qu et al., 2024 ) iteratively: generates a plan p i p_{i} , selects a tool d s ​ i d_{si} , determines tool parameters c i c_{i} , and collects feedback from the tool(s) f i f_{i} . We denote these steps for the i i -th iteration as p i , d s i , c i , f i p_{i},d_{s_{i}},c_{i},f_{i} . The model continues iterating through these steps until the task is completed, at which point the final answer a a is generated. The entire trajectory can be represented as T ​ r ​ a ​ j = [ P ​ r ​ o ​ m ​ p ​ t , ( p 1 , d s 1 , c 1 , f 1 ) , … , ( p t , d s t , c t , f t ) , a ] = [ q , R ⁡ ( q , D ) , ( p 1 , d s 1 , c 1 , f 1 ) , … , ( p t , d s t , c t , f t ) , a ] Traj=[Prompt,(p_{1},d_{s_{1}},c_{1},f_{1}),\ldots,(p_{t},d_{s_{t}},c_{t},f_{t}),a]=[q,R(q,D),(p_{1},d_{s_{1}},c_{1},f_{1}),\ldots,(p_{t},d_{s_{t}},c_{t},f_{t}),a] . This iterative approach allows the model to dynamically adjust and refine its actions at each step based on the feedback received, improving its performance in completing complex tasks. 

3.2 Tool Virtualization 

In ToolGen, we virtualize tools by mapping each tool to a unique new token through a method we call atomic indexing. In this approach, each tool is assigned a unique token by expanding the LLM’s vocabulary. The embedding for each tool token is initialized as the average embedding of its corresponding tool name, ensuring a semantically meaningful starting point for each tool. 

Formally, the token set is defined as T = Index ⁡ ( d ) | ∀ d ∈ D T={\mathrm{Index}(\mathrm{d})\ |\ \forall d\in D} , where Index \mathrm{Index} is the function mapping tools to tokens. We demonstrate that atomic indexing is more efficient and can mitigate hallucination compared to other indexing methods, such as semantic and numeric mappings, discussed in Section 4.3 and 5.4 . 

3.3 Tool Memorization 

After assigning tokens to tools, the LLM still lacks any knowledge of the tools. To address this, we inject tool information by fine-tuning it with tool descriptions as inputs and their corresponding tokens as outputs, which we call tool memorization. We use the following loss function: 
ℒ t ​ o ​ o ​ l = ∑ d ∈ D − log p θ ( Index ( d ) | d d ​ o ​ c ) \mathcal{L}_{tool}=\sum_{d\in D}-\log p_{\theta}(\mathrm{Index}(d)|d_{doc}) 
where θ \theta denotes the LLM parameters, and d d ​ o ​ c d_{doc} represents the tool description. This step equips the LLM with basic knowledge of the tools and their associated actions. 

3.4 Retrieval Training 

We then train the LLMs to link the hidden space of virtual tool token (and its documentation), to the user query space, so that LLM can generate correct tool based on a user’s query. To achieve this, we fine-tune the LLM with user queries as inputs and corresponding tool tokens as outputs: 
ℒ r ​ e ​ t ​ r ​ i ​ e ​ v ​ a ​ l = ∑ q ∈ Q ∑ d ∈ D q − log p θ ′ ( Index ( d ) | q ) \mathcal{L}_{retrieval}=\sum_{q\in Q}\sum_{d\in D_{q}}-\log p_{\theta^{\prime}}(\mathrm{Index}(d)|q) 
where θ ′ \theta^{\prime} represents the LLM parameters after tool memorization, Q Q is the set of user queries, and D q D_{q} is the set of tools relevant to each query. This results in the ToolGen Retriever, which can generate the appropriate tool token given a user query. 

3.5 End-to-End Agent-Tuning 

After retrieval training, the LLM is capable of generating tool tokens from queries. In the final stage, we fine-tune the model with agent task completion trajectories. We adopt a similar inference strategy as Agent-Flan ( Chen et al., 2024c ) , in instead of generating Thought, Action, and Arguments together as ReAct . Our pipeline follows an iterative process, where the LLM first generates a Thought, and the corresponding Action token. This token is used to fetch the tool documentation, which the LLM uses to generate the necessary arguments. The process continues iteratively until the model generates a “finish” token or the maximum number of turns is reached. The generated trajectory is represented as T ​ r ​ a ​ j = [ q , ( p 1 , Index ⁡ ( d s 1 ) , c 1 , f 1 ) , … , ( p t , Index ⁡ ( d s t ) , c t , f t ) , a ] Traj=[q,(p_{1},\mathrm{Index}(d_{s_{1}}),c_{1},f_{1}),\ldots,(p_{t},\mathrm{Index}(d_{s_{t}}),c_{t},f_{t}),a] . In this structure, relevant tools are no longer required. 

3.6 Inference 

During inference, the LLM may generate action tokens outside the predefined tool token set. To prevent this, we designed a constrained beam search generation that restricts the output tokens to the tool token set. We applied this constrained beam search for both tool retrieval, where the model selects tools based on queries, and the end-to-end agent system, significantly reducing hallucination during the action generation step. A detailed analysis can be found in Section 5.4 . The implementation details can be found in Appendix E . 

4 Tool Retrieval Evaluation 

4.1 Experimental Setup 

We use pretrained Llama-3-8B ( Dubey et al., 2024 ) as our foundation model, with a vocabulary size of 128,256. Using the atomic indexing approach, we expand the vocabulary by an additional 46,985 tokens following the tool virtualization process, resulting in a final vocabulary size of 175,241. We fine-tune the model using the Llama-3 chat template with a cosine learning rate scheduler, applying a 3% warm-up steps. The maximum learning is 4 × 10 − 5 4\times 10^{-5} . All models are trained using Deepspeed ZeRO 3 ( Rajbhandari et al., 2020 ) across 4 × \times A100 GPUs. We train 8 epochs for tool memorization and 1 epoch for retrieval training. 

Dataset 

Our experiments are based on ToolBench, a real-world tool benchmark containing more 16k tool collections, each containing several APIs, resulting in a total of 47k unique APIs. Each API is documented with a dictionary, containing the name, description, and parameters for calling the API. A real example is shown in Appendix C . We take each API as an action and map it to a token. Our retrieval and end-to-end agent-tuning data are converted from the original data in ToolBench. Details can be found in Appendix K . Although each tool may consist of multiple APIs, for simplicity, we refer to each API as a tool in this paper. 

We follow the data split of Qin et al. (2023) , where 200k (query, relevant API) pairs are divided into three categories: I1 (single-tool queries), I2 (intra-category multi-tool queries), and I3 (intra-collection multi-tool instructions), containing 87,413, 84,815, and 25,251 instances, respectively. 

Baselines 

We compare ToolGen with the following baselines: 

• 
BM25: A classical unsupervised retrieval method based on TF-IDF, which retrieves documents based on term similarity with the query. 

• 
Long-Context LLMs: We concatenate tools into a long prompt to gpt-4o , and prompt it to choose from the pool. Limit by context length, we cannot input all 47k tools, so we use 2k tools with ground truth tools included. 

• 
Embedding Similarity (EmbSim): Sentence embeddings generated using OpenAI’s sentence embedding model; specifically text-embedding-3-large in our experiences. 

• 
Re-Invoke ( Chen et al., 2024b ) : An unsupervised retrieval method with query rewriting and document expansion. 

• 
IterFeedback ( Xu et al., 2024 ) : BERT-based retriever with gpt-3.5-turbo-0125 as a feedback model with iterative feedback for up to 10 rounds. 

• 
ToolRetriever ( Qin et al., 2023 ) : A BERT-based retriever trained via contrastive learning. 

Settings 

We conduct experiments under two settings. In the first, In-Domain Retrieval , the retrieval search space is restricted to tools within the same domain. For example, when evaluating queries from domain I1, the search is limited to I1 tools. This aligns with ToolBench settings. The second, Multi-Domain Retrieval , is more complex, with the search space expanded to include tools from all three domains. In this case, models are trained on combined data, increasing both the search space and task complexity. Unlike ToolBench, this multi-domain setting reflects real-world scenarios where retrieval tasks may involve overlapping or mixed domains. This setup evaluates the model’s ability to generalize across domains and handle more diverse, complex retrieval cases. 

Metrics 

We evaluate retrieval performance using Normalized Discounted Cumulative Gain (NDCG) ( Järvelin & Kekäläinen, 2002 ) , a widely used metric in ranking tasks, including tool retrieval. NDCG accounts for both the relevance and ranking position of retrieved tools. 
Table 1: Tool retrieval evaluation across two settings: (1) In-Domain , where models are trained and evaluated within the same domain; and (2) Multi-Domain , where models are trained on all domains and evaluated with the full set of tools across all domains. BM25, EmbSim, and Re-Invoke are unsupervised baselines without training. IterFeedback is retrieval system with multiple models and feedback mechanism. ToolRetriever is trained using contrastive learning, while ToolGen is trained with next-token prediction. Results marked with * were not implemented by us and are copied from their original paper, and hence only in the In-Domain setting. For ToolGen in the In-Domain setting, we allow the generation space to include all tokens, which is a more challenging scenario compared to other models. Best results in each category are bolded . Model I1 I2 I3 NDCG1 NDCG3 NDCG5 NDCG1 NDCG3 NDCG5 NDCG1 NDCG3 NDCG5 In-Domain BM25 29.46 31.12 33.27 24.13 25.29 27.65 32.00 25.88 29.78 Long-Context LLM* 32.22 42.87 52.14 25.39 33.91 46.07 25.11 32.57 44.03 EmbSim 63.67 61.03 65.37 49.11 42.27 46.56 53.00 46.40 52.73 Re-Invoke* 69.47 – 61.10 54.56 – 53.79 59.65 – 59.55 IterFeedback* 90.70 90.95 92.47 89.01 85.46 87.10 91.74 87.94 90.20 ToolRetriever 80.50 79.55 84.39 71.18 64.81 70.35 70.00 60.44 64.70 ToolGen 89.17 90.85 92.67 91.45 88.79 91.13 87.00 85.59 90.16 Multi-Domain BM25 22.77 22.64 25.61 18.29 20.74 22.18 10.00 10.08 12.33 EmbSim 54.00 50.82 55.86 40.84 36.67 39.55 18.00 17.77 20.70 ToolRetriever 72.31 70.30 74.99 64.54 57.91 63.61 52.00 39.89 42.92 ToolGen 87.67 88.84 91.54 83.46 86.24 88.84 79.00 79.80 84.79 
4.2 Results 

Table 1 presents the tool retrieval results. As expected, all trained models significantly outperform the untrained baselines (BM25, EmbSim, and Re-Invoke) across all metrics, demonstrating the benefit of training on tool retrieval data. 

Our proposed ToolGen model consistently achieves the best performance across both settings. In the In-Domain setting, ToolGen delivers highly competitive results, achieving comparable performance to the IterFeedback system, which uses multiple models and a feedback mechanism. ToolGen, as a single model, outperforms ToolRetriever by a significant margin in all metrics and even surpasses IterFeedback in several cases, such as NDCG@5 for domain I1 and NDCG@1,@3,@5 for I2. 

In the Multi-Domain setting, where the search space is larger and performance generally drops, ToolGen remains robust, outperforming ToolRetriever and maintaining superiority over other baselines. This demonstrates that ToolGen, despite being a single model, is capable of competing with complex retrieval systems like IterFeedback, showcasing its ability to handle complex real-world retrieval tasks where domain boundaries are less defined. 

4.3 Indexing Method Comparison 

While ToolGen uses atomic indexing for tool virtualization, we explore several alternative generative retrieval approaches. In this section, we compare it with the following three methods: 

• 
Numeric : Map each tool to a unique number. The resulting token is purely numeric, offering no inherent semantic information, but providing a distinct identifier for each tool. 

• 
Hierarchical : This method clusters tools into non-overlapping groups and recursively partitions these clusters, forming a hierarchical structure. The index from the root to the leaf in this tree-like structure represents each tool, similarly to Brown clustering techniques. 

• 
Semantic : In this approach, each tool is mapped to its name, using the semantic content of the tool names to guide the LLM. The tool’s name provides a meaningful representation directly related to its function. 

Figure 3: The distribution of the number of subtokens per tool varies across different indexing methods. 
The implementation details are described in Appendix D . 

First, we conducted an analysis of the number of subtokens required to represent each tool for the different methods, as shown in Figure 3 . Atomic indexing ensures each tool to be a single token, while numeric indexing encodes tools into N tokens for tools numbered in ( 10 N − 1 , 10 N ] (10^{N-1},10^{N}] . In contrast, both semantic indexing and hierarchical indexing produce a variable number of subtokens, with semantic indexing having more outliers with significantly longer sequences. The figure highlights the superiority of atomic indexing, where each tool is represented by a single token, whereas other methods require multiple tokens. This efficiency allows ToolGen to reduce the number of generation tokens and inference time in both the retrieval and agent scenarios. 

Next, we examined the effectiveness of different indexing methods. As shown in Table 2 , semantic indexing demonstrates the best retrieval performance across various metrics and scenarios, while atomic indexing closely follows in many cases. We attribute this to the fact that semantic indexing aligns better with the pretraining data of LLMs. However, this advantage diminishes as the training data and type increase. For example, in Section 5.3 , we show that atomic indexing achieves better end-to-end results. We also show that combining constrained beam search with semantic indexing will cause biased tool usage, which is detailed in Section E.2 . 

Taking all these factors into account, we choose atomic indexing for ToolGen tool virtualization. 
Table 2: Retrieval evaluation for different indexing methods in Multi-Domain setting. Best results are bolded and second best results are underlined . Model I1 I2 I3 NDCG1 NDCG3 NDCG5 NDCG1 NDCG3 NDCG5 NDCG1 NDCG3 NDCG5 Numetric 83.17 84.99 88.73 79.20 79.23 83.88 71.00 74.81 82.95 Hierarchical 85.67 87.38 90.26 82.22 82.70 86.63 78.50 79.47 84.15 Semantic 89.17 91.29 93.29 83.71 84.51 88.22 82.00 78.86 85.43 Atomic 87.67 88.84 91.54 83.46 86.24 88.84 79.00 79.80 84.79 Table 3: Ablation study for tool retrieval. We assess the impact of removing retrieval training, tool memorization, and constrained beam search on ToolGen’s performance, respectively. Model I1 I2 I3 NDCG1 NDCG3 NDCG5 NDCG1 NDCG3 NDCG5 NDCG1 NDCG3 NDCG5 ToolGen 87.67 88.84 91.54 83.46 86.24 88.84 79.00 79.80 84.79 − - memorization 84.00 86.77 89.35 82.21 83.20 86.78 77.00 77.71 84.37 − - retrieval training 10.17 12.31 13.89 0 5.52 0 7.01 0 7.81 0 3.00 0 4.00 0 4.43 − - constraining 87.67 88.79 91.45 83.46 86.24 88.83 79.00 79.93 84.92 
4.4 Ablation 

We perform an ablation study to assess the impact of different training stages of ToolGen, as shown in Table 3 . The results indicate that retrieval training is the crucial factor for tool retrieval performance, as it directly aligns with the retrieval task where inputs are queries and outputs are tool tokens. Removing tool memorization leads to a minor performance drop although it plays a role in improving generalization, which we will discuss further in Appendix J . Similarly, constrained beam search, while not a major contributor to retrieval task, helps prevent hallucinations, making it useful for end-to-end agent tasks, see Section 5.4 . 

5 End-to-End Evaluation 

5.1 Experimental Setup 

We make several modifications to the trajectory data from ToolBench to fit it into ToolGen framework. For example, as ToolGen does not require explicit selection of related tools as input, we remove this information in the system prompt. Further details are provided in Appendix K . Following this, we fine-tune the retrieval model using the reformatted data, resulting in an end-to-end ToolGen agent. 

Baselines 

GPT-3.5 : We use gpt-3.5-turbo-0613 as one of our baselines. The implementation is the same as used in StableToolBench ( Guo et al., 2024 ) , where the tool calling capability of GPT-3.5 is used to form a tool agent. ToolLlama-2 : Qin et al. (2023) introduced ToolLlama-2 by fine-tuning Llama-2 ( Touvron et al., 2023 ) model on ToolBench data. ToolLlama-3 : To ensure a fair comparison, we fine-tuned Llama-3, the same base model used in ToolGen, on the ToolBench dataset, creating the ToolLlama-3 baseline. In the rest of this paper, we refer to ToolLlama-3 as ToolLlama to distinguish it from ToolLlama-2. 

Settings 

w/ Ground Truth Tools (G.T.) Following Qin et al. (2023) , we define ground truth tools for a query as those selected by ChatGPT. For ToolLlama, we directly input the ground truth tools in the prompt, consistent with its training data format. For ToolGen, which is not trained on data with pre-selected tools, we add a prefix during the planning phase: I am using the following tools: [tool tokens] , where [tool tokens] are virtual tokens corresponding to the ground-truth tools. w/ Retriever In the end-to-end experiments, we use a retrieval-based setting. For baselines, we use the tools retrieved by ToolRetriever as the relevant tools. In contrast, ToolGen generates tool tokens directly, so no retriever is used. 

All models are finetuned using a cosine scheduler with maximum learning rate set to 4 × 10 − 5 4\times 10^{-5} . Context length is truncated to 6,144. The total batch size is set to 512. We further use Flash-Attention ( Dao et al., 2022 ; Dao, 2024 ) and Deepspeed ZeRO 3 ( Rajbhandari et al., 2020 ) to save memory. 

ToolGen and ToolLlama follow different paradigms to complete tasks. ToolLlama generates Thought, Action, and Parameters in a single round, while ToolGen separates these steps. For ToolGen, we set a maximum of 16 turns, which allows for 5 rounds of actions and 1 final round for providing the answer. We compare this to ToolLlama, which operates with a 6-turn limit. 

Additionally, we introduce a retry mechanism for all models to prevent early termination, the details are introduced in Section G . Specifically, if a model generates a response containing give up or I’m sorry , we prompt the model to regenerate the response with a higher temperature. 

Metrics 

For end-to-end evaluation, we use StableToolBench ( Guo et al., 2024 ) , a stabilized tool evaluation benchmark that selects solvable queries from ToolBench and uses GPT-4 ( OpenAI, 2024 ) to simulate outputs for failed tools. We employ two metrics to assess performance: Solvable Pass Rate (SoPR) , which is the percentage of queries successfully solved, and Solvable Win Rate (SoWR) , which indicates the percentage of answers outperforming those generated by a reference model (GPT-3.5 in this study). Additionally, we provide micro-average scores for each category. 
Table 4: End-to-end evaluation performance on unseen instructions under two settings. For R. setting, GPT3.5 and ToolLlama use ToolRetriever, while ToolGen does not use external retriever. For all results, SoPR and SoWR are evaluated three time and reported with mean values. Model SoPR SoWR I1 I2 I3 Avg. I1 I2 I3 Avg w/ Ground Truth Tools (G.T.) GPT-3.5 56.60 47.80 54.64 50.91 - - - - ToolLlama-2 53.37 41.98 46.45 48.43 47.27 59.43 27.87 47.58 ToolLlama 55.93 48.27 52.19 52.78 50.31 53.77 31.15 47.88 ToolGen 61.35 49.53 43.17 54.19 51.53 57.55 31.15 49.70 w/ Retriever (R.) GPT-3.5 51.43 41.19 34.43 45.00 53.37 53.77 37.70 50.60 ToolLlama-2 56.13 49.21 34.70 49.95 50.92 53.77 21.31 46.36 ToolLlama 54.60 49.96 51.37 51.55 49.08 61.32 31.15 49.70 ToolGen 56.13 52.20 47.54 53.28 50.92 62.26 34.42 51.51 
5.2 Results 

Table 4 presents the end-to-end evaluation performance of various models in two settings: using Ground Truth Tools (G.T.) and a Retriever (R.). In the G.T. setting, ToolGen achieves the best average SoPR score of 54.19, outperforming GPT-3.5 and ToolLlama, with SoWR also highest for ToolGen at 49.70. In the Retriever setting, ToolGen maintains its lead with an average SoPR of 53.28 and SoWR of 51.51. ToolLlama shows competitive performance, surpassing ToolGen on some individual instances. An ablation study of end-to-end ToolGen is provided in Appendix K . 
Table 5: End-to-end evaluation for different indexing methods. Indexing SoPR SoWR I1 I2 I3 Avg. I1 I2 I3 Avg Numeric 34.76 29.87 46.99 35.45 25.77 33.02 29.51 28.79 Hierarchical 50.20 45.60 32.79 45.50 38.04 43.40 29.51 38.18 Semantic 58.79 45.28 44.81 51.87 49.69 57.55 26.23 47.88 Atomic 58.08 56.13 44.81 55.00 47.85 57.55 29.51 47.58 
5.3 Indexing Method Comparison 

Similar to indexing method comparison for retrieval task (Section 4.3 ), Table 5 presents a comparison of different indexing methods for the end-to-end agent task. In this setting, constrained decoding is removed, allowing the agent to freely generate Thought, Action, and Parameters. From the results, we observe that the Atomic method achieves the best performance among the four indexing methods. We attribute this to the higher hallucination rates in the other methods, as discussed in Section 5.4 . 
Figure 4: The hallucination rates of generating nonexistent tools across different models are shown. ToolGen does not generate any nonexistent tools when using constrained decoding. However, without this constraint, ToolGen generates 7% non-tool tokens during the Action generation stage with atomic indexing, and even more with semantic indexing. For ToolLlama and GPT-3.5, despite being provided with five ground truth tools in the prompt, hallucinations still occur. Without any tools specified in the prompt, ToolLlama generates over 50% nonexistent tool names. 
5.4 Hallucination 

We evaluate model hallucination in tool generation within an end-to-end agent scenario. To do this, we input a query in the format the models were trained on. Specifically, for ToolGen, we input a query directly and prompt the model to respond using the ToolGen agent paradigm (i.e., sequentially generating Thought, Tool, and Parameters). We tested Actions decoding without the beam search constraints described in Section 3.6 . For ToolLlama and GPT-3.5, we input the query along with 5 ground truth tools. In all settings, we report the proportion of generated tools that do not exist in the dataset out of all tool generation actions. Figure 4 shows the hallucination rates of nonexistent tools for different models. From the figure, we observe that, despite being provided with only five ground truth tools, ToolLlama and GPT-3.5 may still generate nonexistent tool names. In contrast, ToolGen, with constrained decoding, does not hallucinate at all due to its design. 

6 Conclusions 

In this paper, we introduced ToolGen, a framework that unifies tool retrieval and execution in large language models (LLMs) by embedding tool-specific virtual tokens into the model’s vocabulary, transforming tool interaction into a generative task. By incorporating a three-stage training process, ToolGen equips LLMs with the ability to efficiently retrieve and execute tools in real-world scenarios. This unified approach sets a new benchmark for scalable and efficient AI agents capable of handling vast tool repositories. Looking ahead, ToolGen opens doors for integrating advanced techniques like chain-of-thought reasoning, reinforcement learning, and ReAct , further enhancing the autonomy and versatility of LLMs in real-world applications. 

References 

Asai et al. (2023) Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection, 2023. URL https://arxiv.org/abs/2310.11511 . 

Brohan et al. (2023) Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023. URL https://arxiv.org/abs/2307.15818 . 

Brown et al. (2024) Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787 , 2024. 

Brown et al. (2020) Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020. URL https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html . 

Chen et al. (2023a) Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. Fireact: Toward language agent fine-tuning. arXiv preprint arXiv:2310.05915 , 2023a. 

Chen et al. (2023b) Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, and Xueqi Cheng. Continual learning for generative retrieval over dynamic corpora. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management , pp. 306–315, 2023b. 

Chen et al. (2023c) Jiangui Chen, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, and Xueqi Cheng. Continual Learning for Generative Retrieval over Dynamic Corpora. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management , CIKM ’23, pp. 306–315, New York, NY, USA, 2023c. Association for Computing Machinery. ISBN 9798400701245. doi: 10.1145/3583780.3614821 . URL https://dl.acm.org/doi/10.1145/3583780.3614821 . 

Chen et al. (2024a) Junzhi Chen, Juhao Liang, and Benyou Wang. Smurfs: Leveraging multiple proficiency agents with context-efficiency for tool planning, 2024a. URL https://arxiv.org/abs/2405.05955 . 

Chen et al. (2024b) Yanfei Chen, Jinsung Yoon, Devendra Singh Sachan, Qingze Wang, Vincent Cohen-Addad, Mohammadhossein Bateni, Chen-Yu Lee, and Tomas Pfister. Re-invoke: Tool invocation rewriting for zero-shot tool retrieval. arXiv preprint arXiv:2408.01875 , 2024b. 

Chen et al. (2024c) Zehui Chen, Kuikun Liu, Qiuchen Wang, Wenwei Zhang, Jiangning Liu, Dahua Lin, Kai Chen, and Feng Zhao. Agent-flan: Designing data and methods of effective agent tuning for large language models, 2024c. URL https://arxiv.org/abs/2403.12881 . 

Dao (2024) Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR) , 2024. 

Dao et al. (2022) Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems (NeurIPS) , 2022. 

Dubey et al. (2024) Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 , 2024. 

Gravitas (2023) Gravitas. AutoGPT, 2023. URL https://github.com/Significant-Gravitas/AutoGPT . 

Guo et al. (2024) Zhicheng Guo, Sijie Cheng, Hao Wang, Shihao Liang, Yujia Qin, Peng Li, Zhiyuan Liu, Maosong Sun, and Yang Liu. StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models, 2024. URL https://arxiv.org/abs/2403.07714 . 

Hao et al. (2023) Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html . 

Järvelin & Kekäläinen (2002) Kalervo Järvelin and Jaana Kekäläinen. Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems (TOIS) , 20(4):422–446, 2002. 

Karpukhin et al. (2020) Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pp. 6769–6781, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550 . URL https://aclanthology.org/2020.emnlp-main.550 . 

Kishore et al. (2023a) Varsha Kishore, Chao Wan, Justin Lovelace, Yoav Artzi, and Kilian Q Weinberger. Incdsi: incrementally updatable document retrieval. In International Conference on Machine Learning , pp. 17122–17134. PMLR, 2023a. 

Kishore et al. (2023b) Varsha Kishore, Chao Wan, Justin Lovelace, Yoav Artzi, and Kilian Q. Weinberger. Incdsi: Incrementally updatable document retrieval. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA , volume 202 of Proceedings of Machine Learning Research , pp. 17122–17134. PMLR, 2023b. URL https://proceedings.mlr.press/v202/kishore23a.html . 

(21) Yilun Kong, Jingqing Ruan, YiHong Chen, Bin Zhang, Tianpeng Bao, Hangyu Mao, Ziyue Li, Xingyu Zeng, Rui Zhao, Xueqian Wang, et al. Tptu-v2: Boosting task planning and tool usage of large language model-based agents in real-world systems. 

Kong et al. (2024) Yilun Kong, Jingqing Ruan, Yihong Chen, Bin Zhang, Tianpeng Bao, Shi Shiwei, Du Qing, Xiaoru Hu, Hangyu Mao, Ziyue Li, et al. Tptu-v2: Boosting task planning and tool usage of large language model-based agents in real-world industry systems, 2024. 

Liu et al. (2024) Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. In The Twelfth International Conference on Learning Representations , 2024. 

Liu et al. (2023) Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, et al. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents. arXiv preprint arXiv:2308.05960 , 2023. 

Mehta et al. (2023a) Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q Tran, Jinfeng Rao, Marc Najork, Emma Strubell, and Donald Metzler. Dsi++: Updating transformer memory with new documents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pp. 8198–8213, 2023a. 

Mehta et al. (2023b) Sanket Vaibhav Mehta, Jai Gupta, Yi Tay, Mostafa Dehghani, Vinh Q. Tran, Jinfeng Rao, Marc Najork, Emma Strubell, and Donald Metzler. DSI++: Updating transformer memory with new documents. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pp. 8198–8213, Singapore, 2023b. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.510 . URL https://aclanthology.org/2023.emnlp-main.510 . 

OpenAI (2024) OpenAI. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774 . 

Ouyang et al. (2022) Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems , 35:27730–27744, 2022. 

Patil et al. (2023) Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis, 2023. URL https://arxiv.org/abs/2305.15334 . 

Qiao et al. (2024) Shuofei Qiao, Ningyu Zhang, Runnan Fang, Yujie Luo, Wangchunshu Zhou, Yuchen Eleanor Jiang, Chengfei Lv, and Huajun Chen. Autoact: Automatic agent learning from scratch for qa via self-planning, 2024. URL https://arxiv.org/abs/2401.05268 . 

Qin et al. (2023) Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs, 2023. URL https://arxiv.org/abs/2307.16789 . 

Qu et al. (2024) Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. Tool learning with large language models: A survey. arXiv preprint arXiv:2405.17935 , 2024. 

Rajbhandari et al. (2020) Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models, 2020. URL https://arxiv.org/abs/1910.02054 . 

Robertson et al. (2009) Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval , 3(4):333–389, 2009. 

(35) Jingqing Ruan, Yihong Chen, Bin Zhang, Zhiwei Xu, Tianpeng Bao, Hangyu Mao, Ziyue Li, Xingyu Zeng, Rui Zhao, et al. Tptu: Task planning and tool usage of large language model-based ai agents. 

Schick et al. (2023) Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023. URL https://arxiv.org/abs/2302.04761 . 

Shen et al. (2024) Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, and Fei Huang. Small llms are weak tool learners: A multi-llm agent, 2024. URL https://arxiv.org/abs/2401.07324 . 

Shinn et al. (2023) Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Proceedings of the 37th International Conference on Neural Information Processing Systems , pp. 8634–8652, 2023. 

Snell et al. (2024) Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314 , 2024. 

Sun et al. (2023a) Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. Learning to tokenize for generative retrieval, 2023a. URL https://arxiv.org/abs/2304.04171 . 

Sun et al. (2023b) Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. Learning to tokenize for generative retrieval. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023b. URL http://papers.nips.cc/paper_files/paper/2023/hash/91228b942a4528cdae031c1b68b127e8-Abstract-Conference.html . 

Team (2024) Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/blog/qwen2.5/ . 

Touvron et al. (2023) Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. URL https://arxiv.org/abs/2302.13971 . 

Wang et al. (2024a) Renxi Wang, Haonan Li, Xudong Han, Yixuan Zhang, and Timothy Baldwin. Learning from failure: Integrating negative examples when fine-tuning large language models as agents. arXiv preprint arXiv:2402.11651 , 2024a. 

Wang et al. (2024b) Shu Wang, Muzhi Han, Ziyuan Jiao, Zeyu Zhang, Ying Nian Wu, Song-Chun Zhu, and Hangxin Liu. Llm3: Large language model-based task and motion planning with motion failure reasoning. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pp. 12086–12092, 2024b. doi: 10.1109/IROS58592.2024.10801328 . 

Wang et al. (2024c) Shu Wang, Lei Ji, Renxi Wang, Wenxiao Zhao, Haokun Liu, Yifan Hou, and Ying Nian Wu. Explore the reasoning capability of llms in the chess testbed, 2024c. URL https://arxiv.org/abs/2411.06655 . 

Wang et al. (2022) Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Sun, Weiwei Deng, Qi Zhang, and Mao Yang. A neural corpus indexer for document retrieval. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022 , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/a46156bd3579c3b268108ea6aca71d13-Abstract-Conference.html . 

Wei et al. (2023) Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/2201.11903 . 

Wu et al. (2024a) Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. Autogen: Enabling next-gen llm applications via multi-agent conversation framework. In COLM , 2024a. 

Wu et al. (2024b) Qinzhuo Wu, Wei Liu, Jian Luan, and Bin Wang. ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback, 2024b. URL https://arxiv.org/abs/2409.14826 . 

(51) Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for llm problem-solving. In The 4th Workshop on Mathematical Reasoning and AI at NeurIPS’24 . 

Xiong et al. (2021) Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021 . OpenReview.net, 2021. URL https://openreview.net/forum?id=zeFrfgyZln . 

Xu et al. (2024) Qiancheng Xu, Yongqi Li, Heming Xia, and Wenjie Li. Enhancing tool retrieval with iterative feedback from large language models. arXiv preprint arXiv:2406.17465 , 2024. 

Yao et al. (2023) Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR) , 2023. 

Yin et al. (2024) Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin. Agent lumos: Unified and modular training for open-source language agents. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pp. 12380–12403, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.670 . URL https://aclanthology.org/2024.acl-long.670 . 

Zeng et al. (2023) Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. Agenttuning: Enabling generalized agent abilities for llms, 2023. 

Appendix A More Related Work 

Previous work include Toolformer and ToolkenGPT, already employed vocabulary expansion for tool learning. The main difference between our work and the others is: previous studies primarily demonstrate that through SFT (in Toolformer) or adding new tool tokens with pre-computed embeddings (in ToolKenGPT), LLMs can learn to use a very small number of tools. However, in real-world tool-calling (agent) scenarios, previous methods require listing available tools in the prompt, which greatly limits their practical use. Examples can be seen in Figure 5 . 

Other studies, such as ToolPlanner ( Wu et al., 2024b ) and AutoACT ( Qiao et al., 2024 ) , have used reinforcement learning or developed multi-agent systems to enhance tool learning or task completion ( Qiao et al., 2024 ; Liu et al., 2023 ; Shen et al., 2024 ; Chen et al., 2024a ) . We do not compare our model with these approaches for two reasons: (1) Most of these works rely on feedback mechanisms, either through Reflection ( Shinn et al., 2023 ) or a reward model, which is similar to ToolBench’s evaluation design, where an LLM serves as an evaluator without access to ground truth answers. However, this is not the focus of our study, and our end-to-end experiment does not rely on such feedback mechanisms. (2) Our method is not in conflict with these approaches; instead, they can be integrated. Exploring this integration is left for future work. 
Figure 5: Real examples from ToolkenGPT, Toolformer, and ToolGen (ours). Both ToolkenGPT and Toolformer describe tools available in the prompt, while ToolGen does not require tools been mentioned in its prompt. 
Appendix B Tool Extension and Maintenance 

In ToolGen and other generative retrieval systems, tools or documents are embedded into the model’s parameters. Therefore, how to add and maintenance new tools/documents become challenging. For ToolGen, it not only generates the proper tool, but also fetches the documentation for that tool. If there are minor changes that the tool usage scenarios keep the same (e.g. small parameter changes), it can still generate the tool and rely on the fetched documentation to do further tasks. 

For vast changes that the usage scenarios are different or adding totally new tools, we admit that ToolGen is not able to utilize these tools. However, this inefficiency exists and is persistent for generative retrieval systems Sun et al. (2023a) ; Chen et al. (2023b) ; Mehta et al. (2023a) . Current methods to adapt these changes include continual training and constrained optimization ( Mehta et al., 2023a ; Kishore et al., 2023a ) , which we believe could also be applied to ToolGen to alleviate the above challenges. 

Despite that ToolGen is inefficient of adopting to new tools, its unified design lead to unique advantages such as easy integration with Chain-of-Thought ( Wei et al., 2023 ) , Reinforcement Learning with Human Feedback ( Ouyang et al., 2022 ) , and inference time scaling ( Brown et al., 2024 ; Snell et al., 2024 ; Wu et al., ) . We leave the problem of maintaining and adding tools to future work. 

Appendix C Real Tool Example 

Figure 6 shows a real tool example. Each tool is a collection of several APIs. In our experiments, the following fields are used: "tool_name" is the name of the tool. "tool_description" describes tool related information such as the functionality of the tool. In each API, "name" is the name of the API. "description" describes API related information. "method" is the http method for calling the API. "required_parameters" are parameters that must be filled when calling the API. Optionally, "optional_parameters" can be set for extra parameters. 
Figure 6: A real tool example. The tool contains one API. We have removed unnecessary fields for simplicity. 
Appendix D Tool Virtualization Implementation 

ToolGen adopts a single and unique token to represent a tool, which shows its superiority for tool retrieval and tool calling. We also introduced other methods to index a tool, including semantic, numeric, and hierarchical. The following is a detailed implementation of how we implement each indexing. 

Atomic 

indexing is the method we use in ToolGen. Compared to other methods, it takes a single token as a tool and does not hallucinate to nonexistent tools. We use <<tool name&&api name>> to combine the tool name and api name to form a single token. For example, for the example in Appendix C , the resulting token is <<Youtube Hub&&Get Video Details>> . 

Semantic 

indexing maps each tool to the name used in ToolBench, which is also a combination between tool name and API name. However, the name can be tokenized into multiple tokens so that the model can perceive its semantic meanings. For the example in Appendix C , the resulted mapping is get_video_details_for_youtube_hub . 

Numeric 

indexing maps each tool to a unique number. We first get a list of all tools, with a length about 47,000. For all tools, we use a five digit number separated by space to represent the tool. If the example in Appendix C is the 128th element in the list, we use 0 0 0 1 2 8 to represent the tool. Since Llama-3 tokenizer encodes each number separately, numeric indexing will lead to tool tokens with same number of sub-tokens. 

Hierarchical 

also maps each tool into a number. Different from Numeric indexing, we inject structure information into the tool representation by iterative clustering. During each iteration, we cluster tools into ten clusters, where each cluster is assigned a number from 0 to 9 . For each cluster, we repeat this clustering process until there is only one tool in the cluster. These steps form a clustering tree. We take the number from root to the leaf as the representation to the tool in that leaf. The example in Appendix C may be assigned a number longer than five digits, such as 0 1 2 2 3 3 3 . 

Appendix E Constrained Beam Search 

E.1 Implementation 

During retrieval and completing end-to-end agent tasks, we use constrained beam search to limit the generated actions to be valid tool tokens. The detailed steps are shown in Algorithm 1 . T
