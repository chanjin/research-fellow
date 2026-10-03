# Process Reward Models for LLM Agents: Practical Framework and Directions

Source URL: https://arxiv.org/abs/2502.10325

[2502.10325] Process Reward Models for LLM Agents: Practical Framework and Directions Skip to main content Press Enter to search · Advanced search 
Computer Science > Machine Learning 
arXiv:2502.10325 (cs) [Submitted on 14 Feb 2025] 
Title: Process Reward Models for LLM Agents: Practical Framework and Directions 
Authors: Sanjiban Choudhury View a PDF of the paper titled Process Reward Models for LLM Agents: Practical Framework and Directions, by Sanjiban Choudhury View PDF HTML (experimental) 
Abstract: We introduce Agent Process Reward Models (AgentPRM), a simple and scalable framework for training LLM agents to continually improve through interactions. AgentPRM follows a lightweight actor-critic paradigm, using Monte Carlo rollouts to compute reward targets and optimize policies. It requires minimal modifications to existing RLHF pipelines, making it easy to integrate at scale. Beyond AgentPRM, we propose InversePRM, which learns process rewards directly from demonstrations without explicit outcome supervision. We also explore key challenges and opportunities, including exploration, process reward shaping, and model-predictive reasoning. We evaluate on ALFWorld benchmark, show that small 3B models trained with AgentPRM and InversePRM outperform strong GPT-4o baselines, and analyze test-time scaling, reward hacking, and more. Our code is available at: this https URL . 
Comments: 17 pages, 7 figures Subjects: Machine Learning (cs.LG) ; Artificial Intelligence (cs.AI) Cite as: arXiv:2502.10325 [cs.LG] (or arXiv:2502.10325v1 [cs.LG] for this version) https://doi.org/10.48550/arXiv.2502.10325 arXiv-issued DOI via DataCite 
Submission history 
From: Sanjiban Choudhury [ view email ] [v1] Fri, 14 Feb 2025 17:34:28 UTC (3,157 KB) Full-text links: 
Access Paper: 
View a PDF of the paper titled Process Reward Models for LLM Agents: Practical Framework and Directions, by Sanjiban Choudhury 
View PDF 

HTML (experimental) 

TeX Source 
view license 
Current browse context: 
cs.LG < prev | next > new | recent | 2025-02 Change to browse by: cs cs.AI 
References & Citations 

NASA ADS 

Google Scholar 

Semantic Scholar 
Loading... 
BibTeX formatted citation 
loading... Data provided by: 
Bookmark 
Bibliographic Tools 
Bibliographic and Citation Tools 
Bibliographic Explorer Toggle Bibliographic Explorer ( What is the Explorer? ) Connected Papers Toggle Connected Papers ( What is Connected Papers? ) Litmaps Toggle Litmaps ( What is Litmaps? ) scite.ai Toggle scite Smart Citations ( What are Smart Citations? ) Code, Data, Media 
Code, Data and Media Associated with this Article 
alphaXiv Toggle alphaXiv ( What is alphaXiv? ) Links to Code Toggle CatalyzeX Code Finder for Papers ( What is CatalyzeX? ) DagsHub Toggle DagsHub ( What is DagsHub? ) GotitPub Toggle Gotit.pub ( What is GotitPub? ) Huggingface Toggle Hugging Face ( What is Huggingface? ) ScienceCast Toggle ScienceCast ( What is ScienceCast? ) Demos 
Demos 
Replicate Toggle Replicate ( What is Replicate? ) Spaces Toggle Hugging Face Spaces ( What is Spaces? ) Spaces Toggle TXYZ.AI ( What is TXYZ.AI? ) Related Papers 
Recommenders and Search Tools 
Link to Influence Flower Influence Flower ( What are Influence Flowers? ) Core recommender toggle CORE Recommender ( What is CORE? ) IArxiv recommender toggle IArxiv Recommender ( What is IArxiv? ) 
Author 

Venue 

Institution 

Topic 
About arXivLabs 
arXivLabs: experimental projects with community collaborators 

arXivLabs is a framework that allows collaborators to develop and share new arXiv features directly on our website. 

Both individuals and organizations that work with arXivLabs have embraced and accepted our values of openness, community, excellence, and user data privacy. arXiv is committed to these values and only works with partners that adhere to them. 

Have an idea for a project that will add value for arXiv's community? Learn more about arXivLabs . 

Which authors of this paper are endorsers? | Disable MathJax ( What is MathJax? )
