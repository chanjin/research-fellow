# Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks

Source URL: https://arxiv.org/abs/2501.11733

[2501.11733] Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks Skip to main content Press Enter to search · Advanced search 
Computer Science > Computation and Language 
arXiv:2501.11733 (cs) [Submitted on 20 Jan 2025 ( v1 ), last revised 28 Jan 2025 (this version, v2)] 
Title: Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks 
Authors: Zhenhailong Wang , Haiyang Xu , Junyang Wang , Xi Zhang , Ming Yan , Ji Zhang , Fei Huang , Heng Ji View a PDF of the paper titled Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks, by Zhenhailong Wang and 7 other authors View PDF HTML (experimental) 
Abstract: Smartphones have become indispensable in modern life, yet navigating complex tasks on mobile devices often remains frustrating. Recent advancements in large multimodal model (LMM)-based mobile agents have demonstrated the ability to perceive and act in mobile environments. However, current approaches face significant limitations: they fall short in addressing real-world human needs, struggle with reasoning-intensive and long-horizon tasks, and lack mechanisms to learn and improve from prior experiences. To overcome these challenges, we introduce Mobile-Agent-E, a hierarchical multi-agent framework capable of self-evolution through past experience. By hierarchical, we mean an explicit separation of high-level planning and low-level action execution. The framework comprises a Manager, responsible for devising overall plans by breaking down complex tasks into subgoals, and four subordinate agents--Perceptor, Operator, Action Reflector, and Notetaker--which handle fine-grained visual perception, immediate action execution, error verification, and information aggregation, respectively. Mobile-Agent-E also features a novel self-evolution module which maintains a persistent long-term memory comprising Tips and Shortcuts. Tips are general guidance and lessons learned from prior tasks on how to effectively interact with the environment. Shortcuts are reusable, executable sequences of atomic operations tailored for specific subroutines. The inclusion of Tips and Shortcuts facilitates continuous refinement in performance and efficiency. Alongside this framework, we introduce Mobile-Eval-E, a new benchmark featuring complex mobile tasks requiring long-horizon, multi-app interactions. Empirical results show that Mobile-Agent-E achieves a 22% absolute improvement over previous state-of-the-art approaches across three foundation model backbones. Project page: this https URL . 
Subjects: Computation and Language (cs.CL) ; Computer Vision and Pattern Recognition (cs.CV) Cite as: arXiv:2501.11733 [cs.CL] (or arXiv:2501.11733v2 [cs.CL] for this version) https://doi.org/10.48550/arXiv.2501.11733 arXiv-issued DOI via DataCite 
Submission history 
From: Zhenhailong Wang [ view email ] [v1] Mon, 20 Jan 2025 20:35:46 UTC (36,446 KB) [v2] Tue, 28 Jan 2025 16:58:02 UTC (36,446 KB) Full-text links: 
Access Paper: 
View a PDF of the paper titled Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks, by Zhenhailong Wang and 7 other authors 
View PDF 

HTML (experimental) 

TeX Source 
view license 
Current browse context: 
cs.CL < prev | next > new | recent | 2025-01 Change to browse by: cs cs.CV 
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
Link to Influence Flower Influence Flower ( What are Influence Flowers? ) Core recommender toggle CORE Recommender ( What is CORE? ) 
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
