# Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation

Source URL: https://arxiv.org/abs/2606.04435

[2606.04435] Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation Skip to main content Press Enter to search · Advanced search 
Computer Science > Artificial Intelligence 
arXiv:2606.04435 (cs) [Submitted on 3 Jun 2026] 
Title: Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation 
Authors: Saroj Mishra View a PDF of the paper titled Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation, by Saroj Mishra View PDF HTML (experimental) 
Abstract: Multi-step agentic retrieval-augmented generation (RAG) pipelines have demonstrated significant capability for complex reasoning tasks, yet remain vulnerable to a class of failure that existing hallucination detection mechanisms systematically miss: cascading hallucination, where errors introduced at early pipeline stages propagate and amplify across successive reasoning steps, producing confident but factually incorrect final outputs. To address this vulnerability, we formalize cascading hallucination as a distinct failure mode in agentic RAG systems, present a four-type taxonomy of cascade patterns, and introduce CHARM (Cascading Hallucination Aware Resolution and Mitigation), an architectural framework for detecting and interrupting error propagation in multi-step reasoning pipelines. CHARM comprises four components - stage-level fact verification, cross-stage consistency tracking, confidence propagation monitoring, and cascade resolution triggering - that operate alongside standard agentic RAG pipelines without requiring architectural replacement. We evaluate CHARM on HotpotQA, MuSiQue, 2WikiMultiHopQA, and a custom adversarial dataset across LangChain agentic pipeline configurations, achieving an 89.4% cascade detection rate with a 5.3% false positive rate and 215 ms +/- 18 ms average latency overhead per stage, achieving an error propagation reduction of 82.1%, compared to 18.5% for output-level detectors. Component ablations confirm that each detection module contributes meaningfully to overall cascade coverage. CHARM integrates with human-in-the-loop oversight frameworks to provide a complete reliability and governance stack for production agentic AI deployment. 
Subjects: Artificial Intelligence (cs.AI) ; Computation and Language (cs.CL); Cryptography and Security (cs.CR); Information Retrieval (cs.IR) Cite as: arXiv:2606.04435 [cs.AI] (or arXiv:2606.04435v1 [cs.AI] for this version) https://doi.org/10.48550/arXiv.2606.04435 arXiv-issued DOI via DataCite 
Submission history 
From: Saroj Mishra [ view email ] [v1] Wed, 3 Jun 2026 04:33:47 UTC (806 KB) Full-text links: 
Access Paper: 
View a PDF of the paper titled Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation, by Saroj Mishra 
View PDF 

HTML (experimental) 

TeX Source 
view license 
Current browse context: 
cs.AI < prev | next > new | recent | 2026-06 Change to browse by: cs cs.CL cs.CR cs.IR 
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
