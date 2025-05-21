# Insight-RAG: Enhancing LLMs with Insight-Driven Augmentation
This repository is the implementation for the paper "[Insight-RAG: Enhancing LLMs with Insight-Driven Augmentation](https://arxiv.org/pdf/2504.00187)".

![alt text](https://github.com/megagonlabs/Insight-RAG/blob/main/overview-insight-rag.png)

---
## 🚀 **Quick Start**
Running the Insight-RAG pipeline is a two-step process. Before you begin, please add your Hugging Face access token and OpenAI API key to 'src/code/keys.py'.

1. **Continually pre-train the Insight Miner**

    ```bash
    # from the repo root
    cd src
    ./run_insight_miner.sh <DOMAIN> <BASE_MODEL>
    ```

    Example:

    ```bash
    ./run_insight_miner.sh AAN meta-llama/Llama-3.2-3B
    ```

    * `<DOMAIN>` — the corpus/domain you want to mine insights from  
    * `<BASE_MODEL>` — the base LLM to continually pre-train  

2. **Run Insight-RAG**

    After the miner has finished training:

    ```bash
    ./run_insight-rag.sh <DOMAIN> <MAIN_MODEL> <INSIGHT_MINER_MODEL>
    ```

    Example:

    ```bash
    ./run_insight-rag.sh AAN gpt-4o-mini meta-llama/Llama-3.2-3B
    ```

    * `<DOMAIN>` — same domain used in Step 1
    * `<MAIN_MODEL>` — model used for insight identification & answer generation 
    * `<INSIGHT_MINER_MODEL>` — model used as the insight miner   


---
## 🛠️ **Dataset Construction Process**

We use the AAN and OC papers abstract datasets to build six focused benchmarks that probe weaknesses of standard RAG pipelines. We start by sampling 5 k papers per dataset via BFS. We then extract subject–relation–object triples with GPT-4o mini, and finally using manual/rule-based approach, filter and normalize the resulting relations. We used the filtered triples to build our benchmarks.

**Benchmarks:**
- **Deeply Buried Insight**    
    - Sample triples with subject-relation pairs that map to a single object, and the subject *and* object appear exactly once in the abstract, ensuring the fact is deeply buried.  
    - Convert each triple into a question using GPT-4o mini; manually vet for quality.
- **Multi-Source Insight**  
    - Choose subject-relation pairs that map to multiple objects drawn from different papers.  
    - Generate corresponding questions with GPT-4o mini.  
    - Manually filter out noisy or vague questions.
- **Non-QA Task – Citation Recommendation**  
    - Leverage the paper-pair (citation) labels from CDA dataset.  
    - Frame a citation-recommendation task to test Insight-RAG on problems beyond traditional QA.

---

## 📝 **Data Source Attribution**

Our benchmarks build upon data derived from a publicly available dataset:

- **CDA**  
   - Source: [CDA Website](https://multitextalign.xuhuiz.com/)  
   - License: **Public Domain**

The CDA benchmarks were created using several publicly available datasets, with two of them serving as the primary data sources for our benchmarks:

- **AAN**
   - Source: [AAN Website](https://clair.eecs.umich.edu/aan/index.php)  
   - License: **Public Domain**
- **OC**
  - Source: [OC GitHub](https://github.com/allenai/citeomatic)
  - License: **[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)**
    
**Please refer to the respective sources for detailed licensing terms.**

---

## 🤖 **AI-Generated Content Disclaimer**

Parts of our datasets, including the extracted triples and their conversion into questions, were **generated using OpenAI's GPT-4o mini model**.  
- The content adheres to OpenAI's [Usage Policies](https://openai.com/policies/terms-of-use).  
- Outputs were reviewed and refined to align with the dataset's objectives.  
- No prohibited use cases or violations of OpenAI's terms are present in this dataset.

Please ensure compliance with OpenAI's policies if redistributing or modifying this dataset.

---

## 🧠 **Usage Guidelines**

- Use this dataset for **research and educational purposes**.  
- Commercial use may require additional permissions depending on source licenses.  

---

## ⭐ **Citation**

If you would like to cite our work, the bibtex is:

    @article{pezeshkpour2025insight,
    title={Insight-RAG: Enhancing LLMs with Insight-Driven Augmentation},
    author={Pezeshkpour, Pouya and Hruschka, Estevam},
    journal={arXiv preprint arXiv:2504.00187},
    year={2025}
    }

---

## 📜 **Disclosure**
Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses, which may be CC license and Apache 2.0 license. In the event of conflicts between Megagon Labs, Inc., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party’s license conditions. All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein. While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.

