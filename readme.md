# SciNetBench: A Relation-Aware Benchmark for Scientific Literature Retrieval Agents


This repository contains the official implementation and dataset for **SciNetBench** (A Relation-Aware Benchmark for Scientific Literature Retrieval Agents), the first benchmark designed to systematically evaluate the relational understanding of literature retrieval systems in the scientific domain.

---

## 🎯 Benchmark Tasks

RARE evaluates retrieval systems across three distinct, increasingly complex levels of relational retrieval:

### 1. Ego-centric Retrieval
Tasking systems with retrieving papers based on their intrinsic properties.  
Examples:
- Finding seminal works  
- Identifying papers with novel methodologies  
- Locating papers contradicted by subsequent research  

### 2. Pair-wise Relationship Identification
Evaluating the ability to identify direct scholarly relationships between two papers.  
Examples:
- Finding papers that support a given claim  
- Identifying contradictory studies  
- Detecting papers that extend prior work  

### 3. Path-wise Trajectory Reconstruction
The most complex task, requiring systems to reconstruct the evolutionary path of a scientific idea or method.  
Examples:
- Tracing a coherent chain of papers representing a scientific lineage  
- Mapping debates and idea evolution over time  

---

## 📊 Dataset

- Built upon a corpus of **18M+ AI-related papers**  
- Includes curated queries, ground-truth documents, and relational metadata  
- Supports evaluation of all three benchmark tasks  

We are committed to making RARE accessible to the research community to foster innovation in this critical area.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+  
- PyTorch 1.10+  
- Hugging Face Transformers  
- Faiss  

### Installation

Clone the repository:
```bash
git clone https://github.com/your-username/xxx.git

bash
pip install -r requirements.txt
Download the dataset:

bash
# Instructions to download and set up the dataset will be provided here.
⚙️ Usage
To evaluate your own retrieval model on RARE, you need to implement a simple wrapper function.
See evaluation.py for a detailed example.

Example evaluation script:

bash
复制代码
python evaluate.py \
    --model_name_or_path your/retrieval/model \
    --task ego-centric \
    --output_dir ./results

```

🤝 Contributing

We welcome contributions from the community!
If you have suggestions for improving the benchmark, adding new tasks, or fixing a bug, please feel free to open an issue or submit a pull request.

📄 License

This project is licensed under the MIT License.
See the LICENSE file for details.