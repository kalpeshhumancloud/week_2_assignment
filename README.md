# week_2_assignment
# BERT Attention Visualization

This project demonstrates how to use HuggingFace's **Transformers** library with **BertViz** to visualize attention heads of a pre-trained BERT model (`bert-base-uncased`).  
It tokenizes an input sentence, runs it through BERT, extracts hidden states and attention weights, and then displays an **interactive attention visualization**.

---

## Features
- Tokenize input text using `BertTokenizer`.
- Run forward pass through `BertModel` with:
  - Hidden states (`output_hidden_states=True`)
  - Attention weights (`output_attentions=True`)
- Print model outputs:
  - Input IDs
  - Tokens
  - Number and shapes of hidden layers
  - Number and shapes of attention layers
- Visualize attention using **BertViz** `head_view`.

---

## Installation

1. **Clone or download this repository**  
   ```bash
   git clone https://github.com/your-username/bert-attention-visualization.git
   cd bert-attention-visualization
---

2. **Install dependencies**

Install required packages (upgrade pip first). For CPU-only setups:

```bash
pip install --upgrade pip
pip install torch transformers bertviz notebook
```

If you have a CUDA-enabled GPU, install the matching PyTorch build following the official PyTorch installation instructions to ensure CUDA compatibility.

--- 
