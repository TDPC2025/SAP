# Image Generation from Contextually-Contradictory Prompts

> **Saar Huberman, Or Patashnik, Omer Dahary, Ron Mokady, Daniel Cohen-Or**
> 
> Text-to-image diffusion models excel at generating high-quality, diverse images from natural language prompts. However, they often fail to produce semantically accurate results when the prompt contains concept combinations that contradict their learned priors. We define this failure mode as contextual contradiction, where one concept implicitly negates another due to entangled associations learned during training. To address this, we propose a stage-aware prompt decomposition framework that guides the denoising process
using a sequence of proxy prompts. Each proxy prompt is constructed to match the semantic content expected to emerge at a specific
stage of denoising, while ensuring contextual coherence. To construct these proxy prompts, we leverage a large language model (LLM) to analyze the target prompt, identify contradictions, and generate alternative expressions that preserve the original intent while resolving contextual conflicts. By aligning prompt information with the denoising progression, our method enables fine-grained semantic control and accurate image generation in the presence of contextual contradictions. Experiments across a variety of challenging prompts show substantial improvements in alignment to the textual prompt.

<a href="https://tdpc2025.github.io/SAP/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://arxiv.org/abs/2506.01929"><img src="https://img.shields.io/badge/arXiv-SAP-b31b1b.svg" height=20.5></a>
<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/omer11a/bounded-attention) -->

<p align="center">
<img src="images/teaser.jpg" width="800px"/>
</p>

## Description  
Official implementation of our "Image Generation from Contextually-Contradictory Prompts" paper.

---

## Setup

### Environment
```
This project uses [`uv`](https://github.com/astral-sh/uv), a modern Python package manager and virtual environment tool.

1. Clone the repository:
git clone https://github.com/TDPC2025/SAP.git
cd SAP

2. install uv (if not already installed):
curl -Ls https://astral.sh/uv/install.sh | sh

3. Create and activate the environment:
uv venv
source .venv/bin/activate

4. Install dependencies:
uv pip install --requirements pyproject.toml

This will install all required packages listed in pyproject.toml and lock their exact versions using uv.lock.
```

## Usage
Once the environment is set up, you can generate images using the SAP pipeline by running:
```
python run_SAP_flux.py --prompt "your prompt" --seeds_list seed1 seed2 seed3
```

for example:
```
python run_SAP_flux.py --prompt "A bear is performing a handstand in the park" --seeds_list 30498
```

Before running, make sure to insert your API key in the run_SAP_flux.py script:
```
API_KEY = "YOUR_API_KEY"
```
All generated images will be saved to:
```
results/<prompt>/Seed<seed>.png
```
## 📊 Benchmarks

We evaluate our method using three benchmarks designed to challenge text-to-image models with **contextually contradictory prompts**:

- **Whoops!**  
  A dataset of 500 prompts designed to expose failures in visual reasoning when faced with commonsense-defying descriptions.

- **Whoops-Hard** (✨ introduced in this paper)  
  A curated subset of 100 particularly challenging prompts from Whoops! where existing models often fail to preserve semantic intent.

- **ContraBench** (🆕 introduced in this paper)  
  A novel benchmark of 40 prompts carefully constructed to include **Contextual contradictions**.

### 🧪 Evaluation

We include `gpt_eval.py`, the automatic evaluator used in the paper.  
It uses GPT-4o to assess image–text alignment by scoring how well generated images reflect the semantics of the prompt.


### 📁 Benchmarks Structure

All benchmark-related resources are organized under the `benchmarks/` folder:

```
benchmarks/
├── original_prompts/ # Raw prompts for Whoops!, Whoops-Hard, and ContraBench
├── SAP_prompts/ # Decomposed proxy prompts from our method
├── evaluated_seeds/ # Fixed seeds used for reproducibility
└── gpt_eval.py # GPT-based evaluator for semantic alignment
```

## Acknowledgements 

This code was built using the code from the following repositories:
- [diffusers](https://github.com/huggingface/diffusers)

## Citation

If you use this code for your research, please cite our paper:

```
@article{huberman2025image,
  title={Image Generation from Contextually-Contradictory Prompts},
  author={Huberman, Saar and Patashnik, Or and Dahary, Omer and Mokady, Ron and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2506.01929},
  year={2025}
}
```