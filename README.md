# Mechanistic Interpretability Analysis on GPT-2 Small

> **Work in progress.** This project is under active development. Current findings are preliminary and exploratory.

This project uses the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library to perform mechanistic interpretability analysis on the GPT-2 Small model, following the methodology of Neel Nanda's [Exploratory Analysis Demo](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb) as a starting point.

This work builds on my Master's [thesis](https://amslaurea.unibo.it/id/eprint/37599/), "Mechanistic Interpretability Tool for Transformer-Based Language Models", completed at the University of Bologna. While the thesis addresses the theoretical and library-level foundations of mechanistic interpretability, this project is its empirical complement — a hands-on investigation of model behaviour conducted independently after graduation.
 
## Research Question
 
How does GPT-2 Small arbitrate between in-context learning and parametric knowledge when the two are in conflict?
 
More specifically: in few-shot prompting scenarios, to what extent do induction heads — the attention mechanism responsible for in-context learning — override factual knowledge stored in the model's MLP layers? And under what conditions does one mechanism dominate the other?
 
## Approach
 
The analysis is conducted through systematic prompt probing across 1- to 5-shot settings. By varying prompt structure, answer ordering, syntactic formatting, and the use of specific keywords, the project observes how changes in context influence the model's output distribution — and what this reveals about the relative contributions of induction heads and MLPs.
 
## Preliminary Findings
 
- In 1-shot settings, the model tends to align with the single in-context example, even when it contradicts well-established factual knowledge.
- In 2-shot settings with alternating answers, the model appears to follow the sequential pattern of responses rather than draw on parametric knowledge.
- From 3 shots onward, the model increasingly aligns with the majority answer among in-context examples, suggesting a frequency-based heuristic.
- Certain keywords (e.g. `Answer:`) appear to amplify specific responses independently of factual content, rather than activating parametric recall.

These observations are consistent with the hypothesis that induction heads drive in-context learning and that MLP-stored knowledge is largely suppressed when a clear contextual pattern is present.
 
## Roadmap
 
Planned next steps to corroborate the behavioural findings with direct mechanistic evidence:
 
- Activation patching to isolate the contribution of specific attention heads and MLP layers to the model's output in the identified prompt configurations.
- Attention pattern visualisation for the most revealing few-shot cases, to verify induction head activation directly.
- Logit lens analysis to trace how the model's prediction evolves across layers.

## Repository Structure
 
```
.
└── exploratory_analysis_gpt2_small.ipynb   # Main analysis notebook
```
 
## Requirements
 
```
transformer_lens
circuitsvis
einops
fancy_einsum
jaxtyping
plotly
torch
```
 
## Running the Notebook
 
The notebook is designed to run in Google Colab with GPU acceleration enabled. Go to **Runtime > Change Runtime Type** and select **GPU** as the hardware accelerator before running.
