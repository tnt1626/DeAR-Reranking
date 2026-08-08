# MedPRS Dataset Reproduction & Evaluation Results

This document summarizes the evaluation results of reproducing the **MedPRS** journal recommendation pipeline. The evaluation compares the **Pointwise Stage** (BioBERT + Custom Projection/Matching Checkpoint) against the **Listwise Reranking Stage** (Qwen-2.5-7B-Instruct running Zero-shot).

The run evaluated **466 papers** before the execution session timed out.

---

## 1. Evaluation Metrics Summary

The table below shows the performance of the Pointwise and Listwise stages evaluated across the 466 completed paper samples:

| Evaluation Metric | Pointwise Stage (BioBERT + Checkpoint) | Listwise Stage (Qwen Reranked) | Delta (Listwise vs Pointwise) |
| :--- | :---: | :---: | :---: |
| **Accuracy@1** | **0.4099** (40.99%) | 0.1652 (16.52%) | ⬇️ -0.2447 |
| **Accuracy@5** | **0.6288** (62.88%) | 0.5923 (59.23%) | ⬇️ -0.0365 |
| **Accuracy@10** | **0.7253** (72.53%) | **0.7253** (72.53%) | ➖ 0.0000 (Same candidate pool) |
| **MRR** (Mean Reciprocal Rank) | **0.5063** | 0.3319 | ⬇️ -0.1744 |
| **NDCG@5** | **0.5271** | 0.3824 | ⬇️ -0.1447 |
| **NDCG@10** | **0.5584** | 0.4261 | ⬇️ -0.1323 |

---

## 2. Key Findings & Analysis

During this evaluation, we observed a significant drop in ranking performance when applying the Listwise reranker (Qwen-2.5-7B-Instruct) compared to the first-stage Pointwise retriever (BioBERT). Below is a detailed analysis of this phenomenon:

### A. Supervised vs. Zero-Shot Capability
* **Pointwise Stage (BioBERT)**: The BioBERT model (specifically `dmis-lab/biobert-v1.1`) combined with custom projection heads (`linear1_1` and `linear2_1`) was **explicitly fine-tuned via contrastive learning** on the MedPRS training dataset. It learned the domain-specific associations between paper text (titles and abstracts) and specific journal aims.
* **Listwise Stage (Qwen)**: The LLM (`Qwen/Qwen2.5-7B-Instruct`) was evaluated in a **Zero-Shot** setting. It has no prior knowledge of the target training distribution, nor has it been fine-tuned for this specific recommendation task.

### B. Popularity & Generalization Bias of LLMs
Large Language Models (LLMs) often exhibit popularity bias, favoring general-interest or highly visible journals over niche specialty journals.
* *Example (Paper 464)*:
  * **Ground Truth**: `Behavior Research Methods` (A niche, specialized journal).
  * **Pointwise (BioBERT)** ranked it **1st** correctly.
  * **Listwise (Qwen)** pushed it down to **3rd**, ranking two broader journals higher:
    1. *Cognitive Research: Principles and Implications*
    2. *Biostatistics*
    3. *Behavior Research Methods*
  * This highlights that Qwen's general pre-training knowledge can override the precise domain-specific matching learned by the pointwise model.

### C. Overlapping Aims & Scope Descriptions
Many biomedical journals share highly similar "Aims & Scope" texts (e.g., repeating generic phrases like *"clinical trials"*, *"cellular mechanisms"*, or *"translational medicine"*). Without task-specific training, a Zero-shot LLM struggles to distinguish the fine-grained differences between these journals, leading to rank degradation.

---

## 3. Recommended Strategies for Improvement

To bridge the performance gap and leverage the reasoning capabilities of the Listwise reranker, the following methods are recommended:

1. **Few-Shot Prompting (In-Context Learning)**:
   Incorporate 2-3 illustrative examples of papers and their expert-assigned ground truth journals in the system prompt. This guides the LLM to understand the criteria and selection style of the MedPRS dataset.
2. **Methodological & Granularity Rules**:
   Add specific constraints to the system prompt, instructing the model to prioritize specialized journals (e.g., *"If a paper is heavily method-focused, prioritize methodological journals like Behavior Research Methods over general cognitive journals"*).
3. **Task-Specific Reranker Fine-Tuning**:
   To replicate the paper's original gains, perform Parameter-Efficient Fine-Tuning (PEFT/LoRA) on the Qwen model using training triples (Query, Relevant Journal, Irrelevant Journals) from the MedPRS training set.
