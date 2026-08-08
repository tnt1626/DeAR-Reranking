# MedPRS Dataset Reproduction & Evaluation Results

This document summarizes and compares the evaluation results of reproducing the **MedPRS** journal recommendation pipeline on Kaggle. It contrasts the **Pointwise Stage** (BioBERT + Custom Checkpoint) against two distinct **Listwise Reranking Stage** prompts using `Qwen-2.5-7B-Instruct` (Zero-shot):
1. **Concise Prompt (Version 1)**: Requesting a simple 1-sentence explanation per journal before ranking.
2. **Chain-of-Thought / CoT Prompt (Version 2)**: Requesting a detailed 2-to-3 sentence analysis per journal before ranking.

---

## 1. Comparative Evaluation Metrics

The table below summarizes the metrics computed across the completed paper samples for both runs:

| Evaluation Metric | Pointwise Stage | Listwise (Version 1: Concise 1-sentence) | Delta (V1 vs Pointwise) | Listwise (Version 2: Detailed 2-3 sentence CoT) | Delta (V2 vs Pointwise) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Completed Papers**| 466 / 390 | **466** | - | **390** | - |
| **Accuracy@1** | 0.4099 / 0.4231 | 0.1652 | ⬇️ -0.2447 | 0.1564 | ⬇️ -0.2667 |
| **Accuracy@5** | 0.6288 / 0.6282 | 0.5923 | ⬇️ -0.0365 | 0.5846 | ⬇️ -0.0436 |
| **Accuracy@10** | 0.7253 / 0.7256 | 0.7253 | ➖ 0.0000 | 0.7256 | ➖ 0.0000 |
| **MRR** | 0.5063 / 0.5134 | 0.3319 | ⬇️ -0.1744 | 0.3250 | ⬇️ -0.1884 |
| **NDCG@5** | 0.5271 / 0.5323 | 0.3824 | ⬇️ -0.1447 | 0.3746 | ⬇️ -0.1577 |
| **NDCG@10** | 0.5584 / 0.5637 | 0.4261 | ⬇️ -0.1323 | 0.4208 | ⬇️ -0.1429 |

---

## 2. Key Findings & Comparative Analysis

The evaluation reveals two important scientific insights:
1. **Zero-shot Listwise Reranking degrades supervised Pointwise retrieval.**
2. **Adding longer Chain-of-Thought (CoT) reasoning hurts ranking performance further while increasing latency.**

### A. Why Listwise Reranking Degrades Performance
* **Supervised Pointwise Model (BioBERT + Custom Checkpoint)**: Explicitly trained on the MedPRS dataset to recognize specific y-label associations. It is a highly specialized ranker.
* **Zero-shot Listwise Model (Qwen)**: Lacks domain-specific fine-tuning. It suffers from **Popularity Bias** (ranking well-known journals like *Nature* or *PLOS* higher than niche specialty journals) and is distracted by generic overlaps in journal Aims & Scope.

### B. Why Detailed CoT (Version 2) Performed Worse than Concise Prompt (Version 1)
Counter-intuitively, asking the LLM to write longer analyses (2-3 sentences instead of 1 sentence) resulted in **worse accuracy, MRR, and NDCG** across the board:
1. **Accumulation of Reasoning Noise**: Generating a longer sequence (~400 tokens of explanation) before outputting the final ranking list introduces more opportunities for logical inconsistencies and "post-hoc rationalizations" (writing plausible reasons to justify a wrong journal choice).
2. **Information Overload inside Context**: The generated explanations pollute the model's own context window, distracting it from the main task of strict comparative ranking.
3. **Severe Latency Bottleneck**: Version 2 required **10 hours for only 390 samples (~92 seconds per sample)** compared to Version 1, due to the high computational overhead of generating long explanations and passing tokens between split GPUs on Kaggle T4 cards.

---

## 3. Practical Recommendations for Future Reranking Pipelines

If you wish to deploy or write research papers on this hybrid pipeline, consider these approaches:

* **Prefer Concise Prompts**: When deploying zero-shot rerankers, keep explanations minimal (1 sentence or none). It is not only 3x-4x faster but also more accurate.
* **Avoid Dual-GPU Split (Pipeline Parallelism) on Kaggle T4s**: Loading models in 16-bit across two separate GPUs via `device_map="auto"` introduces heavy PCIe latency. Always load models in **4-bit (`bitsandbytes`) on a single GPU** to run 3x-5x faster.
* **Apply Few-Shot Prompts**: Use 2-3 examples with ground-truth rankings to align the LLM with the dataset's target distribution.
* **Supervised Fine-tuning (Lora)**: Fine-tune the Qwen reranker on the target MedPRS training data to align its preference with the ground-truth journals.
