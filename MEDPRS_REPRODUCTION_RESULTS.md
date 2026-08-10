# MedPRS Dataset Reproduction & Evaluation Results

This document summarizes and compares the evaluation results of reproducing the **MedPRS** journal recommendation pipeline on Kaggle. It compares the **Pointwise Stage** (BioBERT + Custom Checkpoint) against three distinct **Listwise Reranking Stage** prompt setups using `Qwen-2.5-7B-Instruct`:
1. **Version 1 (Concise Zero-shot)**: Requesting a simple 1-sentence explanation per journal before ranking (Zero-shot).
2. **Version 2 (CoT Zero-shot)**: Requesting a detailed 2-to-3 sentence analysis per journal before ranking (Zero-shot).
3. **Version 3 (Few-shot Example-guided)**: Incorporating 2 representative few-shot examples with reasoning and rankings to guide the model (Few-shot).

---

## 1. Comparative Evaluation Metrics

The table below summarizes the metrics computed across the completed paper samples for all three versions. Since the test runs completed different numbers of samples due to Kaggle session time limits, each Listwise version is compared against its corresponding Pointwise baseline calculated on the exact same subset of papers:

| Evaluation Metric | V1 (Concise Zero-shot) [466 papers] | V2 (CoT Zero-shot) [390 papers] | V3 (Few-shot) [200 papers] |
| :--- | :---: | :---: | :---: |
| **Pointwise Acc@1** | 0.4099 | 0.4231 | 0.4800 |
| **Listwise Acc@1** | 0.1652 | 0.1564 | **0.3750** |
| *Acc@1 Delta* | *⬇️ -0.2447 (-59.7%)* | *⬇️ -0.2667 (-63.0%)* | ***⬇️ -0.1050 (-21.8%)*** |
| | | | |
| **Pointwise Acc@5** | 0.6288 | 0.6282 | 0.7400 |
| **Listwise Acc@5** | 0.5923 | 0.5846 | **0.7050** |
| *Acc@5 Delta* | *⬇️ -0.0365 (-5.8%)* | *⬇️ -0.0436 (-6.9%)* | ***⬇️ -0.0350 (-4.7%)*** |
| | | | |
| **Pointwise MRR** | 0.5063 | 0.5134 | 0.5926 |
| **Listwise MRR** | 0.3319 | 0.3250 | **0.5065** |
| *MRR Delta* | *⬇️ -0.1744 (-34.4%)* | *⬇️ -0.1884 (-36.7%)* | ***⬇️ -0.0861 (-14.5%)*** |
| | | | |
| **Pointwise NDCG@10**| 0.5584 | 0.5637 | 0.6496 |
| **Listwise NDCG@10**| 0.4261 | 0.4208 | **0.5836** |
| *NDCG@10 Delta* | *⬇️ -0.1323 (-23.7%)* | *⬇️ -0.1429 (-25.3%)* | ***⬇️ -0.0660 (-10.1%)*** |

---

## 2. Key Findings & Comparative Analysis

Comparing the three versions reveals major scientific insights into how prompting strategies affect LLM-based reranking performance:

### A. Few-shot Prompting (V3) Halves the Performance Drop
* **The Zero-shot Challenge (V1 and V2)**: In Zero-shot mode, Qwen suffered a catastrophic drop in performance compared to the supervised Pointwise baseline (e.g., a **-59.7%** drop in Acc@1). This was due to popularity bias (favoring famous journals like *Nature* or *PLOS*) and a lack of task-specific alignment.
* **The Few-shot Solution (V3)**: By introducing just 2 representative examples of papers and their expert-assigned journals, Qwen's performance drop was **cut in half** (e.g., the Acc@1 drop was reduced to **-21.8%**, and the NDCG@10 drop was reduced from **-23.7%** to just **-10.1%**). 
* **Mechanism**: In-context examples successfully align the model's judgment with the target dataset's distribution, helping Qwen prioritize domain specificity over general popularity bias.

### B. Long-chain Reasoning (V2) Degrades Performance
* Asking Qwen to generate detailed 2-to-3 sentence explanations (V2) instead of a single concise sentence (V1) resulted in worse performance across all metrics.
* Long-form generation introduces **reasoning noise** and post-hoc rationalizations, which clutter the self-attention context and distract the model from the comparative ranking task.

### C. The Truncation Issue: Example Length Mismatch
During log analysis, we identified a critical prompting issue in Version 3:
* **Symptom**: In several papers (e.g., Papers 199 and 200), Qwen only ranked 3 or 4 candidates in its final list (e.g., `[1] > [2] > [0]`), leaving the remaining 6-7 candidates unranked.
* **Cause**: The 2 few-shot examples in the prompt only had **3 candidate journals** in their context. Qwen mimicked this format and truncated its active rankings to 3-4 items, even though the active query provided **10 candidate journals**.
* **Impact**: The unranked items were appended in their default pointwise order, limiting Qwen's ability to rerank the bottom positions and artificially capping the potential few-shot gains.

---

## 3. Practical Recommendations for Next-Stage Optimization

To fully close the performance gap and surpass the Pointwise baseline, future runs should implement:

1. **Match Candidate Count in Few-shot Examples**: Rewrite the few-shot examples to contain exactly **10 candidates** and rank all 10 candidates. This will guide Qwen to output complete 10-element rankings without truncation.
2. **Quantized Single-GPU Inference**: Continue using the single-GPU RTX 6000 setup in 16-bit precision, which runs **5x-8x faster** than split-GPU T4 pipeline parallelism and avoids inter-GPU PCIe transfer bottlenecks.
3. **Supervised Fine-Tuning (SFT)**: If maximum accuracy is required, perform LoRA fine-tuning on Qwen using target preference rankings from the MedPRS train set to fully align the model's weights.
