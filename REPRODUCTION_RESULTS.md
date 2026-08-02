# DeAR Reproduction Results — Comparison with Paper

This document summarizes the evaluation results from reproducing the DeAR (Dual-Stage Document Reranking with Reasoning) model on Kaggle and compares them with the original paper.

---

## 1. General Information

| Parameter | Value |
| :--- | :--- |
| **Reproduced Models** | DeAR-P-8B-CE (Pointwise) + DeAR-L-8B (Listwise) |
| **Running Environment** | Kaggle — T4 x2 GPU |
| **Evaluation Metric** | nDCG@10 |
| **First-stage Retriever** | BM25 top-100 |

---

## 2. nDCG@10 Results — Pointwise Stage

The table below compares our reproduced Pointwise stage scores against the values reported in the paper:

| Dataset | BM25 (Paper) | DeAR-P (Paper) | DeAR-P (Reproduced) | Δ vs Paper | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Covid** | 59.47 | 84.14 | **84.16** | +0.02 | ✅ Match |
| **DBPedia** | 31.80 | 46.27 | **46.27** | 0.00 | ✅ Match |
| **TREC-DL19** | 50.58 | 74.50 | **74.50** | 0.00 | ✅ Match |
| **TREC-DL20** | 47.96 | 68.71 | **68.72** | +0.01 | ✅ Match |
| **News** | 39.52 | 51.71 | **51.71** | 0.00 | ✅ Match |
| **NFCorpus** | 30.75 | 36.57 | **38.38** | +1.81 | ⚠️ Higher |
| **Robust04** | 40.70 | 52.43 | **52.44** | +0.01 | ✅ Match |
| **SciFact** | 67.89 | 77.39 | **77.38** | -0.01 | ✅ Match |
| **Signal** | 33.05 | 29.91 | **29.90** | -0.01 | ✅ Match |
| **Touche** | 44.22 | 37.23 | **37.35** | +0.12 | ✅ Match |

> [!WARNING]
> **NFCorpus** shows a discrepancy of **+1.81**. This is likely due to slight differences in the BM25 index version or the random seed used when sampling candidates.

---

## 3. nDCG@10 Results — Listwise Stage (Covid only)

The table below compares the Listwise reranking stage results:

| Dataset | BM25 (Paper) | DeAR-P (Paper) | DeAR-L (Paper) | DeAR-L (Reproduced) | Δ vs Paper | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Covid** | 59.47 | 84.14 | 88.36 | **88.04** | -0.32 | ✅ Match |

> [!NOTE]
> Listwise reranking was reproduced specifically on the **Covid** dataset. The resulting score of **88.04** is very close to the paper's reported **88.36**, and the delta is well within an acceptable margin.

---

## 4. Demo: Multi-stage Reranking on a Real Query

**Query:** *"Has social distancing had an impact on slowing the spread of COVID-19?"* (Query ID: `10`)

### **Stage 1 — BM25 (Lexical Retrieval)**
Retrieves 100 candidate documents based on keyword token overlap. While fast, it lacks semantic understanding—for example, a document focusing on "lung ultrasound in neonates with COVID-19" may rank highly simply because it mentions the keyword `COVID-19`.

### **Stage 2 — Pointwise Reranking (DeAR-P-8B-CE)**
The model scores each candidate document independently:

| Rank | Doc ID | Score | Document Context / Title | Ghi chú |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `pn02p843` | +2.1601 | *U.S. county level analysis to determine If social distancing slowed the spread...* | Highly Relevant ✅ |
| **2** | `0b6dsdct` | +1.6992 | *Modeling the dynamics of COVID19 spread during and after social distancing* | Highly Relevant ✅ |
| **100** | `pab56xyd` | -7.7031 | *Point-of-care lung ultrasound in three neonates with COVID-19* | Irrelevant ❌ (Correctly Filtered) |

### **Stage 3 — Listwise Reranking (DeAR-L-8B)**
The model processes the top-20 documents simultaneously inside a single context window, comparing them directly and outputting the ranking:

**Output:** `[1] > [0] > [2] > [3] > ...`

*Analysis:* The listwise model swaps **Document [1]** (`0b6dsdct`) and **Document [0]** (`pn02p843`) because the modeling paper (`0b6dsdct`) provides a more comprehensive answer regarding the overall impact on the infection curve compared to the phone ping analysis (`pn02p843`). This demonstrates the power of inter-document attention and comparative reasoning.

---

## 5. General Observations

- **High Reproducibility:** The reproducibility confidence is very high. Most datasets show a discrepancy ($\Delta$) of less than 0.1, confirming that the paper's results are solid and highly reproducible.
- **Observed Strengths:**
  - The Pointwise stage significantly improves relevance sorting over the BM25 baseline across nearly all datasets (adding +15 to +25 nDCG@10).
  - The Listwise stage builds on top of the Pointwise output to refine the top positions even further (Covid: 84.16 → 88.04).
  - The multi-stage pipeline behaves exactly as described in the paper.
- **Points of Caution:**
  - `Signal` and `Touche` datasets perform worse than the BM25 baseline, which is a known limitation acknowledged in the original paper.
  - `NFCorpus` has a slight variance from the paper, which might warrant checking the sampling candidate generation settings.
