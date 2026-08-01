# Kaggle Run Guide for DeAR-Reranking

This guide explains the code modifications implemented to make this repository compatible with Kaggle's Multi-GPU (T4 x2) environment and provides step-by-step instructions for running evaluations.

---

## 1. Summary of Changes

To fix hardware-related bottlenecks and library dependency errors on Kaggle, the following modifications have been applied:

1. **Java-Free & Pyserini-Free Evaluation Setup:**
   - **Created [eval_trec.py](eval_trec.py):** A pure Python script using the lightweight `pytrec_eval` library to evaluate NDCG@10 metrics. This replaces `python -m pyserini.eval.trec_eval` which requires a complex Java JDK setup that frequently fails in Kaggle containers.
   - **Automated Qrels Download:** The script downloads ground-truth `qrels` files directly from Castorini/Anserini's GitHub repository over HTTPS and caches them locally. It also includes an automatic fallback resolver to handle naming differences (e.g., `-test.txt` vs `.test.txt` in BEIR datasets like COVID).
   - **Local File Parsing in [convert_trec_to_json.py](pointwise_reranker/convert_trec_to_json.py):** Rewrote the TREC-to-JSON format converter to retrieve query and document texts directly from the source BM25 JSONL datasets, removing Lucene `IndexReader` and `get_topics` imports from Pyserini.

2. **Multi-GPU Support & VRAM Optimizations:**
   - **Enabled `device_map="auto"`:** Allows Hugging Face to automatically distribute model layers across both T4 GPUs.
   - **4-Bit Listwise Quantization in [clients.py](listwise_rerank/clients.py):** Configured `load_in_4bit=True` to load the Llama-3.1-8B-Instruct base model in 4-bit precision. This reduces the weight footprint from 16GB to **5.5GB**, leaving enough free VRAM on the GPUs for the large KV cache and activation allocations required during long sequence listwise generation (7000+ tokens), avoiding **CUDA Out of Memory** errors.
   - **Guarded PEFT Model Merging in [modeling.py](pointwise_reranker/modeling.py):** Bounded `student.merge_and_unload()` inside a `try-except` block to prevent crashes caused by merging LoRA weights on multi-device/pipeline parallel model instances.

3. **Modern API Compatibilities:**
   - **Fixed Gated datasets Loading in [data.py](pointwise_reranker/data.py):** Replaced the deprecated `use_auth_token=None` argument with `token=None` to ensure compatibility with newer versions of the Hugging Face `datasets` library pre-installed on Kaggle.

---

## 2. Running on Kaggle (Step-by-Step)

### Step 1: Environment Setup & Hugging Face Authentication
Both Llama-3.1 models require Hugging Face Hub authentication.
1. Go to **Add-ons -> Secrets** in the Kaggle Notebook editor.
2. Add a new secret with the label `HF_TOKEN`, fill in your Hugging Face Access Token (Read permission), and toggle **Access** to enable it in the notebook.
3. Run the following setup cell to copy the code, install dependencies, download BM25 data, and log in:
   ```python
   # 1. Copy the code files from the dataset to the working directory (replace <your-kaggle-username> accordingly)
   !cp -r /kaggle/input/datasets/<your-kaggle-username>/dear-reranking-code/* /kaggle/working/
   %cd /kaggle/working
   
   # 2. Install custom dependencies (PyTorch and major libraries are pre-installed)
   !pip install -r requirements.txt
   !pip install bitsandbytes
   
   # 3. Download and extract the BM25 jsonl datasets
   !wget -L 'https://www.dropbox.com/scl/fi/2ryyzvht45fazrjjuetgk/bm25_beir_dl19_20.zip?rlkey=e3li5e26n12iuq2zrp61ti5tq&st=xnic6wvp&dl=1' \
      -O bm25_beir_dl19_20.zip
   !unzip bm25_beir_dl19_20.zip -d data
   
   # 4. Log in to Hugging Face
   from kaggle_secrets import UserSecretsClient
   from huggingface_hub import login
   user_secrets = UserSecretsClient()
   hf_token = user_secrets.get_secret("HF_TOKEN")
   login(token=hf_token)
   ```

### Step 2: Run Pointwise Reranking
Execute the pointwise script to perform inference and evaluation:
```python
!bash run_pointwise.sh
```

### Step 3: Run Listwise Reranking using Pointwise Cached Output
To run listwise reranking without re-running the 16-minute pointwise inference:

1. Click **Edit** on your notebook, then click **"+ Add Input"** on the right sidebar.
2. Select **Notebook Outputs**, find this notebook (`DeAR: Dual-Stage Document Reranking with Reasoning`), and click **Add** on your pointwise-completed version (Version 1).
3. Create a code cell to copy the pointwise output folder to your current workspace:
   ```python
   !mkdir -p /kaggle/working/results
   !cp -r /kaggle/input/dear-dual-stage-document-reranking-with-reasoning/results/* /kaggle/working/results/
   ```
4. Run the listwise script:
   ```python
   !bash run_listwise.sh
   ```

### Step 4: Run in the Background (Unattended Commit)
1. Click **Save Version** in the top right.
2. Choose **"Save & Run All (Commit)"** as the Version Type and click **Save**.
3. You can safely close your browser or put your computer to sleep. The notebook will run completely in the cloud. Check the **Versions** menu later to download outputs and logs.
