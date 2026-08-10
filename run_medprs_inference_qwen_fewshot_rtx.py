import os
import re
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

class SimCPSRModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.linear1_1 = None
        self.linear2_1 = None
        self.linear_main_1 = None
        
    def init_heads(self, state_dict):
        for k in state_dict.keys():
            if "linear1_1.weight" in k:
                w = state_dict[k]
                self.linear1_1 = nn.Linear(w.shape[1], w.shape[0])
                print(f"Initialized linear1_1: nn.Linear({w.shape[1]}, {w.shape[0]})")
            if "linear2_1.weight" in k:
                w = state_dict[k]
                self.linear2_1 = nn.Linear(w.shape[1], w.shape[0])
                print(f"Initialized linear2_1: nn.Linear({w.shape[1]}, {w.shape[0]})")
            if "linear_main_1.weight" in k:
                w = state_dict[k]
                self.linear_main_1 = nn.Linear(w.shape[1], w.shape[0])
                print(f"Initialized linear_main_1: nn.Linear({w.shape[1]}, {w.shape[0]})")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_journal(self, journal_input_ids, journal_attention_mask):
        outputs = self.base_model(input_ids=journal_input_ids, attention_mask=journal_attention_mask)
        emb = self._mean_pooling(outputs, journal_attention_mask)
        if self.linear2_1 is not None:
            emb = self.linear2_1(emb)
            emb = torch.relu(emb)
        return emb

    def encode_paper(self, paper_input_ids, paper_attention_mask):
        outputs = self.base_model(input_ids=paper_input_ids, attention_mask=paper_attention_mask)
        emb = self._mean_pooling(outputs, paper_attention_mask)
        if self.linear1_1 is not None:
            emb = self.linear1_1(emb)
            emb = torch.relu(emb)
        return emb

    def forward(self, paper_proj_embeddings, journal_proj_embeddings):
        # 1. Normalize representations to unit length for Cosine Similarity
        paper_proj_norm = paper_proj_embeddings / torch.clamp(paper_proj_embeddings.norm(dim=-1, keepdim=True), min=1e-9)
        journal_proj_norm = journal_proj_embeddings / torch.clamp(journal_proj_embeddings.norm(dim=-1, keepdim=True), min=1e-9)
        
        # 2. Compute similarity vector [batch_size, 1406]
        sim_vector = torch.matmul(paper_proj_norm, journal_proj_norm.t())
        
        # 3. Concatenate unnormalized paper representation with similarity score vector
        joint = torch.cat([paper_proj_embeddings, sim_vector], dim=-1)
        
        # 4. Final classification scores over journals
        logits = self.linear_main_1(joint)
        return logits

def calculate_ndcg_at_k(candidates, correct_label, k):
    for i in range(min(len(candidates), k)):
        if candidates[i] == correct_label:
            return 1.0 / math.log2(i + 2)
    return 0.0

def calculate_mrr(candidates, correct_label):
    for i in range(len(candidates)):
        if candidates[i] == correct_label:
            return 1.0 / (i + 1)
    return 0.0

def parse_listwise_output(output_text, num_candidates):
    ranking_line = output_text
    for line in reversed(output_text.split('\n')):
        if '>' in line:
            ranking_line = line
            break
            
    pattern = re.compile(r'\[(\d+)\]')
    ranks = [int(x) for x in pattern.findall(ranking_line)]
    
    valid_ranks = []
    for r in ranks:
        if 0 <= r < num_candidates and r not in valid_ranks:
            valid_ranks.append(r)
            
    for r in range(num_candidates):
        if r not in valid_ranks:
            valid_ranks.append(r)
            
    return valid_ranks

def main():
    print("=== MEDPRS JOURNAL RECOMMENDATION - FEW-SHOT PIPELINE (RTX 6000 SINGLE-GPU) ===")
    
    # Set seed for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    import argparse
    parser = argparse.ArgumentParser(description="MedPRS few-shot journal recommendation on single RTX 6000.")
    parser.add_argument("--num_papers", type=int, default=10, help="Number of papers to evaluate.")
    args, unknown = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Data
    journal_path = "data_MedPRS/journal_category.csv"
    val_path = "data_MedPRS/val_set.csv"
    checkpoint_path = "Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth"
    
    if not os.path.exists(journal_path) or not os.path.exists(val_path):
        print(f"Error: Could not find data files under data_MedPRS/. Please check paths.")
        sys.exit(1)
        
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find checkpoint file {checkpoint_path} in workspace.")
        sys.exit(1)
        
    print("Loading datasets...")
    journal_df = pd.read_csv(journal_path)
    val_df = pd.read_csv(val_path)
    
    journal_df = journal_df.fillna("")
    val_df = val_df.fillna("")
    
    # Create label mapping
    label_to_journal = {}
    for idx, row in journal_df.iterrows():
        try:
            lbl = int(row['Label'])
            label_to_journal[lbl] = {
                "name": row['Journal'],
                "aims": row['Aims'],
                "categories": row['Categories']
            }
        except Exception:
            pass
            
    print(f"Loaded {len(journal_df)} journals and {len(val_df)} validation papers.")
    
    # 2. Load BioBERT Model and Checkpoint
    print("Loading BioBERT tokenizer and base model...")
    # Offline model path options (uncomment the one that matches your environment)
    biobert_model_name = "biobert-v1.1" # Local offline folder in DeAR-Reranking/
    # biobert_model_name = "/kaggle/input/biobert-v1-1-offline" # Kaggle offline dataset path
    # biobert_model_name = "dmis-lab/biobert-v1.1" # Online Hugging Face path
    
    biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
    biobert_base = AutoModel.from_pretrained(biobert_model_name)
    
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    model = SimCPSRModel(biobert_base)
    model.init_heads(state_dict)
    
    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("model_state_dict."):
            name = name[17:]
        if name.startswith("model."):
            name = name[6:]
        if name.startswith("base_model.bert."):
            name = "base_model." + name[16:]
        clean_state_dict[name] = v
        
    info = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Successfully loaded checkpoint weights. Missing keys: {len(info.missing_keys)}")
    
    model.to(device)
    model.eval()
    
    is_multi_class = False
    num_classes = 0
    if model.linear_main_1 is not None:
        num_classes = model.linear_main_1.out_features
        if num_classes > 1000:
            is_multi_class = True
            
    print(f"Model Mode: {'Multi-Class Classification' if is_multi_class else 'Bi-Encoder Similarity'}")
    
    # 3. Compute Projected Embeddings for all 1,406 journals in memory
    journal_proj_embeddings = None
    if is_multi_class:
        print("Computing projected embeddings for all 1,406 journals...")
        journal_proj_embeddings = torch.zeros((num_classes, 512), device=device)
        for idx, row in tqdm(journal_df.iterrows(), total=len(journal_df)):
            name = row['Journal']
            aims = row['Aims']
            cats = row['Categories']
            lbl = int(row['Label'])
            
            text = f"Journal Name: {name}\nAims and Scope: {aims}\nCategories: {cats}"
            
            inputs = biobert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                emb = model.encode_journal(inputs['input_ids'], inputs['attention_mask'])
                
            if 0 <= lbl < num_classes:
                journal_proj_embeddings[lbl] = emb[0]
        print("Journal projected embeddings computed successfully.")
    else:
        print("Computing embeddings for all 1,408 journals in memory...")
        journal_embeddings_list = []
        for idx, row in tqdm(journal_df.iterrows(), total=len(journal_df)):
            name = row['Journal']
            aims = row['Aims']
            cats = row['Categories']
            text = f"Journal Name: {name}\nAims and Scope: {aims}\nCategories: {cats}"
            
            inputs = biobert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                emb = model(inputs['input_ids'], inputs['attention_mask'])
            journal_embeddings_list.append(emb.cpu().numpy()[0])
        journal_embeddings = np.array(journal_embeddings_list)
        print("Journal embeddings computed successfully.")
            
    # 4. Load Qwen 2.5 7B Instruct Model in Native 16-bit (RTX 6000 supports bfloat16 directly)
    print("Loading Qwen-2.5-7B-Instruct Model in native bfloat16...")
    # Offline model path options (uncomment the one that matches your environment)
    qwen_repo = "Qwen/Qwen2.5-7B-Instruct" # Online Hugging Face path
    # qwen_repo = "/kaggle/input/qwen-2.5-7b-instruct/transformers/default/1" # Kaggle offline model path
    
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_repo, use_fast=True)
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load on a single GPU (since RTX 6000 has 24/48GB VRAM)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_repo,
        torch_dtype=dtype,
        device_map="cuda:0", # Pin to first GPU to avoid PCIe transfer delay
        trust_remote_code=True
    )
    
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        
    qwen_model.eval()
    print("Qwen 2.5 7B Instruct loaded successfully on cuda:0.")
    
    # 5. Run Evaluation on a Subset of Validation Papers
    num_eval_papers = args.num_papers
    print(f"\nEvaluating the first {num_eval_papers} papers...")
    
    pointwise_top1_hits = 0
    pointwise_top5_hits = 0
    pointwise_top10_hits = 0
    pointwise_mrr_sum = 0.0
    pointwise_ndcg5_sum = 0.0
    pointwise_ndcg10_sum = 0.0
    
    listwise_top1_hits = 0
    listwise_top5_hits = 0
    listwise_top10_hits = 0
    listwise_mrr_sum = 0.0
    listwise_ndcg5_sum = 0.0
    listwise_ndcg10_sum = 0.0
    
    results_log = []
    
    for i in range(num_eval_papers):
        row = val_df.iloc[i]
        title = row['Title']
        abstract = row['Abstract']
        keywords = row['Keywords']
        correct_label = int(row['Label'])
        
        correct_journal_info = label_to_journal.get(correct_label, {"name": "Unknown", "aims": "", "categories": ""})
        correct_journal_name = correct_journal_info["name"]
        
        paper_text = f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords}"
        
        # Calculate pointwise candidate scores
        if is_multi_class:
            inputs = biobert_tokenizer(paper_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                paper_emb = model.encode_paper(inputs['input_ids'], inputs['attention_mask'])
                logits = model(paper_emb, journal_proj_embeddings)
            logits = logits.cpu().numpy()[0]
            
            scores = []
            for idx, r in journal_df.iterrows():
                try:
                    lbl = int(r['Label'])
                    if 0 <= lbl < len(logits):
                        scores.append(logits[lbl])
                    else:
                        scores.append(-9999.0)
                except Exception:
                    scores.append(-9999.0)
        else:
            inputs = biobert_tokenizer(paper_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                paper_emb = model(inputs['input_ids'], inputs['attention_mask']).cpu().numpy()[0]
            scores = []
            for j_emb in journal_embeddings:
                sim = np.dot(paper_emb, j_emb) / (np.linalg.norm(paper_emb) * np.linalg.norm(j_emb))
                scores.append(sim)
                
        # Get Top 10 candidate journals
        top10_indices = np.argsort(scores)[::-1][:10]
        
        pointwise_candidates = []
        for rank_idx, idx in enumerate(top10_indices):
            j_row = journal_df.iloc[idx]
            pointwise_candidates.append({
                "index": rank_idx,
                "label": int(j_row['Label']),
                "name": j_row['Journal'],
                "aims": j_row['Aims'],
                "categories": j_row['Categories'],
                "score": float(scores[idx])
            })
            
        pointwise_labels = [c['label'] for c in pointwise_candidates]
        if correct_label == pointwise_labels[0]:
            pointwise_top1_hits += 1
        if correct_label in pointwise_labels[:5]:
            pointwise_top5_hits += 1
        if correct_label in pointwise_labels[:10]:
            pointwise_top10_hits += 1
        pointwise_mrr_sum += calculate_mrr(pointwise_labels, correct_label)
        pointwise_ndcg5_sum += calculate_ndcg_at_k(pointwise_labels, correct_label, 5)
        pointwise_ndcg10_sum += calculate_ndcg_at_k(pointwise_labels, correct_label, 10)
            
        # 6. Construct FEW-SHOT Listwise Prompt for Qwen
        SYSTEM_PROMPT = (
            "You are RankLLM, an expert academic editor specializing in recommending publishing venues. "
            "You must carefully evaluate candidate journals based on their Aims & Scope, Categories, and suitability for the provided research paper. "
            "Explain your reasoning in 2-3 sentences first, and then write the final ranking at the very end of your response."
        )
        
        # Build the few-shot context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            
            # --- FEW-SHOT EXAMPLE 1 ---
            {"role": "user", "content": "I will provide you with 3 candidate journals, each indicated by number identifier [].\n"
                                     "Rank the journals based on their suitability and domain match for publishing the research paper:\n"
                                     "Title: Deep learning for detection of covid-19 using chest x-ray images\n"
                                     "Abstract: The outbreak of COVID-19 has caused global panic. In this work, we propose a deep convolutional neural network model to detect COVID-19 cases from chest X-ray radiographs..."},
            {"role": "assistant", "content": "Okay, please provide the journals."},
            {"role": "user", "content": "[0] Journal: Journal of Medical Imaging and Health Informatics\nAims: Focuses on health informatics and general medical databases.\nCategories: Health Informatics, Computer Science.\n"
                                     "[1] Journal: Computers in Biology and Medicine\nAims: Broad application of computers in bioscience and medicine.\nCategories: Biophysics, Biomedical Engineering.\n"
                                     "[2] Journal: Medical Image Analysis\nAims: Specializes in processing and analyzing medical images using deep learning.\nCategories: Radiology, Computer Vision."},
            {"role": "assistant", "content": "Analyses:\n"
                                          "- **[0] Journal of Medical Imaging and Health Informatics**: Covers general health databases and informatics, but is less focused on deep learning for computer vision. Not suitable.\n"
                                          "- **[1] Computers in Biology and Medicine**: Covers general computing in medicine, which could fit, but lacks the specific radiology and image analysis focus. Less suitable.\n"
                                          "- **[2] Medical Image Analysis**: This journal specifically targets the processing and analysis of medical images (X-rays) using machine learning methods, matching the paper's core methodology perfectly. Highly suitable.\n\n"
                                          "Final Ranking List:\n"
                                          "[2] > [1] > [0]"},
                                          
            # --- FEW-SHOT EXAMPLE 2 ---
            {"role": "user", "content": "I will provide you with 3 candidate journals, each indicated by number identifier [].\n"
                                     "Rank the journals based on their suitability and domain match for publishing the research paper:\n"
                                     "Title: A state response measurement model for problem solving process data\n"
                                     "Abstract: This paper proposes a measurement model to analyze process data in problem-solving tasks. We use a Bayesian approach to estimate parameters from behavioral logs..."},
            {"role": "assistant", "content": "Okay, please provide the journals."},
            {"role": "user", "content": "[0] Journal: Cognitive Research: Principles and Implications\nAims: Covers cognitive psychology and behavioral studies.\nCategories: Experimental Psychology.\n"
                                     "[1] Journal: Psychometrika\nAims: Publishes statistical and mathematical methods in psychology and behavioral sciences.\nCategories: Statistics, Psychometrics.\n"
                                     "[2] Journal: Behavior Research Methods\nAims: Focuses on methodology and computer technology in psychological research.\nCategories: Experimental Psychology, Computer Science."},
            {"role": "assistant", "content": "Analyses:\n"
                                          "- **[0] Cognitive Research: Principles and Implications**: Focuses on cognitive principles, but is not primarily a statistics or measurement model journal. Less suitable.\n"
                                          "- **[1] Psychometrika**: Directly targets statistical, mathematical, and psychometric models for behavioral data, aligning perfectly with the proposed measurement model. Highly suitable.\n"
                                          "- **[2] Behavior Research Methods**: Focuses on software and methods for behavioral research, which is relevant but less aligned with the mathematical modeling of Psychometrika. Suitable.\n\n"
                                          "Final Ranking List:\n"
                                          "[1] > [2] > [0]"},
                                          
            # --- ACTIVE QUERY ---
            {"role": "user", "content": f"I will provide you with {len(pointwise_candidates)} candidate journals, each indicated by number identifier [].\n"
                                     f"Rank the journals based on their suitability and domain match for publishing the research paper:\n"
                                     f"Title: {title}\n"
                                     f"Abstract: {abstract}"},
            {"role": "assistant", "content": "Okay, please provide the journals."}
        ]
        
        # Format the active candidate list
        cand_content = []
        for c in pointwise_candidates:
            cand_content.append(f"[{c['index']}] Journal: {c['name']}\nAims: {c['aims']}\nCategories: {c['categories']}")
        
        messages.append({"role": "user", "content": "\n".join(cand_content)})
        
        prompt = qwen_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        qwen_inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
        
        with torch.no_grad():
            out = qwen_model.generate(
                **qwen_inputs,
                max_new_tokens=800,
                do_sample=False,
                eos_token_id=qwen_tokenizer.eos_token_id,
                pad_token_id=qwen_tokenizer.pad_token_id,
            )
        gen_ids = out[0][qwen_inputs.input_ids.shape[1]:]
        output_text = qwen_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        ranks = parse_listwise_output(output_text, len(pointwise_candidates))
        
        listwise_candidates = [pointwise_candidates[r] for r in ranks]
        listwise_labels = [c['label'] for c in listwise_candidates]
        
        if correct_label == listwise_labels[0]:
            listwise_top1_hits += 1
        if correct_label in listwise_labels[:5]:
            listwise_top5_hits += 1
        if correct_label in listwise_labels[:10]:
            listwise_top10_hits += 1
        listwise_mrr_sum += calculate_mrr(listwise_labels, correct_label)
        listwise_ndcg5_sum += calculate_ndcg_at_k(listwise_labels, correct_label, 5)
        listwise_ndcg10_sum += calculate_ndcg_at_k(listwise_labels, correct_label, 10)
            
        print(f"\n--- Paper {i+1} ---")
        print(f"Title: {title[:80]}...")
        print(f"Ground Truth: {correct_journal_name} (Label: {correct_label})")
        print("Pointwise Top 3 Recommendations:")
        for idx, c in enumerate(pointwise_candidates[:3]):
            print(f"  {idx+1}. {c['name']} (Score: {c['score']:.4f})")
        print("Listwise Reranked Top 3 Recommendations:")
        for idx, c in enumerate(listwise_candidates[:3]):
            print(f"  {idx+1}. {c['name']}")
        print(f"Listwise output:\n{output_text}")
        
        # Extract reasoning and ranking output
        lines = output_text.split('\n')
        reasoning_lines = []
        ranking_line = ""
        for line in lines:
            if '>' in line:
                ranking_line = line
            else:
                reasoning_lines.append(line)
        reasoning = "\n".join(reasoning_lines).strip()
        
        results_log.append({
            "paper_index": i,
            "title": title,
            "abstract": abstract,
            "ground_truth_label": correct_label,
            "ground_truth_journal": correct_journal_name,
            "pointwise_top10": [c['name'] for c in pointwise_candidates],
            "listwise_top10": [c['name'] for c in listwise_candidates],
            "reasoning": reasoning,
            "ranking_output": ranking_line.strip()
        })
        
    p_acc1 = pointwise_top1_hits / num_eval_papers
    p_acc5 = pointwise_top5_hits / num_eval_papers
    p_acc10 = pointwise_top10_hits / num_eval_papers
    p_mrr = pointwise_mrr_sum / num_eval_papers
    p_ndcg5 = pointwise_ndcg5_sum / num_eval_papers
    p_ndcg10 = pointwise_ndcg10_sum / num_eval_papers
    
    l_acc1 = listwise_top1_hits / num_eval_papers
    l_acc5 = listwise_top5_hits / num_eval_papers
    l_acc10 = listwise_top10_hits / num_eval_papers
    l_mrr = listwise_mrr_sum / num_eval_papers
    l_ndcg5 = listwise_ndcg5_sum / num_eval_papers
    l_ndcg10 = listwise_ndcg10_sum / num_eval_papers
    
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Papers Evaluated: {num_eval_papers}")
    print(f"Pointwise Stage (BioBERT + Checkpoint):")
    print(f"  Accuracy@1:  {p_acc1:.4f}")
    print(f"  Accuracy@5:  {p_acc5:.4f}")
    print(f"  Accuracy@10: {p_acc10:.4f}")
    print(f"  MRR:         {p_mrr:.4f}")
    print(f"  NDCG@5:      {p_ndcg5:.4f}")
    print(f"  NDCG@10:     {p_ndcg10:.4f}")
    print(f"Listwise Stage (Qwen Few-Shot Reranked):")
    print(f"  Accuracy@1:  {l_acc1:.4f}")
    print(f"  Accuracy@5:  {l_acc5:.4f}")
    print(f"  Accuracy@10: {l_acc10:.4f}")
    print(f"  MRR:         {l_mrr:.4f}")
    print(f"  NDCG@5:      {l_ndcg5:.4f}")
    print(f"  NDCG@10:     {l_ndcg10:.4f}")
    print("====================================================")
    
    output_dir = "results/medprs"
    os.makedirs(output_dir, exist_ok=True)
    log_df = pd.DataFrame(results_log)
    log_df.to_json(os.path.join(output_dir, "medprs_fewshot_results.json"), orient="records", indent=2)
    
    summary_path = os.path.join(output_dir, "medprs_fewshot_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total Papers Evaluated: {num_eval_papers}\n")
        f.write(f"Pointwise Stage (BioBERT):\n")
        f.write(f"  Accuracy@1:  {p_acc1:.4f}\n")
        f.write(f"  Accuracy@5:  {p_acc5:.4f}\n")
        f.write(f"  Accuracy@10: {p_acc10:.4f}\n")
        f.write(f"  MRR:         {p_mrr:.4f}\n")
        f.write(f"  NDCG@5:      {p_ndcg5:.4f}\n")
        f.write(f"  NDCG@10:     {p_ndcg10:.4f}\n\n")
        f.write(f"Listwise Stage (Qwen Few-Shot Reranked):\n")
        f.write(f"  Accuracy@1:  {l_acc1:.4f}\n")
        f.write(f"  Accuracy@5:  {l_acc5:.4f}\n")
        f.write(f"  Accuracy@10: {l_acc10:.4f}\n")
        f.write(f"  MRR:         {l_mrr:.4f}\n")
        f.write(f"  NDCG@5:      {l_ndcg5:.4f}\n")
        f.write(f"  NDCG@10:     {l_ndcg10:.4f}\n")
    print(f"Results and summary saved locally to {output_dir}/")
    
    kaggle_dst = "/kaggle/working/results"
    try:
        if os.path.exists(kaggle_dst):
            shutil.rmtree(kaggle_dst)
        shutil.copytree(output_dir, os.path.join(kaggle_dst, "medprs"), dirs_exist_ok=True)
        print(f"Successfully copied results to Kaggle Output: {kaggle_dst}/")
    except Exception as e:
        print(f"Note: Could not copy to /kaggle/working/ (likely not running on Kaggle): {e}")

if __name__ == "__main__":
    main()
