import os
import re
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

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

    def encode_journal(self, journal_input_ids, journal_attention_mask):
        outputs = self.base_model(input_ids=journal_input_ids, attention_mask=journal_attention_mask)
        token_embeddings = outputs[0]
        mask_expanded = journal_attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        emb = torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        if self.linear2_1 is not None:
            emb = self.linear2_1(emb)
            emb = torch.relu(emb)
        return emb

    def forward(self, paper_input_ids, paper_attention_mask, journal_proj_embeddings=None):
        paper_outputs = self.base_model(input_ids=paper_input_ids, attention_mask=paper_attention_mask)
        paper_token_embeddings = paper_outputs[0]
        paper_mask_expanded = paper_attention_mask.unsqueeze(-1).expand(paper_token_embeddings.size()).float()
        paper_emb = torch.sum(paper_token_embeddings * paper_mask_expanded, 1) / torch.clamp(paper_mask_expanded.sum(1), min=1e-9)
        
        if self.linear1_1 is not None:
            paper_proj = self.linear1_1(paper_emb)
            paper_proj = torch.relu(paper_proj)
        else:
            paper_proj = paper_emb
            
        if journal_proj_embeddings is not None and self.linear_main_1 is not None:
            paper_proj_norm = paper_proj / torch.clamp(paper_proj.norm(dim=-1, keepdim=True), min=1e-9)
            journal_proj_norm = journal_proj_embeddings / torch.clamp(journal_proj_embeddings.norm(dim=-1, keepdim=True), min=1e-9)
            
            sim_vector = torch.matmul(paper_proj_norm, journal_proj_norm.t())
            joint = torch.cat([paper_proj, sim_vector], dim=-1)
            logits = self.linear_main_1(joint)
            return logits
            
        return paper_proj

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
    print("=== MEDPRS JOURNAL RECOMMENDATION INFERENCE PIPELINE (QWEN CO-T) ===")
    
    import argparse
    parser = argparse.ArgumentParser(description="MedPRS journal recommendation inference with Qwen.")
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
    
    # Fill NA values
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
    biobert_model_name = "dmis-lab/biobert-v1.1"
    biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
    biobert_base = AutoModel.from_pretrained(biobert_model_name)
    
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # If state dict is nested inside model_state_dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Instantiate custom wrapper model
    model = SimCPSRModel(biobert_base)
    model.init_heads(state_dict)
    
    # Strip prefixes and load into model
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
            
    # 4. Load Qwen 2.5 7B Instruct Model in 4-bit (VRAM optimized)
    print("Loading Qwen-2.5-7B-Instruct Model...")
    qwen_repo = "Qwen/Qwen2.5-7B-Instruct"
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_repo, use_fast=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    # Load Qwen base model (without PEFT because it's a full model, not an adapter)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_repo,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if qwen_tokenizer.pad_token is None:
        qwen_tokenizer.pad_token = qwen_tokenizer.eos_token
        
    qwen_model.eval()
    print("Qwen 2.5 7B Instruct loaded successfully.")
    
    # 5. Run Evaluation on a Subset of Validation Papers
    num_eval_papers = args.num_papers
    print(f"\nEvaluating the first {num_eval_papers} papers...")
    
    pointwise_top1_hits = 0
    pointwise_top5_hits = 0
    pointwise_top10_hits = 0
    
    listwise_top1_hits = 0
    listwise_top5_hits = 0
    listwise_top10_hits = 0
    
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
                logits = model(inputs['input_ids'], inputs['attention_mask'], journal_proj_embeddings=journal_proj_embeddings)
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
            
        # 6. Construct Listwise Prompt for Qwen
        SYSTEM_PROMPT = "You are RankLLM, an expert assistant that ranks academic journals by their suitability and domain match for publishing a given research paper."
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"I will provide you with {len(pointwise_candidates)} candidate journals, each indicated by number identifier [].\n"
                                     f"Rank the journals based on their suitability and domain match for publishing the research paper:\n"
                                     f"Title: {title}\n"
                                     f"Abstract: {abstract}"},
            {"role": "assistant", "content": "Okay, please provide the journals."}
        ]
        
        for c in pointwise_candidates:
            messages.append({"role": "user", "content": f"[{c['index']}] Journal: {c['name']}\nAims: {c['aims']}\nCategories: {c['categories']}"})
            messages.append({"role": "assistant", "content": f"Received journal [{c['index']}]."})
            
        post_prompt = (
            "Search Query: Rank the journals above based on their suitability for the research paper.\n"
            "Please follow these steps:\n"
            "Step 1: Write a concise 2-sentence analysis explaining why each candidate journal fits or does not fit the research paper, based on their Aims & Scope and Categories.\n"
            "Step 2: Rank the journals in descending order of suitability using the number identifiers [].\n\n"
            "Output format:\n"
            "Write the analyses for all journals first, then at the very end of your response, print the final ranking list in this exact format:\n"
            "[2] > [1] > [3]"
        )
        messages.append({"role": "user", "content": post_prompt})
        
        prompt = qwen_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        qwen_inputs = qwen_tokenizer(prompt, return_tensors="pt").to(qwen_model.device)
        
        with torch.no_grad():
            out = qwen_model.generate(
                **qwen_inputs,
                max_new_tokens=512,  # Give Qwen plenty of space to write explanations
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
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
    
    l_acc1 = listwise_top1_hits / num_eval_papers
    l_acc5 = listwise_top5_hits / num_eval_papers
    l_acc10 = listwise_top10_hits / num_eval_papers
    
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Papers Evaluated: {num_eval_papers}")
    print(f"Pointwise Stage (BioBERT + Checkpoint):")
    print(f"  Accuracy@1:  {p_acc1:.4f}")
    print(f"  Accuracy@5:  {p_acc5:.4f}")
    print(f"  Accuracy@10: {p_acc10:.4f}")
    print(f"Listwise Stage (Qwen Reranked):")
    print(f"  Accuracy@1:  {l_acc1:.4f}")
    print(f"  Accuracy@5:  {l_acc5:.4f}")
    print(f"  Accuracy@10: {l_acc10:.4f}")
    print("====================================================")
    
    # Save log to local results folder
    output_dir = "results/medprs"
    os.makedirs(output_dir, exist_ok=True)
    log_df = pd.DataFrame(results_log)
    log_df.to_json(os.path.join(output_dir, "medprs_rerank_results.json"), orient="records", indent=2)
    
    summary_path = os.path.join(output_dir, "medprs_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Total Papers Evaluated: {num_eval_papers}\n")
        f.write(f"Pointwise Stage (BioBERT):\n")
        f.write(f"  Accuracy@1:  {p_acc1:.4f}\n")
        f.write(f"  Accuracy@5:  {p_acc5:.4f}\n")
        f.write(f"  Accuracy@10: {p_acc10:.4f}\n\n")
        f.write(f"Listwise Stage (Qwen Reranked):\n")
        f.write(f"  Accuracy@1:  {l_acc1:.4f}\n")
        f.write(f"  Accuracy@5:  {l_acc5:.4f}\n")
        f.write(f"  Accuracy@10: {l_acc10:.4f}\n")
    print(f"Results and summary saved locally to {output_dir}/")
    
    # CRITICAL KAGGLING COPIER: Save results out to /kaggle/working/ to survive Git repo exclusions
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
