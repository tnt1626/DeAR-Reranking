import os
import re
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        prefixes = ["base_model.bert.", "base_model.", "model.", "bert.", "encoder."]
        matched = False
        if name in own_state:
            new_state_dict[name] = v
            matched = True
        else:
            for prefix in prefixes:
                if name.startswith(prefix):
                    sub_name = name[len(prefix):]
                    if sub_name in own_state:
                        new_state_dict[sub_name] = v
                        matched = True
                        break
        if not matched:
            print(f"Skipping key {k} (not in base BioBERT model state dict)")
    info = model.load_state_dict(new_state_dict, strict=False)
    print(f"Successfully loaded {len(new_state_dict)} keys. Missing keys: {len(info.missing_keys)}")

def get_embedding(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    attention_mask = inputs['attention_mask']
    token_embeddings = outputs[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return embedding.cpu().numpy()[0]

def parse_listwise_output(output_text, num_candidates):
    # Find the line containing '>' representing the final ranking list
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
    print("=== MEDPRS JOURNAL RECOMMENDATION INFERENCE PIPELINE ===")
    
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
    print("Loading BioBERT tokenizer and model...")
    biobert_model_name = "dmis-lab/biobert-v1.1"
    biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
    biobert_model = AutoModel.from_pretrained(biobert_model_name).to(device)
    
    print(f"Loading checkpoint weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    load_my_state_dict(biobert_model, state_dict)
    biobert_model.eval()
    
    # 3. Compute and Cache Journal Embeddings
    embedding_cache_path = "data_MedPRS/journal_embeddings.npy"
    
    if os.path.exists(embedding_cache_path):
        print(f"Found cached journal embeddings at {embedding_cache_path}. Loading...")
        journal_embeddings = np.load(embedding_cache_path)
        print("Journal embeddings loaded from cache successfully.")
    else:
        print("Computing embeddings for all 1,408 journals (this runs once)...")
        journal_embeddings = []
        for idx, row in tqdm(journal_df.iterrows(), total=len(journal_df)):
            name = row['Journal']
            aims = row['Aims']
            cats = row['Categories']
            text = f"Journal: {name}\nAims: {aims}\nCategories: {cats}"
            
            emb = get_embedding(biobert_model, biobert_tokenizer, text, device)
            journal_embeddings.append(emb)
            
        journal_embeddings = np.array(journal_embeddings)
        os.makedirs(os.path.dirname(embedding_cache_path), exist_ok=True)
        np.save(embedding_cache_path, journal_embeddings)
        print(f"Journal embeddings computed and saved to cache at {embedding_cache_path}.")
    
    # 4. Load Llama 3.1 8B Listwise Reranker in 4-bit (VRAM optimized)
    print("Loading Llama 3.1 8B Listwise Reranker...")
    llama_repo = "abdoelsayed/dear-8b-reranker-listwise-lora-v1"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_repo, use_fast=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    llama_model = AutoPeftModelForCausalLM.from_pretrained(
        llama_repo,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        
    llama_model.eval()
    print("Llama 3.1 8B Listwise Reranker loaded successfully.")
    
    # 5. Run Evaluation on a Subset of Validation Papers
    num_eval_papers = 20  # Adjust as needed
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
        
        # Format query
        paper_text = f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords}"
        paper_emb = get_embedding(biobert_model, biobert_tokenizer, paper_text, device)
        
        # Calculate Cosine Similarities
        similarities = []
        for j_emb in journal_embeddings:
            sim = np.dot(paper_emb, j_emb) / (np.linalg.norm(paper_emb) * np.linalg.norm(j_emb))
            similarities.append(sim)
            
        # Get Top 10 candidate journals from Pointwise
        top10_indices = np.argsort(similarities)[::-1][:10]
        
        pointwise_candidates = []
        for rank_idx, idx in enumerate(top10_indices):
            j_row = journal_df.iloc[idx]
            pointwise_candidates.append({
                "index": rank_idx,
                "label": int(j_row['Label']),
                "name": j_row['Journal'],
                "aims": j_row['Aims'],
                "categories": j_row['Categories'],
                "score": float(similarities[idx])
            })
            
        # Calculate Pointwise hits
        pointwise_labels = [c['label'] for c in pointwise_candidates]
        if correct_label == pointwise_labels[0]:
            pointwise_top1_hits += 1
        if correct_label in pointwise_labels[:5]:
            pointwise_top5_hits += 1
        if correct_label in pointwise_labels[:10]:
            pointwise_top10_hits += 1
            
        # 6. Construct Listwise Prompt for Llama
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
            "The journals should be listed in descending order of suitability using identifiers.\n"
            "Please follow the steps below:\n"
            "Step 1. Analyze the research focus, methods, and contributions of the given paper.\n"
            "Step 2. Match the paper's domain with each candidate journal's aims, scope, and categories.\n"
            "Step 3. Rank the journals from most suitable to least suitable. Include all journals.\n"
            "Output format strictly ends with: [2] > [1] > [3]"
        )
        messages.append({"role": "user", "content": post_prompt})
        
        # Apply Llama chat template
        prompt = llama_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        llama_inputs = llama_tokenizer(prompt, return_tensors="pt").to(llama_model.device)
        
        # Generate Reranking
        with torch.no_grad():
            out = llama_model.generate(
                **llama_inputs,
                max_new_tokens=384,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                eos_token_id=llama_tokenizer.eos_token_id,
                pad_token_id=llama_tokenizer.pad_token_id,
            )
        gen_ids = out[0][llama_inputs.input_ids.shape[1]:]
        output_text = llama_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Parse output
        ranks = parse_listwise_output(output_text, len(pointwise_candidates))
        
        # Rerank candidates based on Llama output
        listwise_candidates = [pointwise_candidates[r] for r in ranks]
        listwise_labels = [c['label'] for c in listwise_candidates]
        
        # Calculate Listwise hits
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
        print(f"Listwise output: {output_text}")
        
        # Split output_text into reasoning and ranking line
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
        
    # Print overall stats
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
    print(f"Listwise Stage (Llama-3.1 Reranked):")
    print(f"  Accuracy@1:  {l_acc1:.4f}")
    print(f"  Accuracy@5:  {l_acc5:.4f}")
    print(f"  Accuracy@10: {l_acc10:.4f}")
    print("====================================================")
    
    # Save log to results folder
    output_dir = "results/medprs"
    os.makedirs(output_dir, exist_ok=True)
    log_df = pd.DataFrame(results_log)
    log_df.to_json(os.path.join(output_dir, "medprs_rerank_results.json"), orient="records", indent=2)
    
    with open(os.path.join(output_dir, "medprs_summary.txt"), "w") as f:
        f.write(f"Total Papers Evaluated: {num_eval_papers}\n")
        f.write(f"Pointwise Stage (BioBERT):\n")
        f.write(f"  Accuracy@1:  {p_acc1:.4f}\n")
        f.write(f"  Accuracy@5:  {p_acc5:.4f}\n")
        f.write(f"  Accuracy@10: {p_acc10:.4f}\n\n")
        f.write(f"Listwise Stage (Llama-3.1 Reranked):\n")
        f.write(f"  Accuracy@1:  {l_acc1:.4f}\n")
        f.write(f"  Accuracy@5:  {l_acc5:.4f}\n")
        f.write(f"  Accuracy@10: {l_acc10:.4f}\n")
    print(f"Results and summary saved to {output_dir}/")

if __name__ == "__main__":
    main()
