import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil
import math
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.optim import AdamW

def find_offline_model(default_name, search_pattern):
    local_path = os.path.basename(default_name)
    if os.path.exists(local_path) and os.path.isdir(local_path):
        print(f"Detected offline model folder in working directory: {local_path}")
        return local_path
        
    kaggle_input = "/kaggle/input"
    if os.path.exists(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            if "config.json" in files:
                if search_pattern.lower() in root.lower():
                    print(f"Automatically detected Kaggle offline model path: {root}")
                    return root
                         
    print(f"Offline model not found for pattern '{search_pattern}'. Falling back to online: {default_name}")
    return default_name

# Custom PyTorch Dataset for Cross-Encoder Training
class CrossEncoderDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.pairs)
        
    def __getitem__(self, idx):
        paper_text, journal_text, label = self.pairs[idx]
        inputs = self.tokenizer(
            paper_text,
            journal_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(label, dtype=torch.float)
        return item

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
        paper_proj_norm = paper_proj_embeddings / torch.clamp(paper_proj_embeddings.norm(dim=-1, keepdim=True), min=1e-9)
        journal_proj_norm = journal_proj_embeddings / torch.clamp(journal_proj_embeddings.norm(dim=-1, keepdim=True), min=1e-9)
        sim_vector = torch.matmul(paper_proj_norm, journal_proj_norm.t())
        joint = torch.cat([paper_proj_embeddings, sim_vector], dim=-1)
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

def main():
    print("=== MEDPRS JOURNAL RECOMMENDATION - CROSS-ENCODER PIPELINE ===")
    
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    import argparse
    parser = argparse.ArgumentParser(description="MedPRS Cross-Encoder Training & Evaluation.")
    parser.add_argument("--num_eval", type=int, default=1000, help="Number of papers to evaluate (from val set).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--samples_per_label", type=int, default=30, help="Max papers per label for training.")
    args, unknown = parser.parse_known_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    journal_path = "data_MedPRS/journal_category.csv"
    train_path = "data_MedPRS/train_set.csv"
    val_path = "data_MedPRS/val_set.csv"
    checkpoint_path = "Epoch_02_SIMCPRS_dmis-lab_biobert-v1_1_CL.pth"
    
    if not os.path.exists(journal_path) or not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: Datasets not found under data_MedPRS/.")
        sys.exit(1)
        
    print("Loading datasets...")
    journal_df = pd.read_csv(journal_path).fillna("")
    train_df = pd.read_csv(train_path).fillna("")
    val_df = pd.read_csv(val_path).fillna("")
    
    print(f"Loaded {len(journal_df)} journals, {len(train_df)} training papers, and {len(val_df)} validation papers.")
    
    # 1. Stratified Sub-sampling on Training Data
    print(f"Performing stratified sampling (max {args.samples_per_label} papers per label)...")
    train_sampled = train_df.groupby('Label').apply(
        lambda x: x.sample(n=min(len(x), args.samples_per_label), random_state=42)
    ).reset_index(drop=True)
    print(f"Sampled training size: {len(train_sampled)} papers.")
    
    # 2. Construct Positive & Negative Pairs for Cross-Encoder
    journal_info = {}
    journal_labels = []
    for idx, row in journal_df.iterrows():
        lbl = int(row['Label'])
        journal_labels.append(lbl)
        journal_info[lbl] = f"Journal Name: {row['Journal']}\nAims and Scope: {row['Aims']}\nCategories: {row['Categories']}"
        
    print("Constructing positive and negative training pairs...")
    train_pairs = []
    for idx, row in tqdm(train_sampled.iterrows(), total=len(train_sampled)):
        title = row['Title']
        abstract = row['Abstract']
        keywords = row['Keywords']
        correct_lbl = int(row['Label'])
        
        paper_text = f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords}"
        
        # Positive pair
        if correct_lbl in journal_info:
            train_pairs.append((paper_text, journal_info[correct_lbl], 1.0))
            
            # Negative pair (randomly choose an incorrect journal)
            wrong_lbl = correct_lbl
            while wrong_lbl == correct_lbl:
                wrong_lbl = random.choice(journal_labels)
            train_pairs.append((paper_text, journal_info[wrong_lbl], 0.0))
            
    print(f"Total training pairs: {len(train_pairs)} (50% positive, 50% negative).")
    
    # 3. Load Model and Tokenizer
    biobert_model_name = find_offline_model("dmis-lab/biobert-v1.1", "biobert")
    tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
    
    print("Initializing Cross-Encoder model...")
    cross_encoder = AutoModelForSequenceClassification.from_pretrained(biobert_model_name, num_labels=1)
    cross_encoder.to(device)
    
    # 4. Prepare PyTorch DataLoader
    train_dataset = CrossEncoderDataset(train_pairs, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    optimizer = AdamW(cross_encoder.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # 5. Training Loop
    print(f"Starting Cross-Encoder training for {args.epochs} epochs...")
    cross_encoder.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch['labels'].to(device).unsqueeze(1)
            
            with torch.cuda.amp.autocast():
                outputs = cross_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs.logits
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Complete. Average Loss: {epoch_loss / len(train_loader):.4f}")
        
    # Save final model
    output_model_dir = "biobert_cross_encoder_final"
    os.makedirs(output_model_dir, exist_ok=True)
    cross_encoder.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Saved final Cross-Encoder model to {output_model_dir}/")
    
    # 6. Evaluation Giai Đoạn 2 (Pointwise + Cross-Encoder Rerank)
    print(f"\nEvaluating on the first {args.num_eval} validation papers...")
    
    # Loading base pointwise model to get candidate list
    biobert_base = AutoModel.from_pretrained(biobert_model_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    pointwise_model = SimCPSRModel(biobert_base)
    pointwise_model.init_heads(state_dict)
    
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
        
    pointwise_model.load_state_dict(clean_state_dict, strict=False)
    pointwise_model.to(device)
    pointwise_model.eval()
    
    is_multi_class = pointwise_model.linear_main_1.out_features > 1000
    journal_proj_embeddings = None
    
    if is_multi_class:
        print("Computing pointwise journal embeddings...")
        journal_proj_embeddings = torch.zeros((pointwise_model.linear_main_1.out_features, 512), device=device)
        for idx, row in journal_df.iterrows():
            name = row['Journal']
            aims = row['Aims']
            cats = row['Categories']
            lbl = int(row['Label'])
            text = f"Journal Name: {name}\nAims and Scope: {aims}\nCategories: {cats}"
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                emb = pointwise_model.encode_journal(inputs['input_ids'], inputs['attention_mask'])
            if 0 <= lbl < len(journal_proj_embeddings):
                journal_proj_embeddings[lbl] = emb[0]
                
    cross_encoder.eval()
    
    p_acc1, p_acc5, p_acc10 = 0, 0, 0
    p_mrr, p_ndcg5, p_ndcg10 = 0.0, 0.0, 0.0
    
    ce_acc1, ce_acc5, ce_acc10 = 0, 0, 0
    ce_mrr, ce_ndcg5, ce_ndcg10 = 0.0, 0.0, 0.0
    
    results_log = []
    
    for i in tqdm(range(args.num_eval), desc="Evaluating papers"):
        row = val_df.iloc[i]
        title = row['Title']
        abstract = row['Abstract']
        keywords = row['Keywords']
        correct_label = int(row['Label'])
        
        paper_text = f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords}"
        
        # Pointwise score
        inputs = tokenizer(paper_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            paper_emb = pointwise_model.encode_paper(inputs['input_ids'], inputs['attention_mask'])
            logits = pointwise_model(paper_emb, journal_proj_embeddings)
        logits = logits.cpu().numpy()[0]
        
        scores = []
        for idx, r in journal_df.iterrows():
            lbl = int(r['Label'])
            scores.append(logits[lbl] if 0 <= lbl < len(logits) else -9999.0)
            
        top10_indices = np.argsort(scores)[::-1][:10]
        
        pointwise_candidates = []
        for rank_idx, idx in enumerate(top10_indices):
            j_row = journal_df.iloc[idx]
            pointwise_candidates.append({
                "label": int(j_row['Label']),
                "name": j_row['Journal']
            })
            
        pointwise_labels = [c['label'] for c in pointwise_candidates]
        if correct_label == pointwise_labels[0]:
            p_acc1 += 1
        if correct_label in pointwise_labels[:5]:
            p_acc5 += 1
        if correct_label in pointwise_labels[:10]:
            p_acc10 += 1
        p_mrr += calculate_mrr(pointwise_labels, correct_label)
        p_ndcg5 += calculate_ndcg_at_k(pointwise_labels, correct_label, 5)
        p_ndcg10 += calculate_ndcg_at_k(pointwise_labels, correct_label, 10)
        
        # Cross-Encoder Rerank
        ce_scores = []
        for c in pointwise_candidates:
            j_lbl = c['label']
            j_text = journal_info[j_lbl]
            
            inputs = tokenizer(
                paper_text,
                j_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = cross_encoder(**inputs)
                score = outputs.logits.item()
            ce_scores.append((j_lbl, c['name'], score))
            
        ce_scores.sort(key=lambda x: x[2], reverse=True)
        ce_labels = [c[0] for c in ce_scores]
        
        if correct_label == ce_labels[0]:
            ce_acc1 += 1
        if correct_label in ce_labels[:5]:
            ce_acc5 += 1
        if correct_label in ce_labels[:10]:
            ce_acc10 += 1
        ce_mrr += calculate_mrr(ce_labels, correct_label)
        ce_ndcg5 += calculate_ndcg_at_k(ce_labels, correct_label, 5)
        ce_ndcg10 += calculate_ndcg_at_k(ce_labels, correct_label, 10)
        
    p_acc1 /= args.num_eval
    p_acc5 /= args.num_eval
    p_acc10 /= args.num_eval
    p_mrr /= args.num_eval
    p_ndcg5 /= args.num_eval
    p_ndcg10 /= args.num_eval
    
    ce_acc1 /= args.num_eval
    ce_acc5 /= args.num_eval
    ce_acc10 /= args.num_eval
    ce_mrr /= args.num_eval
    ce_ndcg5 /= args.num_eval
    ce_ndcg10 /= args.num_eval
    
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Papers Evaluated: {args.num_eval}")
    print(f"Pointwise Stage (BioBERT):")
    print(f"  Accuracy@1:  {p_acc1:.4f}  |  Accuracy@5:  {p_acc5:.4f}  |  NDCG@10: {p_ndcg10:.4f}")
    print(f"Cross-Encoder Stage (Fine-tuned BioBERT Reranked):")
    print(f"  Accuracy@1:  {ce_acc1:.4f}  |  Accuracy@5:  {ce_acc5:.4f}  |  NDCG@10: {ce_ndcg10:.4f}")
    print(f"  MRR:         {ce_mrr:.4f}  |  NDCG@5:      {ce_ndcg5:.4f}")
    print("====================================================")
    
    # Save text summary
    os.makedirs("results/medprs", exist_ok=True)
    with open("results/medprs/medprs_cross_encoder_summary.txt", "w") as f:
        f.write(f"Total Papers Evaluated: {args.num_eval}\n")
        f.write(f"Pointwise Stage (BioBERT):\n")
        f.write(f"  Accuracy@1:  {p_acc1:.4f}\n")
        f.write(f"  Accuracy@5:  {p_acc5:.4f}\n")
        f.write(f"  Accuracy@10: {p_acc10:.4f}\n")
        f.write(f"  MRR:         {p_mrr:.4f}\n")
        f.write(f"  NDCG@5:      {p_ndcg5:.4f}\n")
        f.write(f"  NDCG@10:     {p_ndcg10:.4f}\n\n")
        f.write(f"Cross-Encoder Stage (Fine-tuned BioBERT):\n")
        f.write(f"  Accuracy@1:  {ce_acc1:.4f}\n")
        f.write(f"  Accuracy@5:  {ce_acc5:.4f}\n")
        f.write(f"  Accuracy@10: {ce_acc10:.4f}\n")
        f.write(f"  MRR:         {ce_mrr:.4f}\n")
        f.write(f"  NDCG@5:      {ce_ndcg5:.4f}\n")
        f.write(f"  NDCG@10:     {ce_ndcg10:.4f}\n")
        
    # Copy results to /kaggle/working
    try:
        shutil.copytree("results/medprs", "/kaggle/working/results/medprs", dirs_exist_ok=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
