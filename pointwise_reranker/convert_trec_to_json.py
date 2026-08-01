#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Convert TREC run files to Listwise-compatible JSON."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Directory containing the TREC run files and where JSON will be written.",
    )
    # Optional: separate output dir (defaults to --path)
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for JSON files (defaults to --path).",
    )
    # Directory with the original JSONL files
    parser.add_argument(
        "--bm25_dir",
        default="./data/bm25_beir_dl19_20",
        help="Directory containing the original BM25 JSONL files.",
    )
    args = parser.parse_args()

    in_dir = Path(args.path).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trec_paths = [
        "ranked_dl19.trec", "ranked_dl20.trec", "ranked_covid.trec",
        "ranked_dbpedia.trec", "ranked_news.trec", "ranked_nfc.trec",
        "ranked_robust04.trec", "ranked_scifact.trec", "ranked_signal.trec",
        "ranked_touche.trec"
    ]
    output_json_path = [
        "ranked_dl19.json","ranked_dl20.json", "ranked_covid.json",
        "ranked_dbpedia.json", "ranked_news.json", "ranked_nfc.json",
        "ranked_robust04.json", "ranked_scifact.json", "ranked_signal.json",
        "ranked_touche.json"
    ]

    for i, trec_name in enumerate(trec_paths):
        trec_path = in_dir / trec_name
        out_path = out_dir / output_json_path[i]

        if not trec_path.exists():
            print(f"⚠️  Skipping: {trec_path} not found.")
            continue

        try:
            print(f"\n=== [{trec_name}] ===")
            dataset = trec_name.replace("ranked_", "").replace(".trec", "")
            
            # Load queries and contents from JSONL
            content_cache = {}
            query_text_cache = {}
            jsonl_path = Path(args.bm25_dir) / f"{dataset}.jsonl"
            if not jsonl_path.exists():
                jsonl_path = Path(args.bm25_dir) / f"{dataset}.json"
                
            if jsonl_path.exists():
                print(f"Loading document/query texts from {jsonl_path}...")
                with jsonl_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        qid = str(item.get("query_id", item.get("qid", "")))
                        docid = str(item.get("docid", item.get("doc_id", "")))
                        query = item.get("query", "")
                        title = item.get("title", "")
                        text = item.get("text", "")
                        
                        if title:
                            content = f"Title: {title} Content: {text}"
                        else:
                            content = text
                        content = " ".join(content.split())
                        
                        query_text_cache[qid] = query
                        if qid not in content_cache:
                            content_cache[qid] = {}
                        content_cache[qid][docid] = content
            else:
                print(f"⚠️  Warning: Original JSONL not found at {jsonl_path}. Query/document text will fall back to placeholder IDs.")

            print("Parsing TREC file...")
            results = defaultdict(list)
            with trec_path.open("r", encoding="utf-8") as f:
                for line in f:
                    qid, _, docid, rank, score, _ = line.strip().split()
                    results[qid].append((int(rank), docid, float(score)))

            print("Building Listwise-compatible JSON...")
            final_output = []
            for qid, tuples in results.items():
                query_text = query_text_cache.get(qid, str(qid))
                query_entry = {"query": query_text, "qid": qid, "hits": []}

                for rank, docid, score in sorted(tuples, key=lambda x: x[0]):
                    content = content_cache.get(qid, {}).get(docid, f"Document {docid}")
                    query_entry["hits"].append({
                        "content": content,
                        "qid": qid,
                        "docid": docid,
                        "rank": rank,
                        "score": score
                    })

                final_output.append(query_entry)

            print(f"Saving to {out_path} ...")
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2, ensure_ascii=False)

            print("✅ Done.")
        except Exception as e:
            print(f"❌ Error processing {trec_name}: {e}")

if __name__ == "__main__":
    main()
