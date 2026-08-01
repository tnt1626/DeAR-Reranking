#!/usr/bin/env python3
import sys
import os
import urllib.request
import urllib.error
import pytrec_eval

def get_qrels_file(name):
    if os.path.exists(name):
        return name
    filename = name
    if not filename.startswith("qrels."):
        filename = f"qrels.{filename}"
    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    cache_dir = os.path.expanduser("~/.cache/dear_reranker_qrels")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, filename)
    if os.path.exists(local_path):
        return local_path

    url = f"https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/{filename}"
    print(f"Downloading qrels from {url} to {local_path}...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except urllib.error.HTTPError as e:
        if e.code == 404 and filename.endswith("-test.txt"):
            alt_filename = filename[:-9] + ".test.txt"
            alt_url = f"https://raw.githubusercontent.com/castorini/anserini-tools/master/topics-and-qrels/{alt_filename}"
            print(f"Retrying download with fallback URL {alt_url}...", file=sys.stderr)
            try:
                urllib.request.urlretrieve(alt_url, local_path)
                return local_path
            except Exception as ex:
                print(f"Fallback download failed: {ex}", file=sys.stderr)
        print(f"Failed to download from anserini-tools: {e}", file=sys.stderr)
        if os.path.exists(filename):
            return filename
        raise FileNotFoundError(f"Could not find or download qrels file {name}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python eval_trec.py <qrels_name_or_file> <run_file>", file=sys.stderr)
        sys.exit(1)
        
    qrels_arg = sys.argv[1]
    run_file = sys.argv[2]
    
    qrels_path = get_qrels_file(qrels_arg)
    
    with open(qrels_path, 'r') as f:
        qrels = pytrec_eval.parse_qrel(f)
        
    with open(run_file, 'r') as f:
        run = pytrec_eval.parse_run(f)
        
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10'})
    results = evaluator.evaluate(run)
    
    # Calculate average NDCG@10 across all queries
    ndcg_scores = [query_measures['ndcg_cut_10'] for query_measures in results.values()]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    # Output format matching standard trec_eval: ndcg_cut_10             all     0.XXXX
    print(f"ndcg_cut_10             all     {avg_ndcg:.4f}")

if __name__ == '__main__':
    main()
