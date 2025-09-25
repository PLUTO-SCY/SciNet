import json
import os
import sqlite3
import duckdb
import numpy as np
import itertools
from tqdm import tqdm
from scipy.stats import percentileofscore

# --- 1. Configuration ---

# Input: Your recall results file
RECALL_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/SciBERT/answers_task1_novel_faiss.json"

# Input: Directory containing the reference novelty scores
REFERENCE_SCORES_DIR = "/data5/shaochenyang/AI_Scientist/OpenAlex/task1Result/CNovelty_multiprocess_5000_final"

# Cache file for the reference distribution to speed up subsequent runs
REFERENCE_CACHE_FILE = "novelty_distribution.npy"

# Database files required for calculating raw novelty scores
Z_SCORES_FILE = "/data5/shaochenyang/AI_Scientist/OpenAlex/task1Result/combine-novelty-juhe/final_z_scores_smart_filtered_and_sorted.parquet"
FORWARD_CITATION_DB = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/citing_to_cited.db"

# Output: File to save the detailed evaluation results
OUTPUT_FILE = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/EvaluationScripts/logs/scientometric_novelty_evaluation_results.json"


# --- 2. Module to Build Reference Distribution ---

def build_reference_distribution(directory_path, cache_file):
    """
    Loads novelty scores from all JSON files in a directory.
    Uses a cache file to avoid reloading on subsequent runs.
    """
    if os.path.exists(cache_file):
        print(f"Loading reference score distribution from cache: {cache_file}")
        return np.load(cache_file)

    print(f"Building reference score distribution from: {directory_path}")
    all_scores = []
    file_list = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    for filename in tqdm(file_list, desc="Reading reference files"):
        filepath = os.path.join(directory_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    score = item.get("novelty_score")
                    if score is not None:
                        all_scores.append(score)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read or parse {filename}. Error: {e}")
            
    reference_scores = np.array(all_scores)
    print(f"Built distribution with {len(reference_scores)} scores. Caching to {cache_file}...")
    np.save(cache_file, reference_scores)
    return reference_scores


# --- 3. Modules for Calculating and Transforming Scores ---

class RawNoveltyCalculator:
    """Calculates the raw p10_z novelty score for a paper."""
    def __init__(self, z_scores_path, citations_db_path):
        # DuckDB for Z-Scores
        self.z_con = duckdb.connect(read_only=True)
        self.z_scores_path = z_scores_path
        # SQLite for References
        self.cite_conn = sqlite3.connect(f"file:{citations_db_path}?mode=ro", uri=True)
        self.cite_cur = self.cite_conn.cursor()
        print("Successfully connected to Z-Score and Citation databases.")

    def _get_references(self, paper_id: str) -> list:
        short_id = paper_id.split('/')[-1]
        self.cite_cur.execute("SELECT referenced_work_ids FROM citing_to_cited WHERE work_id = ?", (short_id,))
        row = self.cite_cur.fetchone()
        return json.loads(row[0]) if row and row[0] else []

    def get_raw_novelty_score(self, paper_id: str):
        refs = self._get_references(paper_id)
        if not refs or len(refs) < 2: return None
        
        cleaned_refs = {r.split('/')[-1] for r in refs if r}
        if len(cleaned_refs) < 2: return None
            
        query_pairs = tuple(sorted(tuple(sorted(p)) for p in itertools.combinations(cleaned_refs, 2)))
        if not query_pairs: return None

        query = f"SELECT z_score FROM read_parquet('{self.z_scores_path}') WHERE (id_min, id_max) IN {query_pairs};"
        try:
            result_df = self.z_con.execute(query).fetchdf()
            if result_df.empty: return None
            z_scores = result_df['z_score'].tolist()
        except Exception: return None
            
        if not z_scores: return None
        return np.percentile(z_scores, 10)

    def close(self):
        self.z_con.close()
        self.cite_conn.close()
        print("\nDatabase connections closed.")


def transform_score_by_percentile(raw_score, reference_scores_negated):
    """Transforms a raw score to a 0-10 scale using percentile ranking."""
    if raw_score is None:
        return None
    # `percentileofscore` calculates the percentage of scores less than the given score.
    # By using negated scores, we correctly rank lower (more negative) raw scores as better.
    percentile = percentileofscore(reference_scores_negated, -raw_score)
    return percentile / 10.0


# --- 4. Main Evaluation Workflow ---

def main():
    # Step 1: Build or load the reference distribution
    reference_scores = build_reference_distribution(REFERENCE_SCORES_DIR, REFERENCE_CACHE_FILE)
    # Negate once for efficiency in the transformation function
    reference_scores_negated = reference_scores * -1

    # Step 2: Initialize the raw score calculator
    calculator = RawNoveltyCalculator(Z_SCORES_FILE, FORWARD_CITATION_DB)

    # Step 3: Load recall data
    with open(RECALL_FILE_PATH, 'r', encoding='utf-8') as f:
        recall_data = json.load(f)

    # Step 4: Checkpointing - Load existing results if they exist
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resuming analysis. Loaded {len(results)} completed queries from {OUTPUT_FILE}.")
    else:
        results = {}

    all_query_avg_scores = []
    
    # Step 5: Process each query
    for query, papers in tqdm(recall_data.items(), desc="Evaluating Queries"):
        if query in results:
            if results[query].get("average_transformed_score") is not None:
                all_query_avg_scores.append(results[query]["average_transformed_score"])
            continue

        top_5_papers = papers[:5]
        query_transformed_scores = []
        scored_papers_details = []

        for paper in top_5_papers:
            paper_id = paper.get('id')
            raw_score = calculator.get_raw_novelty_score(paper_id) if paper_id else None
            transformed_score = transform_score_by_percentile(raw_score, reference_scores_negated)
            
            scored_papers_details.append({
                "id": paper_id,
                "title": paper.get("title"),
                "raw_novelty_score": raw_score,
                "transformed_novelty_score": transformed_score
            })
            if transformed_score is not None:
                query_transformed_scores.append(transformed_score)
        
        avg_transformed_score = np.mean(query_transformed_scores) if query_transformed_scores else None
        
        results[query] = {
            "average_transformed_score": avg_transformed_score,
            "scored_papers": scored_papers_details
        }
        
        if avg_transformed_score is not None:
            all_query_avg_scores.append(avg_transformed_score)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    # Step 6: Final report and cleanup
    calculator.close()
    print("\n--- Final Novelty Evaluation Report ---")
    if all_query_avg_scores:
        overall_average = np.mean(all_query_avg_scores)
        print("\n" + "="*50)
        print(f"📊 Overall Average Transformed Novelty Score (0-10): {overall_average:.4f}")
        print("="*50)
    else:
        print("\nCould not calculate an overall average novelty score.")
    print(f"Detailed results have been saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()