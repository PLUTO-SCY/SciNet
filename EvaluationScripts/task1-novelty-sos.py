import json
import os
import sqlite3
import duckdb
import numpy as np
import itertools
from tqdm import tqdm

# --- 1. Configuration ---

# Input file containing the recalled papers for each query
RECALL_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/pasa/oaids/queries_task1_novel.json"

# Database containing pre-computed z-scores for citation pairs
Z_SCORES_FILE = "/data5/shaochenyang/AI_Scientist/OpenAlex/task1Result/combine-novelty-juhe/final_z_scores_smart_filtered_and_sorted.parquet"

# SQLite database file for fetching paper references
# This database maps a citing paper ID to the list of papers it referenced.
FORWARD_CITATION_DB = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/citing_to_cited.db"

# --- 2. Data Fetching & Calculation Modules ---

class ReferenceFetcher:
    """
    Efficiently fetches paper references in batches from the SQLite database.
    """
    def __init__(self, db_path):
        print(f"Connecting to reference database: {db_path}")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Error: Reference database not found at {db_path}")
        try:
            # Connect in read-only mode for safety
            self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            self.cursor = self.conn.cursor()
            print("Reference database connection successful.")
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            exit()

    def get_references_for_papers(self, paper_ids: list[str]) -> dict[str, list]:
        """
        For a given list of paper IDs, fetches their referenced works in a single query.

        Args:
            paper_ids (list): A list of OpenAlex IDs (e.g., 'W2079062170').

        Returns:
            dict: A dictionary mapping each paper ID to its list of references.
                  If an ID is not found, it won't be in the dictionary.
        """
        if not paper_ids:
            return {}
        
        placeholders = ",".join("?" for _ in paper_ids)
        query = f"SELECT work_id, referenced_work_ids FROM citing_to_cited WHERE work_id IN ({placeholders})"
        
        try:
            self.cursor.execute(query, paper_ids)
            rows = self.cursor.fetchall()
            
            references_map = {}
            for work_id, refs_json in rows:
                if refs_json:
                    references_map[work_id] = json.loads(refs_json)
            return references_map
            
        except sqlite3.Error as e:
            print(f"[Warning] An error occurred while fetching references: {e}")
            return {}

    def close(self):
        if self.conn:
            self.conn.close()
            print("Reference database connection closed.")


class NoveltyQuerier:
    """
    Calculates novelty scores for papers based on the z-scores of their reference pairs.
    """
    def __init__(self, db_file_path):
        print(f"Initializing Novelty Querier with z-scores file: {db_file_path}")
        if not os.path.exists(db_file_path):
            raise FileNotFoundError(f"Error: Z-scores file not found at {db_file_path}")
        self.db_file_path = db_file_path
        self.con = duckdb.connect()
        print("DuckDB (Z-scores) connection successful.")

    def get_novelty_score(self, refs: list[str]):
        """
        Calculates the novelty score (p10_z) for a single paper's list of references.
        """
        if not refs or len(refs) < 2:
            return None
        
        # Clean IDs by removing URL prefix. The `if r` handles potential nulls in the list.
        cleaned_refs = {r.split('/')[-1] for r in refs if r and isinstance(r, str)}
        if len(cleaned_refs) < 2:
            return None
            
        query_pairs = tuple(sorted(tuple(sorted(p)) for p in itertools.combinations(cleaned_refs, 2)))
        
        if not query_pairs:
            return None

        query = f"""
            SELECT z_score FROM read_parquet('{self.db_file_path}')
            WHERE (id_min, id_max) IN {query_pairs};
        """
        try:
            result_df = self.con.execute(query).fetchdf()
            if result_df.empty:
                return None
            z_scores = result_df['z_score'].tolist()
        except Exception as e:
            print(f"[Warning] Database error during z-score query: {e}")
            return None
            
        if not z_scores:
            return None

        return np.percentile(z_scores, 10)

    def close(self):
        if self.con:
            self.con.close()
            print("DuckDB (Z-scores) connection closed.")


# --- 3. Main Evaluation Workflow ---

def main():
    print("\n--- Starting Novelty Evaluation of Recalled Papers ---")
    
    ref_fetcher = ReferenceFetcher(FORWARD_CITATION_DB)
    novelty_calculator = NoveltyQuerier(Z_SCORES_FILE)
    
    try:
        with open(RECALL_FILE_PATH, 'r', encoding='utf-8') as f:
            recall_data = json.load(f)
        print(f"Successfully loaded {len(recall_data)} queries from {RECALL_FILE_PATH}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse recall file. {e}")
        return

    all_query_avg_scores = []
    evaluation_results = {}

    for query, papers in tqdm(recall_data.items(), desc="Evaluating Queries"):
        
        # --- MODIFICATION ---
        # Filter the top 5 papers to only include those that have a valid, non-null string ID.
        top_5_papers = [p for p in papers[:5] if p.get('id') and isinstance(p.get('id'), str)]

        # If after filtering, no papers are left, skip this query.
        if not top_5_papers:
            evaluation_results[query] = None
            tqdm.write(f"[Info] Query '{query[:50]}...' skipped as no valid paper IDs were found in the top 5 results.")
            continue
        
        # Now, all subsequent operations are safe.
        paper_ids_full = [p['id'] for p in top_5_papers]
        paper_ids_short = [pid.split('/')[-1] for pid in paper_ids_full]
        
        references_map = ref_fetcher.get_references_for_papers(paper_ids_short)

        query_scores = []
        for paper in top_5_papers:
            # This is now safe because we filtered the list beforehand.
            paper_id_short = paper['id'].split('/')[-1]
            references = references_map.get(paper_id_short)
            
            if references:
                score = novelty_calculator.get_novelty_score(references)
                if score is not None:
                    query_scores.append(score)

        if query_scores:
            avg_novelty = np.mean(query_scores)
            all_query_avg_scores.append(avg_novelty)
            evaluation_results[query] = avg_novelty
        else:
            evaluation_results[query] = None

    ref_fetcher.close()
    novelty_calculator.close()

    # --- 4. Display Final Report ---
    print("\n--- Evaluation Report ---")
    for i, (query, avg_score) in enumerate(evaluation_results.items()):
        score_str = f"{avg_score:.4f}" if avg_score is not None else "N/A (Could not calculate)"
        print(f"{i+1}. Query: {query}\n   - Average Top-5 Novelty (p10_z): {score_str}")

    if all_query_avg_scores:
        overall_average = np.mean(all_query_avg_scores)
        print("\n" + "="*50)
        print(f"📊 Overall Average Novelty Score (p10_z): {overall_average:.4f}")
        print("="*50)
    else:
        print("\nCould not calculate an overall average novelty score.")

if __name__ == "__main__":
    main()