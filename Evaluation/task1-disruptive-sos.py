import json
import os
import sqlite3
from tqdm import tqdm
import numpy as np

# --- 1. Configuration ---

# Input file containing the recalled papers
RECALL_FILE_PATH = "results/queries_task1_disruptive.json"

# SQLite database files required for the calculation
FORWARD_CITATION_DB = "/OpenAlex/sqlite/citing_to_cited.db"
REVERSE_CITATION_DB = "/OpenAlex/sqlite/cited_to_citing.db"

# --- MODIFIED: Added an output file path for the results ---
OUTPUT_FILE = "results/paperqa_sos_disruption_evaluation_results.json"

# --- 2. Optimized Data Fetching & Calculation Modules ---

class DisruptionCalculator:
    """
    An efficient calculator for the Disruption Index.
    It pre-connects to databases and uses batch fetching for high performance.
    """
    def __init__(self, forward_db_path, reverse_db_path):
        print("Connecting to citation databases...")
        try:
            self.conn_forward = sqlite3.connect(f"file:{forward_db_path}?mode=ro", uri=True)
            self.conn_reverse = sqlite3.connect(f"file:{reverse_db_path}?mode=ro", uri=True)
            self.cur_forward = self.conn_forward.cursor()
            self.cur_reverse = self.conn_reverse.cursor()
            print("Database connections successful.")
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            exit()

    def _get_references(self, paper_id: str) -> set:
        self.cur_forward.execute("SELECT referenced_work_ids FROM citing_to_cited WHERE work_id = ?", (paper_id,))
        row = self.cur_forward.fetchone()
        return set(json.loads(row[0])) if row and row[0] else set()

    def _get_citations(self, paper_id: str) -> list:
        self.cur_reverse.execute("SELECT citing_work_ids FROM cited_to_citing WHERE referenced_work_id = ?", (paper_id,))
        row = self.cur_reverse.fetchone()
        return json.loads(row[0]) if row and row[0] else []

    def _batch_get_references(self, paper_ids: list) -> dict:
        if not paper_ids:
            return {}
        references_map = {}
        placeholders = ",".join("?" for _ in paper_ids)
        query = f"SELECT work_id, referenced_work_ids FROM citing_to_cited WHERE work_id IN ({placeholders})"
        self.cur_forward.execute(query, paper_ids)
        for work_id, refs_json in self.cur_forward.fetchall():
            if refs_json:
                references_map[work_id] = set(json.loads(refs_json))
        return references_map

    def compute_disruption_index(self, work_id: str) -> float | None:
        short_id = work_id.split('/')[-1]
        focal_refs = self._get_references(short_id)
        if not focal_refs:
            return None
        citing_ids = self._get_citations(short_id)
        if not citing_ids:
            return None
        citing_papers_refs_map = self._batch_get_references(citing_ids)
        only_cite_focal = 0
        cite_both = 0
        for citer_id in citing_ids:
            citer_refs = citing_papers_refs_map.get(citer_id, set())
            if focal_refs.intersection(citer_refs):
                cite_both += 1
            else:
                only_cite_focal += 1
        denominator = only_cite_focal + cite_both
        if denominator == 0:
            return None
        disruption = (only_cite_focal - cite_both) / denominator
        return disruption

    def close_connections(self):
        if self.conn_forward: self.conn_forward.close()
        if self.conn_reverse: self.conn_reverse.close()
        print("\nDatabase connections closed.")


# --- 3. Main Evaluation Workflow (MODIFIED to save detailed results) ---

def main():
    print("\n--- Starting Disruption Evaluation of Recalled Papers ---")
    
    calculator = DisruptionCalculator(FORWARD_CITATION_DB, REVERSE_CITATION_DB)
    
    try:
        with open(RECALL_FILE_PATH, 'r', encoding='utf-8') as f:
            recall_data = json.load(f)
        print(f"Successfully loaded {len(recall_data)} queries from {RECALL_FILE_PATH}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: Could not load or parse recall file. {e}")
        calculator.close_connections()
        return

    # --- MODIFIED: Checkpointing logic to resume from the output file ---
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resuming analysis. Loaded {len(results)} completed queries from {OUTPUT_FILE}.")
    else:
        results = {}

    all_query_avg_scores = []

    for query, papers in tqdm(recall_data.items(), desc="Evaluating Queries"):
        # --- MODIFIED: Skip queries that are already in the results file ---
        if query in results:
            if results[query].get("average_disruption_score") is not None:
                all_query_avg_scores.append(results[query]["average_disruption_score"])
            continue

        top_5_papers = papers[:5]
        
        query_scores = []
        scored_papers_details = [] # List to store detailed results for each paper

        for paper in top_5_papers:
            paper_id = paper.get('id')
            if not paper_id:
                continue
            
            score = calculator.compute_disruption_index(paper_id)
            
            # --- MODIFIED: Store detailed info for each paper ---
            scored_papers_details.append({
                "id": paper_id,
                "title": paper.get("title"),
                "disruption_score": score
            })
            
            if score is not None:
                query_scores.append(score)
        
        avg_disruption = np.mean(query_scores) if query_scores else None
        
        # --- MODIFIED: Store results in the desired structure ---
        results[query] = {
            "average_disruption_score": avg_disruption,
            "scored_papers": scored_papers_details
        }
        
        if avg_disruption is not None:
            all_query_avg_scores.append(avg_disruption)

        # --- MODIFIED: Save progress to the file after each query ---
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    calculator.close_connections()

    # --- 4. Display Final Report ---
    print("\n--- Disruption Evaluation Report ---")
    
    if all_query_avg_scores:
        overall_average = np.mean(all_query_avg_scores)
        print("\n" + "="*50)
        print(f"💣 Overall Average Disruption Score: {overall_average:.4f}")
        print("="*50)
    else:
        print("\nCould not calculate an overall average disruption score.")
    
    print(f"Detailed results have been saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()