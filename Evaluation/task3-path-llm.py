import json
import os
import sqlite3
import ast
import requests
import time
from tqdm import tqdm
import numpy as np
import sys

# --- 1. Configuration ---
# Path to model recall results
RECALL_FILE_PATH = "results/answers_task3_paths.json"
# Output file path (update filename to match the model)
OUTPUT_FILE = "results/llm_path_evaluation_results.json"

# Database containing paper abstracts
WORKS_DB_PATH = "OpenAlex/sqlite/works.db"

# LLM API configuration
CUSTOM_API_ENDPOINT = "xxx"
CUSTOM_API_KEY = "xxx"
LLM_MODEL = "gpt-5"


# --- 2. Abstract Fetching Module ---
class AbstractFetcher:
    def __init__(self, db_path):
        print(f"Connecting to works database: {db_path}")
        if not os.path.exists(db_path): raise FileNotFoundError(f"Database not found at path {db_path}")
        try:
            self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            self.cursor = self.conn.cursor()
            print("Successfully connected to works database.")
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            sys.exit(1)

    def _reconstruct_abstract(self, abstract_inverted_index: dict) -> str:
        if not abstract_inverted_index: return ""
        try:
            max_pos = -1
            for positions in abstract_inverted_index.values():
                if positions:
                    max_pos = max(max_pos, max(positions))
            if max_pos == -1: return ""
            
            words = [None] * (max_pos + 1)
            for word, positions in abstract_inverted_index.items():
                if positions:
                    for pos in positions: words[pos] = word
            return " ".join(word for word in words if word is not None)
        except (ValueError, TypeError): return ""

    def get_details_for_papers(self, paper_ids: list[str]) -> dict:
        if not paper_ids: return {}
        details_map = {}
        placeholders = ",".join("?" for _ in paper_ids)
        query = f"SELECT id, title, abstract_inverted_index FROM works WHERE id IN ({placeholders})"
        try:
            self.cursor.execute(query, paper_ids)
            rows = self.cursor.fetchall()
            for full_id, title, abstract_str in rows:
                if not title: continue
                abstract = ""
                if abstract_str:
                    try:
                        abstract_index = ast.literal_eval(abstract_str)
                        abstract = self._reconstruct_abstract(abstract_index)
                    except (ValueError, SyntaxError): abstract = ""
                details_map[full_id] = {"title": title, "abstract": abstract}
        except sqlite3.Error as e: print(f"Database query failed: {e}")
        return details_map

    def close(self):
        if self.conn:
            self.conn.close()
            print("Works database connection closed.")


# --- 3. LLM Path Evaluation Module (Optimized Prompt) ---
class LLMPathScorer:
    def __init__(self, api_key, api_endpoint, model):
        self.api_key = api_key
        self.url = f"{api_endpoint}/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _create_prompt(self, original_query: str, path_details: list) -> list:
        system_prompt = (
            "You are an expert in scientometrics and academic research. Your task is to evaluate the quality of a proposed citation path based on its technical evolution.\n\n"
            "### Core Task:\n"
            "Assess if the provided sequence of papers represents a logical and meaningful technological or conceptual evolution from the start paper to the end paper.\n\n"
            "### What constitutes a good technical evolution path? (Key Principles)\n"
            "1.  **Thematic Consistency**: All papers must strictly revolve around the same core research topic defined by the query. Deviations into unrelated subjects indicate a poor path.\n"
            "2.  **Content Cohesion & Logical Flow**: The content of adjacent papers must be closely related. Each paper should logically follow from the previous one, building upon its ideas, refining its methods, or addressing its limitations.\n"
            "3.  **Progressive Development**: The path must demonstrate clear progress. Later papers should represent advancements, extensions, or significant new applications of the concepts introduced in earlier papers. The path should tell a story of innovation.\n"
            "4.  **Represents a Main Line of Inquiry**: The path should follow a significant and recognized line of development within the research field, not an obscure or tangential branch.\n\n"
            "### Scoring Criteria (0-10):\n"
            "- **Score 9-10 (Excellent)**: A perfect or near-perfect path. It is thematically consistent, shows clear progressive development, and represents a major line of inquiry. The logical flow is impeccable.\n"
            "- **Score 7-8 (Good)**: A strong, coherent path. Most papers are relevant and show progression, but there might be a minor logical gap or a less influential paper included.\n"
            "- **Score 4-6 (Mediocre)**: The path has some relevance but lacks strong cohesion. It may include several tangential papers, the logical progression is weak, or it fails to capture the main developmental thread.\n"
            "- **Score 1-3 (Poor)**: The path is largely incoherent. Papers are thematically disconnected, show no clear progress, or are mostly irrelevant to the query.\n"
            "- **Score 0 (Failure)**: A completely random collection of papers with no logical or thematic connection."
        )

        path_str = ""
        for i, paper in enumerate(path_details, 1):
            title = paper.get('title', '[Title not found]')
            abstract = paper.get('abstract', '[Abstract not provided]')
            path_str += f"### Paper {i}:\n**Title**: {title}\n**Abstract**: {abstract}\n\n"

        user_prompt = (
            f"Please evaluate the following citation path based on the detailed criteria provided.\n\n"
            f"**Original Request**: {original_query}\n\n"
            f"--- Proposed Citation Path ---\n{path_str}"
            f"------------------------------\n\n"
            f"Your response MUST be a single JSON object with 'score' (an integer from 0 to 10) and 'reasoning' (a detailed explanation for your score, critiquing the path based on the four key principles)."
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def get_path_score(self, original_query: str, path_details: list, retries=3, delay=5):
        messages = self._create_prompt(original_query, path_details)
        data = {"model": self.model, "messages": messages, "temperature": 0.2}
        for attempt in range(retries):
            try:
                response = requests.post(self.url, headers=self.headers, json=data, timeout=180)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                json_part = content[content.find('{'):content.rfind('}')+1]
                parsed_json = json.loads(json_part)
                score = parsed_json.get("score")
                if isinstance(score, int) and 0 <= score <= 10:
                    return score, parsed_json.get("reasoning", "")
                return None, f"Received an invalid score: {score}"
            except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"[Warning] LLM call attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1: time.sleep(delay)
        return None, "Failed to obtain a valid score after multiple retries."


# --- 4. Main Evaluation Workflow ---
def main():
    print("\n--- Starting LLM-based citation path evaluation (Optimized Prompt) ---")
    fetcher = AbstractFetcher(WORKS_DB_PATH)
    scorer = LLMPathScorer(CUSTOM_API_KEY, CUSTOM_API_ENDPOINT, LLM_MODEL)
    
    with open(RECALL_FILE_PATH, 'r', encoding='utf-8') as f:
        recall_data = json.load(f)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resuming analysis. Loaded {len(results)} completed queries from {OUTPUT_FILE}.")
    else:
        results = {}

    all_query_scores = [res['llm_path_score'] for res in results.values() if res.get('llm_path_score') is not None]

    try:
        for query, papers in tqdm(recall_data.items(), desc="Evaluating citation paths"):
            if query in results:
                continue

            top_5_papers_recalled = papers[:5]
            paper_ids_full = [p['id'] for p in top_5_papers_recalled]
            
            paper_details_map = fetcher.get_details_for_papers(paper_ids_full)
            
            path_details_ordered = []
            for p in top_5_papers_recalled:
                details = paper_details_map.get(p['id'], {"title": p.get('title'), "abstract": "[Details not found in DB]"})
                path_details_ordered.append(details)

            score, reason = scorer.get_path_score(query, path_details_ordered)

            results[query] = {
                "llm_path_score": score,
                "llm_path_reasoning": reason,
                "evaluated_path": [{"id": p['id'], "title": p.get('title')} for p in top_5_papers_recalled]
            }
            if score is not None:
                all_query_scores.append(score)

            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    finally:
        fetcher.close()

    print("\n--- Final Path Evaluation Report ---")
    if all_query_scores:
        overall_average = np.mean(all_query_scores)
        print("\n" + "="*50)
        print(f"📊 Average LLM score across all paths: {overall_average:.4f}")
        print("="*50)
    else:
        print("\nUnable to compute final average score.")
    print(f"Detailed path evaluation results have been saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
