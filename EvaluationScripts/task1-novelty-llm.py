# evaluate_novelty.py
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
RECALL_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/pasa/oaids/queries_task1_novel.json"
OUTPUT_FILE = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/EvaluationScripts/logs/pasa_llm_novelty_evaluation_results.json"


WORKS_DB_PATH = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/works.db"
CUSTOM_API_ENDPOINT = "http://35.220.164.252:3888/v1"
CUSTOM_API_KEY = "sk-B52cka26mugEd4P3EEDyIvMU2jlEabH37wuHz30KNy7825SZ"
LLM_MODEL = "gpt-5-mini-2025-08-07"



# --- 2. Abstract Fetcher Module ---
class AbstractFetcher:
    def __init__(self, db_path):
        print(f"Connecting to works database: {db_path}")
        if not os.path.exists(db_path): raise FileNotFoundError(f"Database not found at {db_path}")
        try:
            self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            self.cursor = self.conn.cursor()
            print("Works database connection successful.")
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            sys.exit(1)

    def _reconstruct_abstract(self, abstract_inverted_index: dict) -> str:
        if not abstract_inverted_index: return ""
        try:
            total_len = max(pos for positions in abstract_inverted_index.values() if positions for pos in positions) + 1
            words = [None] * total_len
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
            for pid, title, abstract_str in rows:
                if not title: continue
                abstract = ""
                if abstract_str:
                    try:
                        abstract_index = ast.literal_eval(abstract_str)
                        abstract = self._reconstruct_abstract(abstract_index)
                    except (ValueError, SyntaxError): abstract = ""
                details_map[pid] = {"title": title, "abstract": abstract}
        except sqlite3.Error as e: print(f"Database query failed: {e}")
        return details_map

    def close(self):
        if self.conn:
            self.conn.close()
            print("Works database connection closed.")


# --- 3. LLM Scorer Module for Novelty ---
class LLMScorer:
    def __init__(self, api_key, api_endpoint, model):
        self.api_key = api_key
        self.url = f"{api_endpoint}/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _create_prompt(self, title: str, abstract: str) -> list:
        system_prompt = (
            "You are an expert academic reviewer. Your task is to evaluate a scientific paper on its Novelty.\n\n"
            "### Definition of Novelty:\n"
            "**Definition**: Novelty refers to the uniqueness and originality of the research question, methodology, data, or conclusions relative to existing research.\n"
            "**Focus**: Does the paper introduce new ideas, perspectives, or methods within the existing body of knowledge? For example, applying a method from Field A to Field B for the first time.\n"
            "**Scoring Criteria**: A score of 0 represents completely derivative work, while a score of 10 represents a highly original and groundbreaking idea."
        )
        if abstract:
            paper_info = f"**Title**: {title}\n\n**Abstract**: {abstract}"
        else:
            paper_info = f"**Title**: {title}\n\n**Abstract**: [No abstract provided. Please evaluate based on the title alone.]"
        
        user_prompt = (
            f"Please evaluate the Novelty of the following paper based on the definition provided.\n\n"
            f"{paper_info}\n\n"
            f"Your response MUST be a single JSON object with 'score' (an integer from 0 to 10) and 'reasoning' (a brief explanation)."
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def get_novelty_score(self, title: str, abstract: str, retries=3, delay=5):
        messages = self._create_prompt(title, abstract)
        data = {"model": self.model, "messages": messages, "temperature": 0.2}
        for attempt in range(retries):
            try:
                response = requests.post(self.url, headers=self.headers, json=data, timeout=60)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                json_part = content[content.find('{'):content.rfind('}')+1]
                parsed_json = json.loads(json_part)
                score = parsed_json.get("score")
                if isinstance(score, int) and 0 <= score <= 10:
                    return score, parsed_json.get("reasoning", "")
                return None, f"Invalid score value received: {score}"
            except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"[Warning] LLM call attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1: time.sleep(delay)
        return None, "Failed to get a valid score after multiple retries."


# --- 4. Main Evaluation Workflow ---
def main():
    print("\n--- Starting LLM-Based Novelty Evaluation ---")
    fetcher = AbstractFetcher(WORKS_DB_PATH)
    scorer = LLMScorer(CUSTOM_API_KEY, CUSTOM_API_ENDPOINT, LLM_MODEL)
    with open(RECALL_FILE_PATH, 'r', encoding='utf-8') as f:
        recall_data = json.load(f)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resuming analysis. Loaded {len(results)} completed queries from {OUTPUT_FILE}.")
    else:
        results = {}
    all_query_avg_scores = []
    for query, papers in tqdm(recall_data.items(), desc="Evaluating Queries for Novelty"):
        if query in results:
            if results[query].get("average_novelty_score") is not None:
                all_query_avg_scores.append(results[query]["average_novelty_score"])
            continue
        top_5_papers = papers[:5]
        paper_ids_full = [p['id'] for p in top_5_papers]
        paper_details = fetcher.get_details_for_papers(paper_ids_full)
        query_scores = []
        scored_papers_details = []
        for paper in top_5_papers:
            pid = paper['id']
            details = paper_details.get(pid)
            score, reason = None, "Paper details not found in DB."
            if details:
                score, reason = scorer.get_novelty_score(details['title'], details['abstract'])
            scored_papers_details.append({
                "id": pid,
                "title": paper.get('title'),
                "llm_novelty_score": score,
                "llm_novelty_reasoning": reason
            })
            if score is not None: query_scores.append(score)
        avg_score = np.mean(query_scores) if query_scores else None
        results[query] = {"average_novelty_score": avg_score, "scored_papers": scored_papers_details}
        if avg_score is not None: all_query_avg_scores.append(avg_score)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    fetcher.close()
    print("\n--- Final Novelty Evaluation Report ---")
    if all_query_avg_scores:
        overall_average = np.mean(all_query_avg_scores)
        print("\n" + "="*50)
        print(f"📊 Overall Average LLM Novelty Score: {overall_average:.4f}")
        print("="*50)
    else:
        print("\nCould not calculate an overall average novelty score.")
    print(f"Detailed novelty results have been saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()