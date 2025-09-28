```python
import os
import requests
import arxiv
import fitz  # PyMuPDF
import json
import re
from tqdm import tqdm
import sys
from thefuzz import fuzz
from lxml import etree
import time
import sqlite3

# --- 1. Configuration Section ---

RESULTS_FILE_PATH = "results/answers_task2_sentiment.json"

QUERIES_FILE_PATH = "queries/queries_task2_sentiment.json"

# --- New: Output log file ---
LOG_DIR = "results/contextlogs"
os.makedirs(LOG_DIR, exist_ok=True)

# Extract the first-level directory name after "results/"
results_dirname = os.path.basename(
    os.path.dirname(
        os.path.dirname(RESULTS_FILE_PATH)  # Two levels up -> parent of papers_oaids
    )
)
# Construct the log filename
OUTPUT_LOG_FILE = os.path.join(LOG_DIR, f"evaluation_log_{results_dirname}.log")

# --- Database & resource paths ---
FORWARD_DB_PATH = "OpenAlex/sqlite/citing_to_cited.db"
ARXIV_LOCAL_ROOT = "Arxiv_Papers/Arxiv"
OUTPUT_PDF_DIR = "results/temp_pdfs"
os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)

# --- Service & API configuration ---
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
# --- Change 1: Add GROBID timeout configuration ---
GROBID_TIMEOUT_SECONDS = 240  # Increase timeout from 60s to 240s
LLM_API_ENDPOINT = "xxx"
LLM_API_KEY = "xxx"
LLM_MODEL = "gpt-5"

# --- Parsing parameters ---
SIMILARITY_THRESHOLD = 85
GROBID_TITLE_MATCH_THRESHOLD = 90

# --- GROBID XML parsing namespace ---
TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
NS_MAP = {'tei': TEI_NAMESPACE}


# --- 2. Core functional modules (class encapsulation) ---

class PDFManager:
    """Handle PDF retrieval, preferring local files and falling back to download."""
    def __init__(self, local_root, download_dir):
        self.local_root = local_root
        self.download_dir = download_dir

    def _get_local_path(self, arxiv_id: str) -> str | None:
        if not arxiv_id: return None
        arxiv_id = arxiv_id.strip()
        if '/' in arxiv_id:
            try:
                archive, id_part = arxiv_id.split('/')
                archive_clean = archive.replace('-', '')
                year_month = id_part[:4]
                return os.path.join(self.local_root, archive_clean, 'pdf', year_month, f"{id_part}.pdf")
            except (ValueError, IndexError): return None
        else:
            try:
                year_month = arxiv_id.split('.')[0]
                return os.path.join(self.local_root, 'arxiv', 'pdf', year_month, f"{arxiv_id}.pdf")
            except IndexError: return None

    def get_pdf(self, title: str) -> str | None:
        try:
            search = arxiv.Search(query=f'ti:"{title}"', max_results=1, sort_by=arxiv.SortCriterion.Relevance)
            result = next(search.results(), None)
            if result and fuzz.token_set_ratio(title.lower(), result.title.lower()) >= SIMILARITY_THRESHOLD:
                arxiv_id = result.get_short_id()
                local_path = self._get_local_path(arxiv_id)
                if local_path and os.path.exists(local_path):
                    return local_path
                else:
                    filename = f"{arxiv_id.replace('/', '_')}v1.pdf"
                    download_path = os.path.join(self.download_dir, filename)
                    if not os.path.exists(download_path):
                        result.download_pdf(dirpath=self.download_dir, filename=filename)
                    return download_path
        except Exception as e:
            tqdm.write(f"[PDFManager Error] Error processing '{title[:30]}...': {e}")
        return None

class ContextExtractor:
    """Use GROBID to extract citation contexts from PDFs."""
    # --- Change 2: Update constructor to accept timeout parameter ---
    def __init__(self, grobid_url, timeout_seconds):
        self.grobid_url = grobid_url
        self.timeout = timeout_seconds  # store timeout as instance variable
        self.parser = etree.XMLParser(recover=True)

    def get_citation_contexts(self, pdf_path: str, target_reference_title: str) -> list[str]:
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
                # --- Change 3: use configured timeout in request ---
                response = requests.post(self.grobid_url, files=files, timeout=self.timeout)
            if response.status_code != 200: return []
            root = etree.fromstring(response.content, self.parser)
        except Exception as e:
            tqdm.write(f"[GROBID Error] Failed to call GROBID: {e}")
            return []

        bibl_structs = root.xpath('//tei:listBibl/tei:biblStruct', namespaces=NS_MAP)
        best_score, best_match_id = 0, None
        for bibl in bibl_structs:
            titles = bibl.xpath('.//tei:title', namespaces=NS_MAP)
            for title_element in titles:
                current_title = "".join(title_element.itertext()).strip()
                score = fuzz.token_set_ratio(target_reference_title, current_title)
                if score > best_score:
                    best_score, best_match_id = score, bibl.get('{http://www.w3.org/XML/1998/namespace}id')
        
        if best_score < GROBID_TITLE_MATCH_THRESHOLD: return []
        
        ref_elements = root.xpath(f'//tei:ref[@target="#{best_match_id}"]', namespaces=NS_MAP)
        contexts = set()
        for ref in ref_elements:
            parent_paragraph = ref.xpath('ancestor::tei:p', namespaces=NS_MAP)
            if parent_paragraph:
                p_text = re.sub(r'\s+', ' ', "".join(parent_paragraph[0].itertext()).strip())
                contexts.add(p_text)
        return list(contexts)

class SentimentAnalyzer:
    """Use LLM to analyze sentiment of citation contexts."""
    def __init__(self, api_key, api_endpoint, model):
        self.url = f"{api_endpoint}/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        self.system_prompt = "You are an expert in academic literature analysis. Your task is to classify the sentiment of a citation context."

    def get_sentiment(self, context: str, target_title: str) -> str | None:
        user_prompt = f"""
Please analyze the following citation context from a research paper, which mentions the target paper titled "{target_title}".
Classify the context into one of three categories:
- **Positive**: The citing paper praises, builds upon, or confirms the findings of the target paper.
- **Negative**: The citing paper criticizes, questions, or points out limitations of the target paper.
- **Neutral**: The citing paper simply mentions or describes the target paper as background information without expressing a strong opinion.
Your response MUST BE only ONE of the three category names: Positive, Negative, or Neutral.

Context to analyze:
---
{context}
---
"""
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        data = {"model": self.model, "messages": messages, "temperature": 0.1}
        try:
            response = requests.post(self.url, headers=self.headers, json=data, timeout=60)
            response.raise_for_status()
            label = response.json()["choices"][0]["message"]["content"].strip()
            if label in ["Positive", "Negative", "Neutral"]:
                return label
        except Exception as e:
            tqdm.write(f"[LLM Error] API call failed: {e}")
        return None

# --- 3. Helper functions ---
def extract_title_from_query(query: str) -> str | None:
    match = re.search(r'titled "(.*?)"', query)
    return match.group(1) if match else None

# --- 4. Main evaluation logic ---
def main():
    print(f"--- Starting evaluation of positive citation recall quality (logs will be saved to: {OUTPUT_LOG_FILE}) ---")
    if not all(os.path.exists(p) for p in [QUERIES_FILE_PATH, RESULTS_FILE_PATH, FORWARD_DB_PATH]):
        print("Error: One or more required files not found. Please check path configuration.")
        return

    pdf_manager = PDFManager(ARXIV_LOCAL_ROOT, OUTPUT_PDF_DIR)
    # --- Change 4: pass timeout parameter on initialization ---
    context_extractor = ContextExtractor(GROBID_URL, GROBID_TIMEOUT_SECONDS)
    sentiment_analyzer = SentimentAnalyzer(LLM_API_KEY, LLM_API_ENDPOINT, LLM_MODEL)
    
    with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f: queries_data = json.load(f)
    with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f: results_data = json.load(f)

    if os.path.exists(OUTPUT_LOG_FILE):
        try:
            with open(OUTPUT_LOG_FILE, 'r', encoding='utf-8') as f:
                results_log = json.load(f)
            tqdm.write(f"Successfully loaded {len(results_log)} existing results from log file.")
        except (json.JSONDecodeError, IOError):
            results_log = {}
            tqdm.write("Warning: Log file exists but cannot be parsed. A new log will be created.")
    else:
        results_log = {}

    conn = sqlite3.connect(f"file:{FORWARD_DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        for source_paper_id_full, query_text in tqdm(queries_data.items(), desc="Evaluating all queries"):
            if query_text in results_log:
                continue

            target_title = extract_title_from_query(query_text)
            if not target_title: continue

            retrieved_papers = results_data.get(query_text, [])
            analyzed_papers_for_query = []
            
            for paper in tqdm(retrieved_papers, desc=f"Processing query '{target_title[:20]}...'", leave=False):
                retrieved_id_full = paper.get("id")
                retrieved_title = paper.get("title")
                
                paper_analysis_log = {
                    "id": retrieved_id_full, "title": retrieved_title,
                    "cited_target": False, "pdf_found": False,
                    "contexts": [], "final_positive_score": 0
                }

                if not retrieved_id_full or not retrieved_title:
                    analyzed_papers_for_query.append(paper_analysis_log)
                    continue

                retrieved_id_short = retrieved_id_full.split('/')[-1]
                cursor.execute("SELECT referenced_work_ids FROM citing_to_cited WHERE work_id = ?", (retrieved_id_short,))
                row = cursor.fetchone()
                references = json.loads(row[0]) if row and row[0] else []
                
                source_paper_id_short = source_paper_id_full.split('/')[-1]
                if source_paper_id_short not in references:
                    analyzed_papers_for_query.append(paper_analysis_log)
                    continue
                
                paper_analysis_log["cited_target"] = True

                pdf_path = pdf_manager.get_pdf(retrieved_title)
                if not pdf_path:
                    analyzed_papers_for_query.append(paper_analysis_log)
                    continue
                
                paper_analysis_log["pdf_found"] = True
                
                contexts_text = context_extractor.get_citation_contexts(pdf_path, target_title)
                if not contexts_text:
                    analyzed_papers_for_query.append(paper_analysis_log)
                    continue

                is_positive = False
                for context in contexts_text:
                    sentiment = sentiment_analyzer.get_sentiment(context, target_title)
                    paper_analysis_log["contexts"].append({"text": context, "sentiment": sentiment})
                    if sentiment == "Positive":
                        is_positive = True
                
                if is_positive:
                    paper_analysis_log["final_positive_score"] = 1
                
                analyzed_papers_for_query.append(paper_analysis_log)

            results_log[query_text] = {
                "target_paper_id": source_paper_id_full,
                "analyzed_papers": analyzed_papers_for_query
            }
            with open(OUTPUT_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(results_log, f, indent=4, ensure_ascii=False)

    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    if not results_log:
        print("No queries were evaluated.")
        return

    total_retrieved_papers = 0
    total_simple_cite_papers = 0
    total_positive_cite_papers = 0
    query_simple_accuracies = []
    query_positive_accuracies = []

    for query, result in results_log.items():
        analyzed_papers = result["analyzed_papers"]
        num_retrieved = len(analyzed_papers)
        if num_retrieved == 0: continue

        total_retrieved_papers += num_retrieved
        
        simple_cites = sum(1 for p in analyzed_papers if p["cited_target"])
        positive_cites = sum(p["final_positive_score"] for p in analyzed_papers)

        total_simple_cite_papers += simple_cites
        total_positive_cite_papers += positive_cites
        
        query_simple_accuracies.append(simple_cites / num_retrieved)
        query_positive_accuracies.append(positive_cites / num_retrieved)

    if total_retrieved_papers > 0:
        micro_avg_simple = total_simple_cite_papers / total_retrieved_papers
        macro_avg_simple = sum(query_simple_accuracies) / len(query_simple_accuracies)
        
        micro_avg_positive = total_positive_cite_papers / total_retrieved_papers
        macro_avg_positive = sum(query_positive_accuracies) / len(query_positive_accuracies)

        print("\n--- Final Evaluation Report (generated from log file) ---")
        print(f"Total queries evaluated: {len(results_log)}")
        print(f"Total retrieved papers: {total_retrieved_papers}")
        
        print("\n--- Metric 1: Simple citation accuracy (Structural Check) ---")
        print("Measures whether retrieved papers actually cited the target paper")
        print(f"Total citation count: {total_simple_cite_papers}")
        print(f"Macro-average accuracy: {macro_avg_simple:.2%}")
        print(f"Micro-average accuracy: {micro_avg_simple:.2%}")
        
        print("\n--- Metric 2: Positive citation accuracy (Semantic Check) ---")
        print("Measures whether retrieved papers cited the target paper positively")
        print(f"Total positive citation count: {total_positive_cite_papers}")
        print(f"Macro-average accuracy: {macro_avg_positive:.2%}")
        print(f"Micro-average accuracy: {micro_avg_positive:.2%}")
        print("-" * 45)

if __name__ == "__main__":
    main()
```
