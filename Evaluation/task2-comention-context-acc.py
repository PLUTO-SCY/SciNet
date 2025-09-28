import os
import requests
import arxiv
import fitz
import json
import re
from tqdm import tqdm
import sys
from thefuzz import fuzz
from lxml import etree
import time
import sqlite3

# --- 1. Configuration Section ---

# --- Specify files ---
# Note: Your task is co-occur. Ensure you are using the corresponding query and result files.
QUERIES_FILE_PATH = "Queries/queries_task2_comention.json"
RESULTS_FILE_PATH = "results/answers_task2_comention.json"

# --- New: Output log file ---
LOG_DIR = "results/cocitelogs"
os.makedirs(LOG_DIR, exist_ok=True)
RESULTS_FILENAME = os.path.basename(RESULTS_FILE_PATH)
OUTPUT_LOG_FILE = os.path.join(LOG_DIR, f"evaluation_log_o4-mini-deep-research-2025-06-26")

# --- Database and resource paths ---
# New: Reverse citation database (who cites me)
REVERSE_DB_PATH = "OpenAlex/sqlite/cited_to_citing.db"
ARXIV_LOCAL_ROOT = "Arxiv_Papers/Arxiv"
OUTPUT_PDF_DIR = "results/temp_pdfs"
os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)

# --- Services and API configuration ---
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# --- Parsing parameters ---
SIMILARITY_THRESHOLD = 85
GROBID_TITLE_MATCH_THRESHOLD = 85

# --- GROBID XML parsing namespace ---
TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
NS_MAP = {'tei': TEI_NAMESPACE}


# --- 2. Core Functional Modules (Class Wrappers) ---

class CiterFetcher:
    """Fetch the list of citers from the reverse citation database."""
    def __init__(self, db_path):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.cursor = self.conn.cursor()

    def get_citers(self, paper_id_short: str) -> set:
        self.cursor.execute(
            "SELECT citing_work_ids FROM cited_to_citing WHERE referenced_work_id = ?", 
            (paper_id_short,)
        )
        row = self.cursor.fetchone()
        return set(json.loads(row[0])) if row and row[0] else set()

    def close(self):
        self.conn.close()


class PDFManager:
    """Manage PDF access, local or via arXiv download."""
    # ... This class is the same as before; shortened here for brevity ...
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
                print('local_path: ', local_path)
                if local_path and os.path.exists(local_path):
                    return local_path
                else:
                    filename = f"{arxiv_id.replace('/', '_')}v1.pdf"
                    download_path = os.path.join(self.download_dir, filename)
                    if not os.path.exists(download_path):
                        result.download_pdf(dirpath=self.download_dir, filename=filename)
                    return download_path
        except Exception as e:
            tqdm.write(f"[PDFManager Error] Failed to process '{title[:30]}...': {e}")
        return None


class CoCitationAnalyzer:
    """Use GROBID to analyze PDFs and check whether two target papers are cited in the same paragraph."""
    def __init__(self, grobid_url):
        self.grobid_url = grobid_url
        self.parser = etree.XMLParser(recover=True)

    def _get_ref_id_from_title(self, root, title_to_match):
        bibl_structs = root.xpath('//tei:listBibl/tei:biblStruct', namespaces=NS_MAP)
        best_score, best_match_id = 0, None
        for bibl in bibl_structs:
            titles = bibl.xpath('.//tei:title', namespaces=NS_MAP)
            for title_element in titles:
                current_title = "".join(title_element.itertext()).strip()
                score = fuzz.token_set_ratio(title_to_match, current_title)
                if score > best_score:
                    best_score, best_match_id = score, bibl.get('{http://www.w3.org/XML/1998/namespace}id')
        return best_match_id if best_score >= GROBID_TITLE_MATCH_THRESHOLD else None

    def check_contextual_co_citation(self, pdf_path: str, title_A: str, title_B: str) -> bool:
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
                response = requests.post(self.grobid_url, files=files, timeout=60)
            if response.status_code != 200: return False
            root = etree.fromstring(response.content, self.parser)
        except Exception as e:
            tqdm.write(f"[GROBID Error] Failed to call GROBID: {e}")
            return False

        # 1. Find the reference IDs of A and B in the bibliography
        ref_id_A = self._get_ref_id_from_title(root, title_A)
        ref_id_B = self._get_ref_id_from_title(root, title_B)

        if not ref_id_A or not ref_id_B:
            return False  # If either paper is not in references, cannot determine

        # 2. Iterate over all paragraphs in the main text
        paragraphs = root.xpath('//tei:body//tei:p', namespaces=NS_MAP)
        for p in paragraphs:
            # 3. Collect all reference targets in the paragraph
            refs_in_p = {ref.get('target').replace('#', '') for ref in p.xpath('.//tei:ref[@target]', namespaces=NS_MAP)}
            # 4. Check if both A and B IDs are in the paragraph references
            if ref_id_A in refs_in_p and ref_id_B in refs_in_p:
                return True  # Found!

        return False  # None found


# --- 3. Helper Functions ---
def extract_title_from_query(query: str) -> str | None:
    match = re.search(r'titled "(.*?)"', query)
    return match.group(1) if match else None


# --- 4. Main Evaluation Logic ---
def main():
    print(f"--- Starting co-citation context evaluation (logs will be saved to: {OUTPUT_LOG_FILE}) ---")
    pdf_manager = PDFManager(ARXIV_LOCAL_ROOT, OUTPUT_PDF_DIR)
    analyzer = CoCitationAnalyzer(GROBID_URL)
    
    with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f: queries_data = json.load(f)
    with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f: results_data = json.load(f)

    if os.path.exists(OUTPUT_LOG_FILE):
        with open(OUTPUT_LOG_FILE, 'r', encoding='utf-8') as f: results_log = json.load(f)
    else: results_log = {}

    citer_fetcher = CiterFetcher(REVERSE_DB_PATH)

    try:
        for target_paper_id_full, query_text in tqdm(queries_data.items(), desc="Evaluating all queries"):
            if query_text in results_log: continue
            
            target_title = extract_title_from_query(query_text)
            if not target_title: continue
            
            recalled_papers = results_data.get(query_text, [])
            analyzed_papers_for_query = []
            
            target_id_short = target_paper_id_full.split('/')[-1]
            citers_of_target = citer_fetcher.get_citers(target_id_short)

            for paper in tqdm(recalled_papers, desc=f"Processing query '{target_title[:20]}...'", leave=False):
                recalled_id_full = paper.get("id")
                recalled_title = paper.get("title")
                
                paper_log = {
                    "id": recalled_id_full, "title": recalled_title,
                    "is_co_cited": False, "found_contextual_co_citation": False,
                    "common_citers_checked": []
                }

                if not recalled_id_full or not recalled_title:
                    analyzed_papers_for_query.append(paper_log)
                    continue

                # Step 1: Find common citers
                recalled_id_short = recalled_id_full.split('/')[-1]
                citers_of_recalled = citer_fetcher.get_citers(recalled_id_short)
                common_citers = citers_of_target.intersection(citers_of_recalled)
                
                if not common_citers:
                    analyzed_papers_for_query.append(paper_log)
                    continue
                
                paper_log["is_co_cited"] = True

                # Step 2: Verify co-citation context
                for citer_id in common_citers:
                    # Note: citer_id is a full URL but has no title, so PDF access is limited.
                    # Placeholder: we assume we can get the PDF somehow.
                    citer_title = f"Paper with ID {citer_id}"  # Placeholder
                    
                    pdf_path = pdf_manager.get_pdf(citer_title)
                    paper_log["common_citers_checked"].append({"id": citer_id, "pdf_found": bool(pdf_path)})
                    
                    if pdf_path:
                        if analyzer.check_contextual_co_citation(pdf_path, target_title, recalled_title):
                            paper_log["found_contextual_co_citation"] = True
                            break  # One valid context is enough
                
                analyzed_papers_for_query.append(paper_log)

            results_log[query_text] = {"analyzed_papers": analyzed_papers_for_query}
            with open(OUTPUT_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(results_log, f, indent=4, ensure_ascii=False)

    finally:
        citer_fetcher.close()

    # --- 5. Compute and print final report from log ---
    total_retrieved, total_simple_cocites, total_contextual_cocites = 0, 0, 0
    query_simple_acc, query_contextual_acc = [], []

    for result in results_log.values():
        papers = result["analyzed_papers"]
        num_retrieved = len(papers)
        if num_retrieved == 0: continue
        total_retrieved += num_retrieved
        
        simple_cocites = sum(1 for p in papers if p["is_co_cited"])
        contextual_cocites = sum(1 for p in papers if p["found_contextual_co_citation"])
        
        total_simple_cocites += simple_cocites
        total_contextual_cocites += contextual_cocites
        query_simple_acc.append(simple_cocites / num_retrieved)
        query_contextual_acc.append(contextual_cocites / num_retrieved)

    if total_retrieved > 0:
        print("\n--- Final Evaluation Report ---")
        print(f"Total evaluated queries: {len(results_log)}")
        print(f"Total retrieved papers: {total_retrieved}")
        
        print("\n--- Metric 1: Simple Co-Citation Rate (Structural Check) ---")
        micro = total_simple_cocites / total_retrieved
        macro = sum(query_simple_acc) / len(query_simple_acc)
        print(f"Total co-cited papers: {total_simple_cocites}")
        print(f"Macro-average accuracy
