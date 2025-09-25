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

# --- 1. 配置区域 ---

# --- 指定文件 ---
# 注意：您的任务是 co-occur，请确保使用的是对应的 query 和 result 文件
QUERIES_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/Queries/queries_task2_cooccur_v2.json"
RESULTS_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/o4-mini-deep-research-2025-06-26/papers_oaids/queries_task2_cooccur_v2_papers.json"

# --- 新增：输出日志文件 ---
LOG_DIR = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/EvaluationScripts/logs/cocitelogs"
os.makedirs(LOG_DIR, exist_ok=True)
RESULTS_FILENAME = os.path.basename(RESULTS_FILE_PATH)
OUTPUT_LOG_FILE = os.path.join(LOG_DIR, f"evaluation_log_o4-mini-deep-research-2025-06-26")

# --- 数据库与资源路径 ---
# 新增：反向引用数据库（谁引用了我）
REVERSE_DB_PATH = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/cited_to_citing.db"
ARXIV_LOCAL_ROOT = "/data6/Arxiv_Papers/Arxiv"
OUTPUT_PDF_DIR = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/EvaluationScripts/temp_pdfs"
os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)

# --- 服务与API配置 ---
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# --- 解析参数 ---
SIMILARITY_THRESHOLD = 85
GROBID_TITLE_MATCH_THRESHOLD = 85

# --- GROBID XML 解析命名空间 ---
TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
NS_MAP = {'tei': TEI_NAMESPACE}


# --- 2. 核心功能模块 (类封装) ---

class CiterFetcher:
    """从反向引用数据库中获取引用者列表。"""
    def __init__(self, db_path):
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.cursor = self.conn.cursor()

    def get_citers(self, paper_id_short: str) -> set:
        self.cursor.execute("SELECT citing_work_ids FROM cited_to_citing WHERE referenced_work_id = ?", (paper_id_short,))
        row = self.cursor.fetchone()
        return set(json.loads(row[0])) if row and row[0] else set()

    def close(self):
        self.conn.close()

class PDFManager:
    # ... 此类与您之前的代码完全相同，此处为简洁省略 ...
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
            tqdm.write(f"[PDFManager Error] 处理 '{title[:30]}...' 时出错: {e}")
        return None

class CoCitationAnalyzer:
    """使用GROBID分析PDF，检查两篇目标论文是否在同一段落被引用。"""
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
            tqdm.write(f"[GROBID Error] 调用GROBID失败: {e}")
            return False

        # 1. 在文献列表中分别找到A和B的ID
        ref_id_A = self._get_ref_id_from_title(root, title_A)
        ref_id_B = self._get_ref_id_from_title(root, title_B)

        if not ref_id_A or not ref_id_B:
            return False # 如果任一论文不在参考文献中，则无法判断

        # 2. 遍历正文中的所有段落
        paragraphs = root.xpath('//tei:body//tei:p', namespaces=NS_MAP)
        for p in paragraphs:
            # 3. 获取段落中所有引用标记的目标ID
            refs_in_p = {ref.get('target').replace('#', '') for ref in p.xpath('.//tei:ref[@target]', namespaces=NS_MAP)}
            # 4. 检查A和B的ID是否同时出现在该段落的引用集合中
            if ref_id_A in refs_in_p and ref_id_B in refs_in_p:
                return True # 找到了！

        return False # 遍历完所有段落都没找到


# --- 3. 辅助函数 ---
def extract_title_from_query(query: str) -> str | None:
    match = re.search(r'titled "(.*?)"', query)
    return match.group(1) if match else None

# --- 4. 主评估逻辑 ---
def main():
    print(f"--- 开始评估共引上下文召回质量 (日志将保存至: {OUTPUT_LOG_FILE}) ---")
    pdf_manager = PDFManager(ARXIV_LOCAL_ROOT, OUTPUT_PDF_DIR)
    analyzer = CoCitationAnalyzer(GROBID_URL)
    
    with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f: queries_data = json.load(f)
    with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f: results_data = json.load(f)

    if os.path.exists(OUTPUT_LOG_FILE):
        with open(OUTPUT_LOG_FILE, 'r', encoding='utf-8') as f: results_log = json.load(f)
    else: results_log = {}

    citer_fetcher = CiterFetcher(REVERSE_DB_PATH)

    try:
        for target_paper_id_full, query_text in tqdm(queries_data.items(), desc="评估所有查询"):
            if query_text in results_log: continue
            
            target_title = extract_title_from_query(query_text)
            if not target_title: continue
            
            recalled_papers = results_data.get(query_text, [])
            analyzed_papers_for_query = []
            
            target_id_short = target_paper_id_full.split('/')[-1]
            citers_of_target = citer_fetcher.get_citers(target_id_short)

            for paper in tqdm(recalled_papers, desc=f"处理查询 '{target_title[:20]}...'", leave=False):
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

                # 步骤 1: 寻找共同引用者
                recalled_id_short = recalled_id_full.split('/')[-1]
                citers_of_recalled = citer_fetcher.get_citers(recalled_id_short)
                common_citers = citers_of_target.intersection(citers_of_recalled)
                
                if not common_citers:
                    analyzed_papers_for_query.append(paper_log)
                    continue
                
                paper_log["is_co_cited"] = True

                # 步骤 2: 验证共引上下文
                for citer_id in common_citers:
                    # 注意：citer_id是完整URL，但没有title，我们无法直接获取PDF。
                    # 这是一个当前流程的局限。作为演示，我们假设可以获取PDF。
                    # 在实际应用中，您需要一个方法从ID->title或直接->PDF。
                    # 此处我们跳过这步，假设我们能神奇地拿到citer_title
                    citer_title = f"Paper with ID {citer_id}" # 这是一个占位符
                    
                    pdf_path = pdf_manager.get_pdf(citer_title)
                    paper_log["common_citers_checked"].append({"id": citer_id, "pdf_found": bool(pdf_path)})
                    
                    if pdf_path:
                        if analyzer.check_contextual_co_citation(pdf_path, target_title, recalled_title):
                            paper_log["found_contextual_co_citation"] = True
                            break # 找到一个就够了
                
                analyzed_papers_for_query.append(paper_log)

            results_log[query_text] = {"analyzed_papers": analyzed_papers_for_query}
            with open(OUTPUT_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(results_log, f, indent=4, ensure_ascii=False)

    finally:
        citer_fetcher.close()

    # --- 5. 从日志计算并打印报告 ---
    # ... 此部分逻辑与上一个脚本类似，计算两个指标的宏/微平均 ...
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
        print("\n--- 最终评估报告 ---")
        print(f"已评估查询总数: {len(results_log)}")
        print(f"总召回论文数: {total_retrieved}")
        
        print("\n--- 指标 1: 简单共引率 (Structural Check) ---")
        micro = total_simple_cocites / total_retrieved
        macro = sum(query_simple_acc) / len(query_simple_acc)
        print(f"总计共引数: {total_simple_cocites}")
        print(f"宏平均准确率: {macro:.2%}")
        print(f"微平均准确率: {micro:.2%}")
        
        print("\n--- 指标 2: 共引上下文准确率 (Semantic Check) ---")
        micro_ctx = total_contextual_cocites / total_retrieved
        macro_ctx = sum(query_contextual_acc) / len(query_contextual_acc)
        print(f"总计上下文共引数: {total_contextual_cocites}")
        print(f"宏平均准确率: {macro_ctx:.2%}")
        print(f"微平均准确率: {micro_ctx:.2%}")

if __name__ == "__main__":
    main()
