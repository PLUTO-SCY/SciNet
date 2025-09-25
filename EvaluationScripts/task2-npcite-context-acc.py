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

# --- 1. 配置区域 ---

RESULTS_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/o3-deep-research-2025-06-26/papers_oaids/queries_task2_pncites_v2_papers.json"

QUERIES_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/Queries/queries_task2_pncites_v2.json"

# --- 新增：输出日志文件 ---
LOG_DIR = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/EvaluationScripts/logs/contextlogs"
os.makedirs(LOG_DIR, exist_ok=True)

# 提取 "results/" 之后的第一层目录名
results_dirname = os.path.basename(
    os.path.dirname(
        os.path.dirname(RESULTS_FILE_PATH)  # 上两级目录 -> papers_oaids 的上级
    )
)
# 构造日志文件名
OUTPUT_LOG_FILE = os.path.join(LOG_DIR, f"evaluation_log_{results_dirname}.log")

# --- 数据库与资源路径 ---
FORWARD_DB_PATH = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/citing_to_cited.db"
ARXIV_LOCAL_ROOT = "/data6/Arxiv_Papers/Arxiv"
OUTPUT_PDF_DIR = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/EvaluationScripts/temp_pdfs"
os.makedirs(OUTPUT_PDF_DIR, exist_ok=True)

# --- 服务与API配置 ---
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
# --- 修改点 1: 新增GROBID超时配置 ---
GROBID_TIMEOUT_SECONDS = 240  # 为GROBID设置更长的超时时间（秒），从60秒增加到180秒
LLM_API_ENDPOINT = "http://35.220.164.252:3888/v1"
LLM_API_KEY = "sk-B52cka26mugEd4P3EEDyIvMU2jlEabH37wuHz30KNy7825SZ"
LLM_MODEL = "gpt-5-mini-2025-08-07"

# --- 解析参数 ---
SIMILARITY_THRESHOLD = 85
GROBID_TITLE_MATCH_THRESHOLD = 90

# --- GROBID XML 解析命名空间 ---
TEI_NAMESPACE = "http://www.tei-c.org/ns/1.0"
NS_MAP = {'tei': TEI_NAMESPACE}


# --- 2. 核心功能模块 (类封装) ---

class PDFManager:
    """处理PDF的获取，优先本地，其次下载。"""
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
            tqdm.write(f"[PDFManager Error] 处理 '{title[:30]}...' 时出错: {e}")
        return None

class ContextExtractor:
    """使用GROBID从PDF中提取特定引用的上下文。"""
    # --- 修改点 2: 更新构造函数以接收超时参数 ---
    def __init__(self, grobid_url, timeout_seconds):
        self.grobid_url = grobid_url
        self.timeout = timeout_seconds  # 将超时时间存储为实例变量
        self.parser = etree.XMLParser(recover=True)

    def get_citation_contexts(self, pdf_path: str, target_reference_title: str) -> list[str]:
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': (os.path.basename(pdf_path), f, 'application/pdf')}
                # --- 修改点 3: 在请求中使用配置的超时时间 ---
                response = requests.post(self.grobid_url, files=files, timeout=self.timeout)
            if response.status_code != 200: return []
            root = etree.fromstring(response.content, self.parser)
        except Exception as e:
            tqdm.write(f"[GROBID Error] 调用GROBID失败: {e}")
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
    """使用LLM分析引用上下文的情感倾向。"""
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
            tqdm.write(f"[LLM Error] API调用失败: {e}")
        return None

# --- 3. 辅助函数 ---
def extract_title_from_query(query: str) -> str | None:
    match = re.search(r'titled "(.*?)"', query)
    return match.group(1) if match else None

# --- 4. 主评估逻辑 ---
def main():
    print(f"--- 开始评估正面引用召回质量 (日志将保存至: {OUTPUT_LOG_FILE}) ---")
    if not all(os.path.exists(p) for p in [QUERIES_FILE_PATH, RESULTS_FILE_PATH, FORWARD_DB_PATH]):
        print("错误：一个或多个必需文件未找到。请检查路径配置。")
        return

    pdf_manager = PDFManager(ARXIV_LOCAL_ROOT, OUTPUT_PDF_DIR)
    # --- 修改点 4: 在初始化时传入超时参数 ---
    context_extractor = ContextExtractor(GROBID_URL, GROBID_TIMEOUT_SECONDS)
    sentiment_analyzer = SentimentAnalyzer(LLM_API_KEY, LLM_API_ENDPOINT, LLM_MODEL)
    
    with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f: queries_data = json.load(f)
    with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f: results_data = json.load(f)

    if os.path.exists(OUTPUT_LOG_FILE):
        try:
            with open(OUTPUT_LOG_FILE, 'r', encoding='utf-8') as f:
                results_log = json.load(f)
            tqdm.write(f"已成功加载 {len(results_log)} 条来自日志文件的已有结果。")
        except (json.JSONDecodeError, IOError):
            results_log = {}
            tqdm.write("警告：日志文件存在但无法解析，将创建一个新的日志。")
    else:
        results_log = {}

    conn = sqlite3.connect(f"file:{FORWARD_DB_PATH}?mode=ro", uri=True)
    cursor = conn.cursor()

    try:
        for source_paper_id_full, query_text in tqdm(queries_data.items(), desc="评估所有查询"):
            if query_text in results_log:
                continue

            target_title = extract_title_from_query(query_text)
            if not target_title: continue

            retrieved_papers = results_data.get(query_text, [])
            analyzed_papers_for_query = []
            
            for paper in tqdm(retrieved_papers, desc=f"处理查询 '{target_title[:20]}...'", leave=False):
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
            print("\n数据库连接已关闭。")

    if not results_log:
        print("没有评估任何查询。")
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

        print("\n--- 最终评估报告 (根据日志文件生成) ---")
        print(f"已评估查询总数: {len(results_log)}")
        print(f"总召回论文数: {total_retrieved_papers}")
        
        print("\n--- 指标 1: 简单引用准确率 (Structural Check) ---")
        print("衡量召回的论文是否真的引用了目标论文")
        print(f"总计引用数: {total_simple_cite_papers}")
        print(f"宏平均准确率: {macro_avg_simple:.2%}")
        print(f"微平均准确率: {micro_avg_simple:.2%}")
        
        print("\n--- 指标 2: 正面引用准确率 (Semantic Check) ---")
        print("衡量召回的论文是否对目标论文进行了正面引用")
        print(f"总计正面引用数: {total_positive_cite_papers}")
        print(f"宏平均准确率: {macro_avg_positive:.2%}")
        print(f"微平均准确率: {micro_avg_positive:.2%}")
        print("-" * 45)

if __name__ == "__main__":
    main()

