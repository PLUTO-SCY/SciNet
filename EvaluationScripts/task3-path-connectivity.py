import json
import os
import sqlite3
import random
import sys
from tqdm import tqdm

# --- 1. 配置路径 ---

# 模型召回结果文件路径
RESULTS_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/pasa/oaids/queries_task3_paths.json"

# 查询文件路径
QUERIES_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/Queries/queries_task3_paths.json"

# Ground Truth 答案文件路径
GROUND_TRUTH_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/answers_task3_paths.json"

# 用于查询参考文献的数据库路径
FORWARD_DB_PATH = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/citing_to_cited.db"


# --- 2. 数据库查询与辅助函数 ---
def get_references(paper_id_short: str, cursor: sqlite3.Cursor) -> set:
    """
    根据论文的OpenAlex ID（短格式），查询其参考文献列表。
    返回一个集合以便快速查找。
    """
    try:
        cursor.execute("SELECT referenced_work_ids FROM citing_to_cited WHERE work_id = ?", (paper_id_short,))
        row = cursor.fetchone()
        if row and row[0]:
            return set(json.loads(row[0]))
    except Exception as e:
        print(f"查询参考文献时出错 {paper_id_short}: {e}")
    return set()

def _is_sequence_connected(sequence: list, cursor: sqlite3.Cursor) -> bool:
    """
    【辅助函数】检查一个【特定顺序】的论文列表是否连通。
    检查逻辑: paper_{i+1} 是否引用了 paper_i
    """
    for i in range(len(sequence) - 1):
        current_paper_full_id = sequence[i].get("id")
        next_paper_full_id = sequence[i+1].get("id")

        current_paper_short_id = current_paper_full_id.split('/')[-1]
        
        if not current_paper_full_id or not next_paper_full_id:
            return False

        next_paper_short_id = next_paper_full_id.split('/')[-1]
        references = get_references(next_paper_short_id, cursor)
        
        if current_paper_short_id not in references:
            return False
            
    return True

def check_path_connectivity(path: list, cursor: sqlite3.Cursor) -> bool:
    """
    【重写函数】检查给定的论文列表（路径）是否存在任何连通的可能性。
    会尝试7种排列方式：正序、逆序、5次随机打乱。
    只要有一次检查成功，即认为路径是连通的。
    """
    if len(path) < 2:
        return False
    
    if _is_sequence_connected(path, cursor):
        return True
        
    path_reversed = path[::-1]
    if _is_sequence_connected(path_reversed, cursor):
        return True

    for _ in range(5):
        path_shuffled = random.sample(path, len(path))
        if _is_sequence_connected(path_shuffled, cursor):
            return True
            
    return False

# --- 3. 主评估逻辑 ---
def main():
    """
    执行召回的引用路径评估的主函数。
    """
    required_files = [QUERIES_FILE_PATH, RESULTS_FILE_PATH, GROUND_TRUTH_FILE_PATH, FORWARD_DB_PATH]
    for path in required_files:
        if not os.path.exists(path):
            print(f"错误：必需文件未找到: {path}")
            return

    print("--- 开始评估召回的引用路径质量 ---")
    
    # 加载数据
    try:
        with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        with open(GROUND_TRUTH_FILE_PATH, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
    except Exception as e:
        print(f"错误：读取JSON文件失败: {e}")
        return

    # 连接数据库
    conn = None
    try:
        conn = sqlite3.connect(f"file:{FORWARD_DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        print("成功连接到数据库。")
    except sqlite3.Error as e:
        print(f"数据库连接错误: {e}")
        return

    # 初始化统计变量
    total_queries_evaluated = 0
    total_recalled_papers = 0
    total_correct_papers_in_path = 0
    query_precisions = []
    connected_paths_count = 0

    try:
        for field_name, query_text in tqdm(queries_data.items(), desc="评估查询中"):
            recalled_papers_raw = results_data.get(query_text)
            ground_truth_entry = ground_truth_data.get(field_name)

            if recalled_papers_raw is None or ground_truth_entry is None:
                continue

            # --- 新增步骤：过滤掉召回结果中的 null/None 条目 ---
            # 这样可以确保后续计算的安全性
            recalled_papers = [paper for paper in recalled_papers_raw if paper is not None and paper['id'] is not None]
            # ---------------------------------------------------

            total_queries_evaluated += 1
            
            # --- 指标1: Path Precision 计算 ---
            gt_path_short_ids = ground_truth_entry.get("most_influential_path", [])
            gt_path_full_ids_set = {f"https://openalex.org/{pid}" for pid in gt_path_short_ids}
            
            num_recalled = len(recalled_papers)
            if num_recalled > 0:
                correct_hits_for_query = 0
                for paper in recalled_papers:
                    if paper.get("id") in gt_path_full_ids_set:
                        correct_hits_for_query += 1
                
                precision = correct_hits_for_query / num_recalled
                query_precisions.append(precision)
                total_recalled_papers += num_recalled
                total_correct_papers_in_path += correct_hits_for_query

            # --- 指标2: Connectivity Rate 计算 ---
            # 从过滤后的列表中取前5个
            top_5_papers = recalled_papers[:5]
            if check_path_connectivity(top_5_papers, cursor):
                connected_paths_count += 1

    finally:
        if conn:
            conn.close()
            print("\n数据库连接已关闭。")

    # --- 4. 计算并打印最终评估报告 ---
    if total_queries_evaluated > 0:
        macro_avg_precision = sum(query_precisions) / len(query_precisions) if query_precisions else 0.0
        micro_avg_precision = total_correct_papers_in_path / total_recalled_papers if total_recalled_papers > 0 else 0.0
        connectivity_rate = connected_paths_count / total_queries_evaluated

        print("\n--- 最终评估报告 ---")
        print(f"已评估查询总数: {total_queries_evaluated}")

        print("\n--- 指标 1: 路径精确率 (Path Precision) ---")
        print("衡量召回的论文有多少在真实的GT路径中")
        print(f"宏平均精确率 (Macro-Avg Precision): {macro_avg_precision:.2%}")
        print(f"微平均精确率 (Micro-Avg Precision): {micro_avg_precision:.2%}")

        print("\n--- 指标 2: 连通率 (Connectivity Rate @ Top 5) ---")
        print("衡量Top-5召回结果构成有效引用链的查询占比")
        print(f"成功连通的查询数: {connected_paths_count}")
        print(f"连通率: {connectivity_rate:.2%}")
        print("-" * 45)

    else:
        print("没有评估任何查询，请检查输入文件是否匹配。")

if __name__ == "__main__":
    main()