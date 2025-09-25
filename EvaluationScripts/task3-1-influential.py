import json
import os
import sqlite3
from tqdm import tqdm

# --- 1. 配置路径 ---
# 查询文件路径
QUERIES_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/Queries/queries_task3_influential.json"

# 模型召回结果文件路径
RESULTS_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/SciBERT/answers_task3_influential_faiss.json"

# Ground Truth 答案文件路径
GROUND_TRUTH_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/answers_task3-2_topciting.json"

# 新增：用于查询参考文献的数据库路径
FORWARD_DB_PATH = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/citing_to_cited.db"


# --- 2. 新增：数据库查询函数 ---
def get_references(paper_id_short: str, cursor: sqlite3.Cursor) -> list[str]:
    """
    根据论文的OpenAlex ID（短格式），查询其参考文献列表。
    """
    try:
        cursor.execute("SELECT referenced_work_ids FROM citing_to_cited WHERE work_id = ?", (paper_id_short,))
        row = cursor.fetchone()
        if row and row[0]:
            # 使用集合返回，便于快速查找
            return set(json.loads(row[0]))
    except Exception as e:
        print(f"Error querying references for {paper_id_short}: {e}")
    # 如果没有找到或发生错误，返回空集合
    return set()


# --- 3. 主评估逻辑 ---
def main():
    """
    执行召回结果评估的主函数，同时计算Hit Rate和Precision。
    """
    # 检查所有必要文件是否存在
    required_files = [QUERIES_FILE_PATH, RESULTS_FILE_PATH, GROUND_TRUTH_FILE_PATH, FORWARD_DB_PATH]
    for path in required_files:
        if not os.path.exists(path):
            print(f"错误：必需文件未找到: {path}")
            return

    print("--- 开始评估召回质量 (Influential Papers Task with Precision) ---")
    
    # 加载所有JSON文件
    try:
        with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        with open(GROUND_TRUTH_FILE_PATH, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        print(f"\n成功加载 {len(queries_data)} 条查询, {len(results_data)} 条召回结果, 和 {len(ground_truth_data)} 条GT记录。")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"错误：读取JSON文件失败: {e}")
        return

    # 连接数据库
    conn = None
    try:
        conn = sqlite3.connect(f"file:{FORWARD_DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        print(f"成功连接到数据库: {FORWARD_DB_PATH}")
    except sqlite3.Error as e:
        print(f"数据库连接错误: {e}")
        if conn:
            conn.close()
        return

    # 初始化统计变量
    total_queries_evaluated = 0
    total_retrieved_papers = 0
    
    # Hit Rate (影响力) 统计
    total_influence_hits = 0
    query_hit_rates = []

    # Precision (相关性) 统计
    total_citing_hits = 0
    query_precisions = []
    
    # 主循环
    try:
        for source_paper_id_short, query_text in tqdm(queries_data.items(), desc="评估查询中"):
            
            recalled_papers = results_data.get(query_text)
            ground_truth_entry = ground_truth_data.get(source_paper_id_short)

            if recalled_papers is None or ground_truth_entry is None:
                continue
            
            # 准备GT影响力论文ID集合
            top_papers_list = ground_truth_entry.get("top_citing_papers", [])
            ground_truth_ids_set = {p['id'] for p in top_papers_list if 'id' in p}

            # 准备源论文的完整ID，用于检查引用关系
            source_paper_id_full = f"https://openalex.org/{source_paper_id_short}"

            # 初始化当前查询的计数器
            influence_hits_for_query = 0
            citing_hits_for_query = 0
            num_retrieved = len(recalled_papers)

            if num_retrieved == 0:
                continue

            for paper in recalled_papers:
                retrieved_paper_id_full = paper.get("id")
                if not retrieved_paper_id_full:
                    continue

                # 1. 检查影响力 (是否在GT列表中)
                if retrieved_paper_id_full in ground_truth_ids_set:
                    influence_hits_for_query += 1
                
                # 2. 检查相关性 (是否引用了源论文)
                retrieved_paper_id_short = retrieved_paper_id_full.split('/')[-1]
                references = get_references(retrieved_paper_id_short, cursor)
                if source_paper_id_short in references:
                    citing_hits_for_query += 1

            # 计算当前查询的两个指标
            query_hit_rates.append(influence_hits_for_query / num_retrieved)
            query_precisions.append(citing_hits_for_query / num_retrieved)
            
            # 更新全局统计
            total_queries_evaluated += 1
            total_retrieved_papers += num_retrieved
            total_influence_hits += influence_hits_for_query
            total_citing_hits += citing_hits_for_query
    
    finally:
        # 确保数据库连接被关闭
        if conn:
            conn.close()
            print("\n数据库连接已关闭。")

    # --- 4. 计算并打印最终评估报告 ---
    if total_queries_evaluated > 0:
        # 计算 Hit Rate 指标
        macro_avg_hit_rate = sum(query_hit_rates) / len(query_hit_rates)
        micro_avg_hit_rate = total_influence_hits / total_retrieved_papers
        
        # 计算 Precision 指标
        macro_avg_precision = sum(query_precisions) / len(query_precisions)
        micro_avg_precision = total_citing_hits / total_retrieved_papers

        print("\n--- 最终评估报告 ---")
        print(f"已评估查询总数: {total_queries_evaluated}")
        print(f"总召回论文数: {total_retrieved_papers}")

        print("\n--- 指标 1: Hit Rate (影响力) ---")
        print("衡量召回的论文是否在'高影响力'GT列表中")
        print(f"总命中数 (In Ground Truth): {total_influence_hits}")
        print(f"宏平均命中率 (Macro-Avg Hit Rate): {macro_avg_hit_rate:.2%}")
        print(f"微平均命中率 (Micro-Avg Hit Rate): {micro_avg_hit_rate:.2%}")

        print("\n--- 指标 2: Precision (相关性/有效性) ---")
        print("衡量召回的论文是否真正引用了源论文")
        print(f"总有效召回数 (Cites Source): {total_citing_hits}")
        print(f"宏平均精确率 (Macro-Avg Precision): {macro_avg_precision:.2%}")
        print(f"微平均精确率 (Micro-Avg Precision): {micro_avg_precision:.2%}")
        print("-" * 35)

    else:
        print("没有评估任何查询，请检查输入文件是否匹配。")

if __name__ == "__main__":
    main()