import json
import sqlite3
import os
from tqdm import tqdm

# --- 1. 配置路径 ---

# 新的模型召回结果文件路径
RESULTS_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/pasa/oaids/queries_task2_cooccur_v2.json"

# 查询文件路径
QUERIES_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/Queries/queries_task2_cooccur_v2.json"

# 数据库文件路径 (用于查询引用者)
REVERSE_DB_PATH = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/cited_to_citing.db"


# --- 2. 数据库查询函数 ---
def get_citations(paper_id_short: str, cursor: sqlite3.Cursor) -> list[str]:
    """
    根据论文的OpenAlex ID（短格式），查询所有引用了该论文的论文列表。
    
    Args:
        paper_id_short: 论文的短ID, e.g., 'W12345'.
        cursor: 数据库游标对象。
        
    Returns:
        一个包含所有引用论文ID（完整格式）的列表。
    """
    try:
        cursor.execute("SELECT citing_work_ids FROM cited_to_citing WHERE referenced_work_id = ?", (paper_id_short,))
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
    except Exception as e:
        print(f"Error querying citations for {paper_id_short}: {e}")
    return []


# --- 3. 主评估逻辑 ---
def main():
    """
    执行召回结果（共被引）评估的主函数。
    """
    # 检查必要的文件是否存在
    if not all(os.path.exists(p) for p in [QUERIES_FILE_PATH, RESULTS_FILE_PATH, REVERSE_DB_PATH]):
        print("错误：一个或多个所需文件（查询、结果或数据库）未找到。请检查路径配置。")
        return

    print("--- 开始评估召回质量 (Co-citation Logic) ---")
    print(f"结果文件: {RESULTS_FILE_PATH}")


    # 加载查询和召回结果
    try:
        with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        print(f"成功加载 {len(queries_data)} 条查询和 {len(results_data)} 条召回结果。")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"错误：读取JSON文件失败: {e}")
        return

    # 连接数据库
    conn = None
    try:
        conn = sqlite3.connect(f"file:{REVERSE_DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        print(f"成功连接到数据库: {REVERSE_DB_PATH}")
    except sqlite3.Error as e:
        print(f"数据库连接错误: {e}")
        return

    # 初始化统计变量
    total_queries_evaluated = 0
    total_retrieved_papers = 0
    total_correct_papers = 0
    query_accuracies = []
    
    # 遍历所有查询
    for source_paper_id_full, query_text in tqdm(queries_data.items(), desc="评估共被引查询"):
        source_paper_id_short = source_paper_id_full.split('/')[-1]

        retrieved_papers = results_data.get(query_text)
        if retrieved_papers is None:
            # print(f"\n警告：在结果文件中找不到查询 '{query_text[:50]}...' 的召回列表。")
            continue
        
        # 为提高效率，先获取源论文A的引用者列表并转为集合
        source_citations_set = set(get_citations(source_paper_id_short, cursor))

        # 如果召回列表为空或源论文没有任何引用者，则准确率为0
        if not retrieved_papers or not source_citations_set:
            if retrieved_papers:
                query_accuracies.append(0.0)
                total_retrieved_papers += len(retrieved_papers)
            total_queries_evaluated += 1
            continue

        correct_count_for_query = 0
        num_retrieved_for_query = len(retrieved_papers)

        for paper in retrieved_papers:
            retrieved_paper_id_full = paper.get("id")
            if not retrieved_paper_id_full:
                continue

            retrieved_paper_id_short = retrieved_paper_id_full.split('/')[-1]
            
            # 获取召回论文B的引用者列表
            retrieved_citations = get_citations(retrieved_paper_id_short, cursor)
            
            # 检查两个引用者列表的交集是否非空
            # isdisjoint() 是检查两个集合是否没有共同元素，比手动做交集更快
            if not source_citations_set.isdisjoint(retrieved_citations):
                correct_count_for_query += 1

        # 计算当前query的准确率
        accuracy = correct_count_for_query / num_retrieved_for_query
        query_accuracies.append(accuracy)

        # 更新全局统计
        total_queries_evaluated += 1
        total_retrieved_papers += num_retrieved_for_query
        total_correct_papers += correct_count_for_query

    # 关闭数据库连接
    if conn:
        conn.close()
        print("\n数据库连接已关闭。")

    # --- 4. 计算并打印最终评估报告 ---
    if total_queries_evaluated > 0:
        # 微平均准确率 (Micro-Average)
        micro_avg_accuracy = total_correct_papers / total_retrieved_papers if total_retrieved_papers > 0 else 0.0
        # 宏平均准确率 (Macro-Average)
        macro_avg_accuracy = sum(query_accuracies) / len(query_accuracies) if query_accuracies else 0.0

        print("\n--- 最终评估报告 (Co-citation) ---")
        print(f"已评估查询总数: {total_queries_evaluated}")
        print(f"总召回论文数: {total_retrieved_papers}")
        print(f"总正确召回数 (存在共被引): {total_correct_papers}")
        print("-" * 35)
        print(f"宏平均准确率 (Macro-Average Accuracy): {macro_avg_accuracy:.2%}")
        print(f"微平均准确率 (Micro-Average Accuracy): {micro_avg_accuracy:.2%}")
        print("-" * 35)

    else:
        print("没有评估任何查询，请检查输入文件和路径。")

if __name__ == "__main__":
    main()