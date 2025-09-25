import json
import sqlite3
import os
from tqdm import tqdm
import sys

# --- 1. 配置路径 ---

# 模型召回结果文件路径
RESULTS_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/pasa/oaids/queries_task2_pncites_v2.json"

# 查询文件路径
QUERIES_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/Queries/queries_task2_pncites_v2.json"

# 数据库文件路径 (用于查询参考文献)
FORWARD_DB_PATH = "/data5/shaochenyang/AI_Scientist/OpenAlex/sqlite/citing_to_cited.db"


# --- 2. 数据库查询函数 ---
def get_references(paper_id: str, cursor: sqlite3.Cursor) -> list[str]:
    """
    根据论文的OpenAlex ID（短格式，如'W12345'），查询其参考文献列表。
    
    Args:
        paper_id: 论文的短ID。
        cursor: 数据库游标对象。
        
    Returns:
        一个包含所有被引用论文ID（完整格式）的列表。
    """
    try:
        cursor.execute("SELECT referenced_work_ids FROM citing_to_cited WHERE work_id = ?", (paper_id,))
        row = cursor.fetchone()
        # 如果找到了记录并且内容不为空，则解析JSON
        if row and row[0]:
            return json.loads(row[0])
    except Exception as e:
        print(f"Error querying references for {paper_id}: {e}")
    # 如果没有找到或发生错误，返回空列表
    return []


# --- 3. 主评估逻辑 ---
def main():
    """
    执行召回结果评估的主函数。
    """
    # 检查必要的文件是否存在
    if not all(os.path.exists(p) for p in [QUERIES_FILE_PATH, RESULTS_FILE_PATH, FORWARD_DB_PATH]):
        print("错误：一个或多个所需文件（查询、结果或数据库）未找到。请检查路径配置。")
        return

    print("--- 开始评估召回质量 ---")

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
        conn = sqlite3.connect(f"file:{FORWARD_DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        print(f"成功连接到数据库: {FORWARD_DB_PATH}")
    except sqlite3.Error as e:
        print(f"数据库连接错误: {e}")
        return

    # 初始化统计变量
    total_queries_evaluated = 0
    total_retrieved_papers = 0
    total_correct_papers = 0
    query_accuracies = []
    
    # 使用tqdm创建进度条
    for source_paper_id_full, query_text in tqdm(queries_data.items(), desc="评估查询中"):
        # 从完整的URL中提取短ID，例如 'W3092732263'
        source_paper_id_short = source_paper_id_full.split('/')[-1]

        # 根据query文本在结果中找到对应的召回列表
        retrieved_papers = results_data.get(query_text)

        if retrieved_papers is None:
            print(f"\n警告：在结果文件中找不到查询 '{query_text[:50]}...' 的召回列表。")
            continue
            
        if not retrieved_papers: # 如果召回列表为空
            accuracy = 0.0
            query_accuracies.append(accuracy)
            print(f"\n查询 '{query_text[:50]}...' 的召回列表为空。准确率: {accuracy:.2%}")
            total_queries_evaluated += 1
            continue

        correct_count_for_query = 0
        num_retrieved_for_query = len(retrieved_papers)

        for paper in retrieved_papers:
            retrieved_paper_id_full = paper.get("id")
            if not retrieved_paper_id_full:
                continue

            # 获取被召回论文的短ID以查询数据库
            retrieved_paper_id_short = retrieved_paper_id_full.split('/')[-1]
            
            # 查询其参考文献列表
            references = get_references(retrieved_paper_id_short, cursor)
            
            # if len(references)>0:
            #     print(references)
            #     print(source_paper_id_short)
            # 文献召回和匹配都没啥问题
            
            # 检查源论文是否在参考文献列表中
            if source_paper_id_short in references:
                correct_count_for_query += 1

        # 计算当前query的准确率
        accuracy = correct_count_for_query / num_retrieved_for_query if num_retrieved_for_query > 0 else 0.0
        query_accuracies.append(accuracy)

        # 更新全局统计
        total_queries_evaluated += 1
        total_retrieved_papers += num_retrieved_for_query
        total_correct_papers += correct_count_for_query

        # 可以取消下面的注释来查看每个查询的详细结果
        # print(f"\n[结果] 查询源ID: {source_paper_id_short}")
        # print(f"  - 召回数量: {num_retrieved_for_query}")
        # print(f"  - 正确数量: {correct_count_for_query}")
        # print(f"  - 准确率: {accuracy:.2%}")

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

        print("\n--- 最终评估报告 ---")
        print(f"已评估查询总数: {total_queries_evaluated}")
        print(f"总召回论文数: {total_retrieved_papers}")
        print(f"总正确召回数: {total_correct_papers}")
        print("-" * 25)
        print(f"宏平均准确率 (Macro-Average Accuracy): {macro_avg_accuracy:.4f} ({macro_avg_accuracy:.2%})")
        print(f"微平均准确率 (Micro-Average Accuracy): {micro_avg_accuracy:.4f} ({micro_avg_accuracy:.2%})")
        print("----------------------\n")
        print("说明:")
        print(" - 宏平均准确率: 先计算每个查询的准确率，然后取所有查询准确率的平均值。它平等地对待每个查询。")
        print(" - 微平均准确率: 将所有查询的召回结果视为一个大集合，计算总的正确比例。它受召回数量多的查询影响更大。")

    else:
        print("没有评估任何查询，请检查输入文件和路径。")


if __name__ == "__main__":
    main()