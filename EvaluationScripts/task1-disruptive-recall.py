import json
import os
from tqdm import tqdm
import argparse

# --- 1. 配置路径 (已更新为Disruption任务) ---

# 模型召回结果文件路径
RESULTS_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/results/pasa/oaids/queries_task1_disruptive.json"

# 查询文件路径
QUERIES_FILE_PATH = "/data5/shaochenyang/AI_Scientist/Baselines/Task4Evaluation/Queries/queries_task1_disruptive.json"

# Ground Truth 文件所在的目录
GROUND_TRUTH_DIR = "/data5/shaochenyang/AI_Scientist/OpenAlex/task1Result/disruption"


# --- 2. 主评估逻辑 ---
def main(recall_k):
    """
    执行颠覆性召回评估（Recall@K）的主函数。
    """
    # 检查核心文件和目录是否存在
    if not os.path.exists(QUERIES_FILE_PATH):
        print(f"错误：查询文件未找到: {QUERIES_FILE_PATH}")
        return
    if not os.path.exists(RESULTS_FILE_PATH):
        print(f"错误：结果文件未找到: {RESULTS_FILE_PATH}")
        return
    if not os.path.isdir(GROUND_TRUTH_DIR):
        print(f"错误：Ground Truth目录未找到: {GROUND_TRUTH_DIR}")
        return

    print(f"--- 开始评估颠覆性召回质量 (Recall@{recall_k}) ---")

    # 加载查询和召回结果
    try:
        with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"错误：读取核心查询或结果文件失败: {e}")
        return

    # 初始化统计变量
    all_query_recalls = []
    total_queries_evaluated = 0

    # 遍历所有查询
    for field_name, query_text in tqdm(queries_data.items(), desc="评估查询中"):
        
        # 1. 定位并加载Ground Truth文件
        gt_file_path = os.path.join(GROUND_TRUTH_DIR, f"{field_name}.json")
        
        if not os.path.exists(gt_file_path):
            print(f"\n警告：找不到领域 '{field_name}' 对应的Ground Truth文件，跳过。路径: {gt_file_path}")
            continue

        try:
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except json.JSONDecodeError:
            print(f"\n警告：JSON解析失败，文件可能已损坏，跳过。文件: {gt_file_path}")
            continue 
        except Exception as e:
            print(f"\n警告：读取Ground Truth文件时发生未知错误，跳过。错误: {e}")
            continue

        # 2. 提取Top-K的Ground Truth ID (假设GT文件已按颠覆性降序排列)
        top_k_gt_papers = gt_data[:recall_k]
        ground_truth_ids_set = {paper['id'] for paper in top_k_gt_papers if 'id' in paper}

        # 3. 获取召回结果并计算命中数
        recalled_papers = results_data.get(query_text)
        if recalled_papers is None:
            print(f"\n警告：在结果文件中找不到查询 '{query_text[:50]}...' 的召回列表，跳过。")
            continue
        
        hits = 0
        for paper in recalled_papers:
            if paper.get("id") in ground_truth_ids_set:
                hits += 1
        
        # 4. 计算当前查询的Recall@K
        recall_for_query = hits / recall_k if recall_k > 0 else 0.0
        all_query_recalls.append(recall_for_query)
        total_queries_evaluated += 1

    # --- 3. 计算并打印最终评估报告 ---
    if total_queries_evaluated > 0:
        # 计算宏平均Recall
        macro_average_recall = sum(all_query_recalls) / len(all_query_recalls)

        print(f"\n--- 最终评估报告 (Disruption Recall@{recall_k}) ---")
        print(f"已成功评估查询总数: {total_queries_evaluated}")
        print(f"评估标准: 召回的论文是否命中各领域Top {recall_k}高颠覆性论文列表")
        print("-" * 50)
        print(f"宏平均召回率 (Macro-Average Recall@{recall_k}): {macro_average_recall:.2%}")
        print("-" * 50)

    else:
        print("没有成功评估任何查询，请检查文件路径和内容是否匹配。")

if __name__ == "__main__":
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description="评估颠覆性（Disruption）召回率 Recall@K.")
    parser.add_argument(
        '--k',
        type=int,
        default=50,
        help="用于计算Recall的Top-K值 (默认: 50)"
    )
    args = parser.parse_args()
    
    # 将解析出的k值传入main函数
    main(args.k)