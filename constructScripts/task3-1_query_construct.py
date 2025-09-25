import json

# 文件路径
INPUT_FILE = "/data5/shaochenyang/AI_Scientist/OpenAlex/task3Result/citations_top100_results.json"
OUTPUT_FILE = "/data5/shaochenyang/AI_Scientist/OpenAlex/Task4Evaluation/queries_task3_influential.json"

# 读取 JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

queries_topk = {}

# 构造 query
for paper_id, paper_info in data.items():
    title = paper_info.get("title", "").strip()
    if not title:
        continue
    # 构造 query，使用 paper_id 作为 key
    queries_topk[paper_id] = (
        f'Which are the top-k most influential papers among those citing the paper titled "{title}"?'
    )

# 保存查询到本地
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(queries_topk, f, ensure_ascii=False, indent=2)

print(f"共生成 {len(queries_topk)} 条 query，已保存至 {OUTPUT_FILE}")
