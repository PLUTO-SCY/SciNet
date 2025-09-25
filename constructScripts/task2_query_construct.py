import json

# 文件路径
INPUT_FILE = "/data5/shaochenyang/AI_Scientist/OpenAlex/task2Result/filtered_200papers_citation1000+.json"
OUTPUT_CITES = "/data5/shaochenyang/AI_Scientist/OpenAlex/Task4Evaluation/queries_task2_pncites_v2.json"
OUTPUT_CO_OCCUR = "/data5/shaochenyang/AI_Scientist/OpenAlex/Task4Evaluation/queries_task2_cooccur_v2.json"

# 读取 JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    papers = json.load(f)

# 取前300条
papers = papers[:200]

# 构造 query (用 id 作为 key)
queries_cites = {}
queries_co_occur = {}

for paper in papers:
    paper_id = paper.get("id", "").strip()
    title = paper.get("title", "").strip()

    if not paper_id or not title:
        continue

    queries_cites[paper_id] = f'Which papers positively cite the paper titled "{title}"?'
    queries_co_occur[paper_id] = f'Which papers co-occur with the paper titled "{title}" in citation contexts and are similar?'

# 保存到本地 JSON (字典格式)
with open(OUTPUT_CITES, "w", encoding="utf-8") as f:
    json.dump(queries_cites, f, ensure_ascii=False, indent=2)

with open(OUTPUT_CO_OCCUR, "w", encoding="utf-8") as f:
    json.dump(queries_co_occur, f, ensure_ascii=False, indent=2)

print(f"已生成 query 文件:\n- {OUTPUT_CITES}\n- {OUTPUT_CO_OCCUR}")
