import json, os, re

# ============================================================
# 0. 股票代码 → 公司名 映射
# ============================================================
CODE_TO_NAME = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "000568": "泸州老窖",
    "600809": "山西汾酒",
    "002304": "洋河股份",
    "000596": "古井贡酒",
    "603369": "今世缘",
    "603198": "迎驾贡酒",
    "600702": "舍得酒业",
    "600779": "水井坊",
}

def get_company(filename):
    m = re.search(r'(\d{6})', filename)
    if m and m.group(1) in CODE_TO_NAME:
        return CODE_TO_NAME[m.group(1)]
    name = filename.replace(".pdf", "").replace(".json", "")
    m2 = re.match(r'([\u4e00-\u9fa5]{2,6})', name)
    if m2:
        return m2.group(1)
    return "未知公司"

# ============================================================
# 1. 章节识别（贪婪匹配 + 目录过滤）
# ============================================================
SECTION_PAT = re.compile(
    r'(?<![\.\d])'
    r'(第[一二三四五六七八九十]+节[\u4e00-\u9fa5A-Za-z0-9]{2,20})'
    r'(?!\s*[\d\.]{2,})'
)

def build_section_index(full_text):
    positions = []
    for m in SECTION_PAT.finditer(full_text):
        after = full_text[m.end():m.end()+30]
        if "..." in after or "…" in after:
            continue
        if re.match(r'^\s*\d+\s*$', after[:8]):
            continue
        positions.append((m.start(), m.group(1).strip()))
    return positions

# ============================================================
# 2. 章节名清洗：只保留「第X节 + 最多8个中文字」
# ============================================================
CLEAN_SECTION_PAT = re.compile(r'(第[一二三四五六七八九十]+节[\u4e00-\u9fa5]{2,8})')

def clean_section(name):
    m = CLEAN_SECTION_PAT.match(name)
    return m.group(1) if m else name

# ============================================================
# 3. 切块
# ============================================================
def chunk_text(text, chunk_size=800, overlap=120):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append((start, text[start:end]))
        start = end - overlap
    return chunks

# ============================================================
# 4. 主流程
# ============================================================
all_chunks = []

for f in os.listdir("corpus"):
    if not f.endswith(".json"):
        continue

    company = get_company(f)
    with open(f"corpus/{f}", "r") as fp:
        pages = json.load(fp)

    full_text = ""
    page_starts = []
    for p in pages:
        page_starts.append((len(full_text), p["page"]))
        full_text += p["text"]

    section_positions = build_section_index(full_text)

    def find_section(pos):
        cur = "正文"
        for s_pos, s_name in section_positions:
            if s_pos <= pos:
                cur = s_name
            else:
                break
        return clean_section(cur)

    def find_page(pos):
        page = 1
        for s_pos, p_num in page_starts:
            if pos >= s_pos:
                page = p_num
            else:
                break
        return page

    for i, (c_start, c_text) in enumerate(chunk_text(full_text)):
        all_chunks.append({
            "company": company,
            "section": find_section(c_start),
            "page": find_page(c_start),
            "chunk_id": i,
            "source": f,
            "text": c_text
        })

# ============================================================
# 5. 保存 + 打印样本
# ============================================================
with open("chunks.json", "w") as fp:
    json.dump(all_chunks, fp, ensure_ascii=False)

print(f"共切出 {len(all_chunks)} 块")
print("=" * 60)

for c in all_chunks[:3]:
    print(f"公司={c['company']} | 章节={c['section']} | 页码={c['page']} | 块序号={c['chunk_id']}")
    print(c["text"][:150].replace("\n", " "))
    print("-" * 60)
