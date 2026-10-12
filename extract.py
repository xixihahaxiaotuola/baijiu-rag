import pdfplumber, json, os

def table_to_text(table):
    """把表格按行还原成文字：每一行用 ' | ' 分隔，方便后面检索命中"""
    if not table:
        return ""
    lines = []
    for row in table:
        # 过滤 None，转成字符串
        cells = [str(c).strip() if c is not None else "" for c in row]
        # 去掉全空的行
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)

def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. 抽正文
            text = page.extract_text() or ""

            # 2. 抽表格，并按行拼成文字
            tables = page.extract_tables() or []
            table_texts = []
            for t in tables:
                t_text = table_to_text(t)
                if t_text:
                    table_texts.append(t_text)

            # 3. 把表格文字接在正文后面（这样切块时表格就一起进去了）
            full_page_text = text
            if table_texts:
                full_page_text += "\n\n【本页表格】\n" + "\n\n".join(table_texts)

            pages.append({
                "page": i + 1,
                "text": full_page_text
            })
    return pages

os.makedirs("corpus", exist_ok=True)
for f in os.listdir("pdfs"):
    if f.endswith(".pdf"):
        print(f"正在提取: {f}")
        data = extract_pdf(f"pdfs/{f}")
        out_name = f.replace(".pdf", ".json")
        with open(f"corpus/{out_name}", "w") as fp:
            json.dump(data, fp, ensure_ascii=False)
print("提取完成！（表格已按行还原）")
