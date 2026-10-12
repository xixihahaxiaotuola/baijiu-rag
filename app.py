import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 镜像加速

import json
import numpy as np
import pickle
import jieba
import gradio as gr
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI

# ============================================================
# 1. 加载模型和索引（启动时执行一次）
# ============================================================
print("加载向量模型 Qwen3-Embedding-0.6B ...")
embed_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

print("加载重排模型 bge-reranker-v2-m3 ...")
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')

print("加载向量索引 ...")
embeddings = np.load("embeddings.npy")

print("加载 BM25 索引 ...")
with open("bm25_index.pkl", "rb") as fp:
    bm25_data = pickle.load(fp)
bm25 = bm25_data["bm25"]
chunks = bm25_data["chunks"]

print("全部加载完成，准备就绪。")

# ============================================================
# 2. 大模型客户端（换你自己的 API key）
# ============================================================
client = OpenAI(
    api_key="sk-0756b09718c04ac1b016a3043cd8bb9a",                     # ← 改这里
    base_url="https://api.deepseek.com"        # ← DeepSeek 用这个
)

# ============================================================
# 3. 问题改写：把口语问题压成干净的关键词
# ============================================================
def rewrite_query(query):
    prompt = f"""把下面的问题压缩成不超过15个字的关键词，只输出关键词，不要解释、不要标点：
{query}"""
    try:
        r = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return query   # 失败就退回原问题

# ============================================================
# 4. 混合检索：向量 + BM25，各召回 top_k，合并去重
# ============================================================
def hybrid_search(query, top_k=10):
    clean_q = rewrite_query(query)

    # 向量检索
    q_vec = embed_model.encode([clean_q], normalize_embeddings=True)[0]
    vec_scores = embeddings @ q_vec
    vec_top = np.argsort(vec_scores)[::-1][:top_k]

    # BM25 检索
    tokens = list(jieba.cut(clean_q))
    bm25_scores = bm25.get_scores(tokens)
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

    # 合并去重
    seen = set()
    candidates = []
    for i in vec_top:
        if i not in seen:
            seen.add(i)
            candidates.append({"idx": int(i), "chunk": chunks[i],
                               "from": "向量", "vec_score": float(vec_scores[i]),
                               "bm25_score": float(bm25_scores[i])})
    for i in bm25_top:
        if i not in seen:
            seen.add(i)
            candidates.append({"idx": int(i), "chunk": chunks[i],
                               "from": "BM25", "vec_score": float(vec_scores[i]),
                               "bm25_score": float(bm25_scores[i])})

    return clean_q, candidates

# ============================================================
# 5. 重排：用 bge-reranker 对候选重新打分，取 top_n
# ============================================================
def rerank(query, candidates, top_n=5):
    if not candidates:
        return []
    pairs = [(query, c["chunk"]["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_n]]

# ============================================================
# 6. 生成带引用的回答
# ============================================================
def answer(question):
    clean_q, candidates = hybrid_search(question, top_k=10)
    hits = rerank(clean_q, candidates, top_n=5)

    # 拼上下文
    context_parts = []
    for h in hits:
        c = h["chunk"]
        context_parts.append(
            f"【{c['company']}·{c['section']}·第{c['page']}页】\n{c['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""你是白酒行业财报助手。请严格根据以下财报片段回答用户问题。
要求：
1. 第一句先给结论；
2. 引用原文关键句，用「」标出；
3. 每一句结论后注明依据：【公司·章节·第X页】；
4. 如果片段里找不到答案，直接说：材料未涉及，不要编造。

【财报片段】：
{context}

【用户问题】：{question}
"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    ans = resp.choices[0].message.content

    # 附上召回信息（做实验记录用）
    debug = "\n\n---\n**检索过程（做实验记录用）**\n"
    debug += f"改写后的问题：`{clean_q}`\n\n"
    debug += f"混合检索共 {len(candidates)} 条候选，重排后取 top-{len(hits)}：\n"
    for h in hits:
        c = h["chunk"]
        debug += f"- [{h['from']}] {c['company']} | {c['section']} | 第{c['page']}页\n"

    return ans + debug

# ============================================================
# 7. Gradio 页面
# ============================================================
iface = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(
        label="输入你的问题",
        placeholder="例如：贵州茅台2026年上半年直销渠道收入是多少？",
        lines=2
    ),
    outputs=gr.Textbox(label="带出处的回答", lines=25),
    title="白酒行业财报问答助手（改写+混合检索+重排）",
    description="10家白酒公司2026年半年报 · RAG 实验"
)

iface.launch()
