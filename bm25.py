import json
import jieba
from rank_bm25 import BM25Okapi
import pickle

# 1. 读取切好的块
with open("chunks.json", "r") as fp:
    chunks = json.load(fp)

# 2. jieba 分词（PPT Page 35：中文要分词，英文按空格）
def tokenize(text):
    return list(jieba.cut(text))

print("正在分词...")
tokenized_corpus = [tokenize(c["text"]) for c in chunks]
print(f"完成分词，共 {len(tokenized_corpus)} 块")

# 3. 建 BM25 索引
print("正在建 BM25 索引...")
bm25 = BM25Okapi(tokenized_corpus)

# 4. 保存（用 pickle，方便下次直接加载）
with open("bm25_index.pkl", "wb") as fp:
    pickle.dump({
        "bm25": bm25,
        "chunks": chunks,
        "tokenized_corpus": tokenized_corpus
    }, fp)

print("BM25 索引已保存到 bm25_index.pkl")

# 5. 做个简单测试：搜「贵州茅台 直销渠道」
query = "贵州茅台 直销渠道"
tokens = tokenize(query)
scores = bm25.get_scores(tokens)
top3 = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]

print(f"\n测试检索：{query}")
for i in top3:
    c = chunks[i]
    print(f"\n分数={scores[i]:.2f} | 公司={c['company']} | 章节={c['section']} | 页码={c['page']}")
    print(c["text"][:150].replace("\n", " "))
