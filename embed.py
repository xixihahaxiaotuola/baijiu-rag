import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 加载 Qwen3-Embedding-0.6B（第一次运行会下载约 1.2G，需要等几分钟）
print("正在加载模型...")
model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

# 读切好的块
with open("chunks.json", "r") as fp:
    chunks = json.load(fp)

texts = [c["text"] for c in chunks]
print(f"共 {len(texts)} 块，开始向量化...")

# 批量计算向量
embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True   # 归一化后，点积=余弦相似度，检索更快
)

# 保存
np.save("embeddings.npy", embeddings)
print(f"完成！向量维度：{embeddings.shape}")
