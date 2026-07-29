# 大模型参数选择与调优指南

> 本文档讲解 kb-rag 项目中调用 LLM 时用到的三个核心生成参数（temperature、max_tokens、top_p）的含义、取值理由、调优方法，以及不同场景下的推荐配置。

---

## 1. 三个参数是什么——用做菜来比喻

把 LLM 想象成一个厨师，你的问题是"顾客点的菜"，生成的回答是"厨师做出来的菜"：

| 参数 | 比喻 | 含义 |
|------|------|------|
| `temperature` | 厨师的"发挥程度" | 0 = 严格按菜谱做，每次都一样；1 = 允许自由发挥；2 = 完全乱来 |
| `max_tokens` | 盘子的大小 | 限制厨师最多能盛多少菜。太小装不下，太大浪费空间 |
| `top_p` | 厨师选食材的"范围" | 1.0 = 所有食材都考虑；0.1 = 只从最常见的前 10% 食材里选 |

---

## 2. temperature——最重要的参数

### 2.1 它到底是什么

LLM 生成回答时，每一步都要从词表中"选下一个词"。temperature 控制这个选择的**随机性**：

```
temperature = 0.0（贪心解码）
  每次都选概率最高的那个词
  → 同样的问题，每次回答都一模一样
  → 适合：知识问答、事实检索、RAG

temperature = 0.7（默认创意区）
  大部分时候选概率高的，偶尔选概率低的
  → 同样的问题，每次回答措辞不同但意思接近
  → 适合：聊天、文案、头脑风暴

temperature = 1.5~2.0（高度随机）
  经常选概率很低的词
  → 回答可能不连贯、甚至荒谬
  → 适合：刻意制造创意/随机性（很少用）
```

### 2.2 数学原理（选读）

LLM 在每一步计算每个词的概率分布，temperature 对 logits 除以 T 后再 softmax：

```
原始概率：词A=0.8, 词B=0.15, 词C=0.05

temperature=0.1 → 几乎确定选A：词A=0.999, 词B=0.001, 词C≈0
temperature=1.0 → 保持原始概率：词A=0.8,  词B=0.15,  词C=0.05
temperature=2.0 → 概率被拉平：  词A=0.40, 词B=0.33,  词C=0.27
```

temperature 越低，概率分布越"尖锐"（集中在一个词上）；越高越"平坦"（均匀分散）。

### 2.3 RAG 场景为什么用 0.0

本项目 `.env` 默认 `LLM_TEMPERATURE=0.0`，理由：

1. **减少幻觉**：temperature=0 时模型只选概率最高的词，不会"异想天开"编造信息
2. **可复现**：同一个问题 + 同一份上下文，每次回答完全一样，便于调试和评测
3. **引用准确**：高 temperature 会让模型"改写"上下文内容，可能导致引用编号错乱

### 2.4 什么时候调高

| 场景 | 推荐 temperature | 理由 |
|------|-------------------|------|
| 知识库问答（RAG） | 0.0 | 严格基于检索内容回答 |
| 客服对话 | 0.3 | 略带自然感但不跑偏 |
| 文案生成 | 0.7 | 需要创意和多样性 |
| 诗歌/创意写作 | 1.0~1.3 | 鼓励新颖表达 |

### 2.5 怎么改

```bash
# .env 文件
LLM_TEMPERATURE=0.0    # 改成你想要的值
```

### 2.6 对应代码

```python
# app/generation/llm.py — OpenAIGenerator / ZhipuGenerator
response = client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=self.temperature,    # ← 从 settings 传入
    max_tokens=self.max_tokens,
    top_p=self.top_p,
)

# app/generation/llm.py — OllamaGenerator
payload = {
    "model": self.model,
    "messages": messages,
    "stream": False,
    "options": {
        "temperature": self.temperature,   # ← Ollama 用 options 包裹
        "num_predict": self.max_tokens,
        "top_p": self.top_p,
    },
}
```

---

## 3. max_tokens——回答长度限制

### 3.1 它是什么

限制 LLM 生成的回答最多包含多少个 token。注意：

- 1 个中文 ≈ 1~2 个 token
- 1 个英文单词 ≈ 1~3 个 token
- 所以 2048 token ≈ 1000~1500 字中文

### 3.2 为什么默认 2048

| 值 | 大约字数 | 场景 |
|----|----------|------|
| 256 | ~200 字 | 简短问答（"是/否" + 一句解释） |
| 1024 | ~700 字 | 标准问答，大部分 RAG 场景够用 |
| **2048** | **~1500 字** | **详细回答，含多个引用片段的总结** |
| 4096 | ~3000 字 | 长报告、多文档综合分析 |
| 8192+ | ~5000 字+ | 超长生成（不推荐，贵且慢） |

2048 是"够用但不浪费"的平衡点。RAG 回答一般只需要总结 3~5 个片段，1500 字足够。

### 3.3 调大调小的后果

- **调小（如 256）**：回答被截断，可能半句话就断了。适合"一句话回答"场景
- **调大（如 4096）**：允许更详细的回答，但如果模型喜欢"废话"，会生成冗长内容；且 API 按 token 计费，更贵

### 3.4 怎么改

```bash
# .env 文件
LLM_MAX_TOKENS=1024    # 改短可以省钱、加速
```

### 3.5 特殊说明：Ollama 的参数名不同

Ollama 的 API 里 `max_tokens` 不叫 `max_tokens`，而叫 `num_predict`，且必须放在 `options` 里。代码里已经处理了这个差异：

```python
# OllamaGenerator 里
"options": {
    "num_predict": self.max_tokens,    # ← Ollama 专用名称
}
```

---

## 4. top_p——核采样

### 4.1 它是什么

top_p（核采样）是另一种控制随机性的方式。它不是像 temperature 那样调整概率分布，而是**直接砍掉低概率的词**：

```
top_p = 1.0（默认）
  所有词都保留，不做任何过滤
  → 随机性完全由 temperature 控制

top_p = 0.9
  只保留累计概率达到 90% 的词，剩下 10% 概率的词被丢弃
  → 过滤掉"不太可能"的词，减少荒谬输出

top_p = 0.1
  只保留累计概率达到 10% 的词（通常只有最top的 1~2 个词）
  → 效果类似于 temperature=0，非常确定
```

### 4.2 例子

假设模型在生成"今天天气很"后面一个词时，概率分布是：

```
好    0.60
热    0.20
冷    0.10
不错  0.05
糟糕  0.03
奇怪  0.02
```

- `top_p=1.0`：所有词都可能被选中（包括"糟糕"和"奇怪"）
- `top_p=0.8`：只保留"好"(0.6)+"热"(0.2)=0.8，后面的词全部丢弃
- `top_p=0.6`：只保留"好"(0.6)，等于确定性选择

### 4.3 为什么默认 1.0

**经验法则：不要同时调 temperature 和 top_p。** 选一个控制即可。

本项目用 `temperature=0.0` 已经实现了确定性输出，`top_p=1.0`（不限制）就不会干扰。如果你同时设 `temperature=0.7` 和 `top_p=0.5`，两个参数会互相影响，效果难以预测。

### 4.4 什么时候用 top_p 而不是 temperature

| 情况 | 推荐方案 |
|------|----------|
| RAG 知识问答 | temperature=0, top_p=1.0 |
| 聊天但不想出现奇怪词 | temperature=0.7, top_p=0.9 |
| 创意写作 | temperature=0.9, top_p=1.0 |
| 严格不跑题 | temperature=0.3, top_p=0.85 |

### 4.5 怎么改

```bash
# .env 文件
LLM_TOP_P=0.9    # 如果想过滤低概率词
```

---

## 5. 参数组合速查表

| 场景 | temperature | max_tokens | top_p | 说明 |
|------|-------------|------------|-------|------|
| **RAG 知识问答（默认）** | **0.0** | **2048** | **1.0** | **本项目默认，确定性输出** |
| 简短问答（只要一句话） | 0.0 | 256 | 1.0 | 省钱省时间 |
| 客服对话（要自然） | 0.3 | 1024 | 1.0 | 略带变化但不跑偏 |
| 多文档综合分析 | 0.0 | 4096 | 1.0 | 需要更长回答 |
| 头脑风暴（非 RAG） | 0.9 | 2048 | 1.0 | 鼓励创意 |
| 严格不跑题 | 0.1 | 1024 | 0.85 | 双重保险 |

---

## 6. 不同 LLM 提供商的差异

### 6.1 智谱 GLM

| 参数 | 支持范围 | 注意事项 |
|------|----------|----------|
| temperature | 0.0 ~ 1.0 | 超过 1.0 可能报错（智谱上限比 OpenAI 低） |
| max_tokens | 最大 4096 | glm-4-flash 免费版可能限制更低 |
| top_p | 0.0 ~ 1.0 | 同 OpenAI |

```bash
# 智谱推荐配置
LLM_PROVIDER=zhipu
LLM_MODEL=glm-4-flash         # 免费版
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_TOP_P=1.0
```

### 6.2 OpenAI

| 参数 | 支持范围 | 注意事项 |
|------|----------|----------|
| temperature | 0.0 ~ 2.0 | 范围最宽 |
| max_tokens | 视模型而定 | gpt-4o-mini 最大 16384 |
| top_p | 0.0 ~ 1.0 | — |

```bash
# OpenAI 推荐配置
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_TOP_P=1.0
```

### 6.3 Ollama 本地

| 参数 | Ollama 参数名 | 注意事项 |
|------|---------------|----------|
| temperature | `temperature` | 放在 `options` 里 |
| max_tokens | `num_predict` | **名字不同！** |
| top_p | `top_p` | 放在 `options` 里 |

```bash
# Ollama 推荐配置
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_TOP_P=1.0
```

代码中已经处理了 Ollama 的参数名差异（`num_predict` vs `max_tokens`），你只需要在 `.env` 里统一用 `LLM_MAX_TOKENS` 即可。

---

## 7. 参数校验

项目在 [app/config.py](file:///d:/TRAE%20SOLO%20CN/trae_chat/6a66fce4e6efa496eef7921c/kb-rag/app/config.py) 里对这三个参数做了校验：

```python
@field_validator("llm_temperature")
def _temperature_range(cls, v):
    if v < 0.0 or v > 2.0:
        raise ValueError("llm_temperature must be between 0.0 and 2.0")
    return v

@field_validator("llm_top_p")
def _top_p_range(cls, v):
    if v <= 0.0 or v > 1.0:
        raise ValueError("llm_top_p must be between 0.0 and 1.0")
    return v

@field_validator("llm_max_tokens")
def _max_tokens_positive(cls, v):
    if v <= 0:
        raise ValueError("llm_max_tokens must be positive")
    return v
```

如果你在 `.env` 里填了非法值（如 `LLM_TEMPERATURE=3.0`），启动时会直接报错，不会等到运行时才发现。

---

## 8. 调优实战：常见问题怎么调

### 问题 1：回答有幻觉（编造不存在的信息）

```
现象：问"X-2025 续航多久"，模型回答"续航 20 小时"（实际是 12 小时）
```

**调法**：

| 步骤 | 操作 | 理由 |
|------|------|------|
| 1 | 确认 `LLM_TEMPERATURE=0.0` | 高 temperature 会让模型"自由发挥" |
| 2 | 调低 `RERANK_THRESHOLD`（如 0.2） | 可能是相关内容被拒答了，模型在"猜" |
| 3 | 检查检索结果 | 可能根本没检索到正确片段，问题不在 LLM |

### 问题 2：回答被截断（话没说完就断了）

```
现象：回答到"X-2025 支持蓝牙"就突然结束了
```

**调法**：调大 `LLM_MAX_TOKENS`

```bash
LLM_MAX_TOKENS=4096    # 从 2048 调到 4096
```

### 问题 3：回答太啰嗦

```
现象：问"续航多久"，回答了 1000 字
```

**调法**：

| 步骤 | 操作 | 理由 |
|------|------|------|
| 1 | 调小 `LLM_MAX_TOKENS=512` | 物理限制回答长度 |
| 2 | 修改 system prompt | 加上"请简短回答" |

### 问题 4：同一个问题每次回答不一样

```
现象：问"X-2025 续航多久"，第一次说 12 小时，第二次说"约 12 小时"
```

**调法**：确认 `LLM_TEMPERATURE=0.0`。如果已经是 0 还有差异，可能是：
- OpenAI 的 `temperature=0` 不是真正确定性的（极少数情况下有微小差异）
- 智谱同理。Ollama 的 `temperature=0` 是真正确定性的

### 问题 5：回答太死板/机械

```
现象：回答像机器人在念稿，没有自然语言的感觉
```

**调法**：适当调高 temperature

```bash
LLM_TEMPERATURE=0.3    # 从 0 调到 0.3，略带自然感
```

---

## 9. 完整 .env 配置示例

### 9.1 智谱 GLM（国内推荐）

```bash
LLM_PROVIDER=zhipu
LLM_MODEL=glm-4-flash
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_TOP_P=1.0
ZHIPU_API_KEY=你的key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4
```

### 9.2 OpenAI GPT

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_TOP_P=1.0
OPENAI_API_KEY=你的key
OPENAI_BASE_URL=https://api.openai.com/v1
```

### 9.3 Ollama 本地（完全私有化）

```bash
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048
LLM_TOP_P=1.0
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 10. 参数依赖关系

```
LLM_TEMPERATURE ──┐
                   ├──→ chat.completions.create() ──→ LLM 回答
LLM_MAX_TOKENS ────┤
                   │
LLM_TOP_P ─────────┘
     │
     └── 注意：temperature 和 top_p 不要同时"生效"
         推荐组合：temperature=0 + top_p=1.0
                   或 temperature=0.7 + top_p=1.0
         不推荐：  temperature=0.7 + top_p=0.5（互相干扰）
```

另外，LLM 参数和 RAG 其他参数之间也有关系：

| LLM 参数 | 相关 RAG 参数 | 关系 |
|----------|--------------|------|
| `max_tokens` | `RERANK_TOP_K` | top_k 越大→送入上下文越多→需要更大 max_tokens 才能覆盖所有片段 |
| `temperature` | `RERANK_THRESHOLD` | temperature 高时更依赖 threshold 拦截无关内容 |
| `max_tokens` | `CHUNK_SIZE` | chunk_size 越大→单块信息越多→回答可能更长→需要更大 max_tokens |
