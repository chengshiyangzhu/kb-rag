# kb-rag RAG 评测报告

> 评测模式：{{MODE}}

## 整体指标（均值）

| 指标 | 数值 |
|------|------|
| Faithfulness | {{OVERALL_FAITHFULNESS}} |
| Answer Relevancy | {{OVERALL_ANSWER_RELEVANCY}} |
| Context Precision | {{OVERALL_CONTEXT_PRECISION}} |

## 指标说明

- **Faithfulness（忠实度）**：答案是否完全基于检索到的上下文，未引入臆造信息。取值范围 [0, 1]，越接近 1 越好。
- **Answer Relevancy（答案相关性）**：答案与问题的相关程度。取值范围 [0, 1]，越接近 1 越好。
- **Context Precision（上下文精确度）**：检索到的上下文中与 ground truth 相关的比例。取值范围 [0, 1]，越接近 1 越好。

> 当 ragas 不可用时，指标降级为启发式（heuristic）估算，仅用于粗略对比，不具严格语义。

## 逐条评测详情

{{DETAILS_TABLE}}
