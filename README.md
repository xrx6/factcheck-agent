# factcheck-agent

基于大语言模型与检索增强(RAG)的事实核查 Agent:判断 AI 回答是否存在事实错误(三分类)。**数据自建、引擎自写、评测自做**的完整闭环项目。

> 前身:2026 春 cocraw 课程赛同题实现(LLM 拆解 → 检索 → 判决,演进存档见 `reference/`)。本项目从零重写,方法论复用,代码全新。

## 任务定义

输入:用户问题 + AI 回答。输出三分类:

- `0` 主需存在事实错误(核心需求答错)
- `1` 次需存在事实错误(主体正确,细节有错)
- `2` 无事实错误

## 目录

```
data/dataset.json    标注集:自建条目,带人工标签(A 层金标)
data/test_set.json   测试输入集:课程真实样例 30 条(去重清洗过,标签待人工补标)
solve.py             核查引擎 v1:事实拆解版(拆解→逐条判决→规则聚合;--strategy direct 可跑 v0 基线)
eval.py              评测脚本:accuracy / macro-F1 / 混淆矩阵 / 分类明细 / 标签分布
requirements.txt     依赖(仅 openai)
output/results.json  引擎输出
output/eval_report.json  评测报告
reference/           上学期同题实现的演进存档(空模板 → 检索转折 → 690 行最终版)
```

## 数据格式

```json
{"id": "fc_0001", "question": "...", "answer": "...", "label": 2, "note": "标注理由"}
```

## 环境要求

- Python 3.10+
- 依赖安装:`pip install -r requirements.txt`
- 复制 `.env.example` 为 `.env`,填入 `GLM_API_KEY`(可选 `GLM_BASE_URL`/`GLM_MODEL`)

## 快速开始

```bash
cp .env.example .env        # 填入 GLM_API_KEY
pip install -r requirements.txt

python solve.py             # 引擎 → output/results.json
python eval.py              # 评测 → output/eval_report.json
```

常用参数:`python solve.py --input data/test_set.json --limit 5`(只跑前 5 条调试);
`python solve.py --strategy direct`(跑 v0 基线做对比,建议输出到 `output/results_v0.json`);
`python eval.py --pred output/results_v0.json`(换预测文件对比)。

## 方法论来源

v1 的设计借鉴两个代表工作(论文与代码均可公开获取):

- **SAFE**(Google DeepMind, NeurIPS 2024):拆解时把 claim 改写成自包含句(补全主语、消解代词)——v1 拆解 prompt 的核心要求。[论文](https://arxiv.org/abs/2403.18802)
- **RefChecker**(Amazon, EMNLP 2024):逐条 claim 三值判决(支持/矛盾/无法验证),再聚合——v1 的判决层与聚合规则。[论文](https://arxiv.org/abs/2405.14486)

本任务的差异化点:聚合规则按「主需/次需」严重度分级(核心 claim 矛盾→0,仅细节矛盾→1),文献多为比例得分或整体二分类。

## 版本路线(每版跑一次 eval,分数记进 CHANGELOG)

- **v0** ✅ 纯 LLM 直接判断(无检索)——「引擎 → 输出 → 评测」闭环已打通,分数待补
- **v1** ✅ 事实拆解:自包含 claim 拆解(SAFE)→ 逐条三值判决(RefChecker)→ 规则聚合(核心矛盾→0/细节矛盾→1/否则→2),代码完成,分数待补
- **v2** 加搜索取证:拆解 → 搜索 → 证据判决(检索方案届时选:GLM 搜索 API / Playwright+Bing)
- **v3** 打磨:并发、兜底、prompt 迭代——复刻上学期走过的路

## 评测指标(eval.py 要算的)

- 总体 accuracy
- 三类各自的准确率与混淆(哪类最常判错)
- 标签分布(防止引擎偷懒全猜一类)

## 规则

- 密钥走仓库根的 `.env`,代码一律 `os.environ` 读取
- 每条数据必须带 `note`,标注理由说不清的条目宁可不用
