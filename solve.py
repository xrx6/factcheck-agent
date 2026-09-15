#!/usr/bin/env python3
"""
factcheck-agent v0 —— 纯 LLM 直接判断(无检索)

流程:读数据集 → 逐条送 LLM 三分类 → 写 output/results.json
标签:0 主需存在事实错误 / 1 次需存在事实错误 / 2 无事实错误

用法:
    python solve.py                                    # 跑 data/dataset.json
    python solve.py --input data/test_set.json --limit 5
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

from openai import OpenAI


# =============================================================================
# .env 加载(手写迷你版,避免引入 python-dotenv;真实环境变量优先于 .env)
# =============================================================================
def load_env(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


load_env()

API_KEY = os.environ.get("GLM_API_KEY")
BASE_URL = os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
MODEL_NAME = os.environ.get("GLM_MODEL", "glm-4.6")

MAX_RETRIES = 2        # 单条 LLM 调用(含解析)失败后的重试次数
MAX_TOKENS = 1024
ANSWER_CHAR_LIMIT = 4000
DEFAULT_INPUT = "data/dataset.json"
DEFAULT_OUTPUT = "output/results.json"


# =============================================================================
# 文本清洗:课程样例的 answer 里有 [citation:N] 标记和字面 \n,都是判决噪声
# =============================================================================
CITATION_RE = re.compile(r"\[citation:\d+\]")


def normalize_answer(text: str) -> str:
    text = CITATION_RE.sub("", text or "")
    text = text.replace("\\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =============================================================================
# 提示词:两步判决(先核心需求,后细节)——方法论沿用 reference/ 最终版,零样本
# =============================================================================
JUDGE_PROMPT = """你是事实核查法官。给定「用户问题」和某个 AI 的「回答」,判断该回答是否存在事实错误,输出三分类之一:

- 0 主需存在事实错误:回答在用户的核心需求上答错了(主体、结论或关键事实错误,含凭空虚构)
- 1 次需存在事实错误:回答主体正确,但存在细节性事实错误(数字、时间、人名、名称、地位、因果等与事实不符)
- 2 无事实错误:未发现事实错误

判断步骤(想清楚再给结论):
1. 用户的核心需求是什么?回答是否答到了点上、主体结论是否成立?
   主体错误或关键事实虚构 → 直接 0。
2. 主体正确时,逐个检查回答中的具体细节是否有明确事实错误。
   存在你确定与事实矛盾的细节 → 1;全部正确,或仅含糊、无法验证 → 2。

注意:
- 无法验证 ≠ 错误,只有你确定与事实矛盾时才计为错误。
- 时间敏感的问题(赛果、新闻、价格),结合「提问时间」与你的知识判断。
- 只看事实,不因回答的格式、立场、文风扣分。

输出纯 JSON(不要 markdown 代码块):
{"label": 0, "core": "用户核心需求一句话", "analyse": "指出错在哪、或为什么没错,须具体"}
"""


def extract_json(raw: str):
    """容忍 ```json 围栏与前后废话,取第一个 JSON 对象。"""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# =============================================================================
# 单条判决
# =============================================================================
def judge_one(item: dict, client: OpenAI) -> dict:
    qid = item.get("id", "unknown")
    question = (item.get("question") or "").strip()
    answer = normalize_answer(item.get("answer", ""))[:ANSWER_CHAR_LIMIT]
    time_str = item.get("time") or "未提供"

    user_msg = (
        f"用户问题:{question}\n"
        f"提问时间:{time_str}\n\n"
        f"AI 回答:\n{answer}\n\n"
        f"请按两步法判断:先核心需求,再细节,输出纯 JSON。"
    )
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    last_err = "未调用"
    for _ in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
            )
            data = extract_json(resp.choices[0].message.content)
            if data is None:
                last_err = "响应不是合法 JSON"
                continue
            try:
                label = int(data.get("label"))
            except (TypeError, ValueError):
                last_err = f"label 非法: {data.get('label')!r}"
                continue
            if label not in (0, 1, 2):
                last_err = f"label 越界: {label}"
                continue
            return {
                "id": qid,
                "label": label,
                "core": str(data.get("core", "")).strip(),
                "analyse": str(data.get("analyse", "")).strip() or "未给出理由",
            }
        except Exception as e:  # 网络/限流/超时等,重试
            last_err = str(e)[:120]

    # 兜底:调用或解析反复失败时保守判 1,并打 fallback 标记供 eval 单独统计
    return {
        "id": qid,
        "label": 1,
        "analyse": f"LLM 调用/解析失败,兜底判 1({last_err})",
        "fallback": True,
    }


# =============================================================================
# 主流程
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="factcheck-agent v0:纯 LLM 直接判断")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"输入数据集(默认 {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"输出文件(默认 {DEFAULT_OUTPUT})")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条(调试用)")
    return parser.parse_args()


def main():
    try:  # Windows 控制台可能是 gbk,避免中文打印崩
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()
    if not API_KEY:
        sys.exit("错误:未检测到 GLM_API_KEY。请复制 .env.example 为 .env 并填入密钥。")

    with open(args.input, encoding="utf-8") as f:
        dataset = json.load(f)
    items = dataset[: args.limit] if args.limit else dataset

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"引擎启动 v0(纯 LLM)| 模型 {MODEL_NAME} | 输入 {args.input} | 共 {len(items)} 条")

    results = []
    t_start = time.time()
    for i, item in enumerate(items, 1):
        t0 = time.time()
        res = judge_one(item, client)
        results.append(res)
        flag = " [兜底]" if res.get("fallback") else ""
        reason = res["analyse"][:60]
        print(f"[{i}/{len(items)}] {res['id']} → {res['label']}{flag} ({time.time() - t0:.1f}s) {reason}")

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    dist = Counter(r["label"] for r in results)
    fallback_cnt = sum(1 for r in results if r.get("fallback"))
    print(f"\n{'=' * 60}")
    print(f"运行完成,耗时 {time.time() - t_start:.1f}s,已写入 {args.output}")
    print(f"标签分布:{' | '.join(f'{k}: {dist.get(k, 0)}' for k in (0, 1, 2))}"
          f"(兜底 {fallback_cnt} 条)")


if __name__ == "__main__":
    main()
