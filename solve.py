#!/usr/bin/env python3
"""
factcheck-agent v1 —— 事实拆解版引擎

两种策略:
  decompose(默认, v1): 拆解 → 逐条判决 → 规则聚合
      借鉴 SAFE(自包含 claim 改写) 与 RefChecker(逐条三值判决),聚合规则对应本任务三分类:
      核心声明矛盾 → 0;仅细节矛盾 → 1;无矛盾/无法验证 → 2
  direct(v0 基线): 纯 LLM 直接三分类,prompt 保持不变以保证基线可比

流程:读数据集 → 逐条判决 → 写 output/results.json
用法:
    python solve.py                                        # v1 拆解版
    python solve.py --strategy direct                      # v0 基线
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

MAX_RETRIES = 2        # 单次 LLM 调用(含解析)失败后的重试次数
MAX_TOKENS = 1024
ANSWER_CHAR_LIMIT = 4000
MAX_CLAIMS = 6         # 每条回答最多拆出的 claim 数
VERDICTS = ("supported", "contradicted", "unverifiable")
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
# LLM 调用封装:带重试,返回 (解析后的dict | None, 最后一次错误)
# =============================================================================
def chat_json(client: OpenAI, system_prompt: str, user_msg: str):
    messages = [
        {"role": "system", "content": system_prompt},
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
            if data is not None:
                return data, None
            last_err = "响应不是合法 JSON"
        except Exception as e:  # 网络/限流/超时等
            last_err = str(e)[:120]
    return None, last_err


# =============================================================================
# 策略一(v0 基线):纯 LLM 直接判断。prompt 与 v0 完全一致,保证基线可比
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


def judge_direct(item: dict, client: OpenAI) -> dict:
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
    data, err = chat_json(client, JUDGE_PROMPT, user_msg)
    if data is not None:
        try:
            label = int(data.get("label"))
        except (TypeError, ValueError):
            label = None
        if label in (0, 1, 2):
            return {
                "id": qid,
                "label": label,
                "core": str(data.get("core", "")).strip(),
                "analyse": str(data.get("analyse", "")).strip() or "未给出理由",
            }
        err = f"label 非法: {data.get('label')!r}"

    # 兜底:调用或解析反复失败时保守判 1,并打 fallback 标记供 eval 单独统计
    return {
        "id": qid,
        "label": 1,
        "analyse": f"LLM 调用/解析失败,兜底判 1({err})",
        "fallback": True,
    }


# =============================================================================
# 策略二(v1):拆解 → 逐条三值判决 → 规则聚合
# =============================================================================
EXTRACT_PROMPT = """你是事实核查助手。读「用户问题」和「AI 回答」,把回答拆解成可独立验证的事实声明(claim):

要求:
1. 每条 claim 必须自包含:补全主语、消解代词,单独拿出来也能看懂。
   例如回答里"它全长约6300公里",要写成"长江全长约6300公里"。
2. 每条 claim 只含一个可验证事实(数字、时间、人名、事件、地位、因果等),不拆观点、评价和套话。
3. 给每条 claim 标注主次:
   - core: 直接回答用户核心需求的声明(用户问的就是它,错了用户就被误导)
   - detail: 回答里的辅助细节(背景、数字、时间、身份等)
4. 最多 """ + str(MAX_CLAIMS) + """ 条,优先保留 core 声明和最可能出错的细节。

输出纯 JSON(不要 markdown 代码块):
{"claims": [{"text": "自包含的声明", "type": "core", "why": "为什么拆这条,一句话"}, ...]}
没有可验证的事实时输出 {"claims": []}
"""


VERIFY_PROMPT = """你是事实核查判决员。逐条判断下列声明与事实的关系,基于你自己的知识:

- supported: 声明与事实相符
- contradicted: 声明与事实明确矛盾(你确定真实情况不是声明说的这样)
- unverifiable: 无法验证(超出知识范围、时效性问题、太模糊)

注意:
- 动态信息(赛果、新闻、价格)要结合「提问时间」判断,超出你知识时效的给 unverifiable,不要猜。
- 只有确定矛盾才给 contradicted;拿不准一律 unverifiable。
- 每条声明独立判断,不要互相影响。

输出纯 JSON(不要 markdown 代码块):
{"verdicts": [{"index": 0, "verdict": "supported", "reason": "一句依据"}, ...]}
index 对应声明编号,一条不落。
"""


def _extract_claims(item: dict, client: OpenAI):
    """拆解。返回 (claims列表 | None, 错误信息)。"""
    question = (item.get("question") or "").strip()
    answer = normalize_answer(item.get("answer", ""))[:ANSWER_CHAR_LIMIT]

    user_msg = f"用户问题:{question}\n\nAI 回答:\n{answer}\n\n请拆解为自包含的可验证声明。"
    data, err = chat_json(client, EXTRACT_PROMPT, user_msg)
    if data is None:
        return None, err
    raw_claims = data.get("claims")
    if not isinstance(raw_claims, list):
        return None, "claims 字段不是列表"

    claims = []
    for c in raw_claims[:MAX_CLAIMS]:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        claims.append({
            "text": text,
            "type": c.get("type") if c.get("type") in ("core", "detail") else "detail",
            "why": str(c.get("why", "")).strip(),
        })
    return claims, None


def _verify_claims(item: dict, claims: list, client: OpenAI):
    """逐条三值判决。返回 (判决后的claims | None, 错误信息)。"""
    question = (item.get("question") or "").strip()
    time_str = item.get("time") or "未提供"
    lines = "\n".join(f"{i}. {c['text']}" for i, c in enumerate(claims))

    user_msg = (
        f"用户问题:{question}(供理解语境,声明本身已自包含)\n"
        f"提问时间:{time_str}\n\n"
        f"声明列表:\n{lines}\n\n"
        f"请逐条给出判决。"
    )
    data, err = chat_json(client, VERIFY_PROMPT, user_msg)
    if data is None:
        return None, err
    raw_verdicts = data.get("verdicts")
    if not isinstance(raw_verdicts, list):
        return None, "verdicts 字段不是列表"

    by_index = {}
    for v in raw_verdicts:
        if isinstance(v, dict):
            try:
                by_index[int(v.get("index"))] = v
            except (TypeError, ValueError):
                continue

    judged = []
    for i, c in enumerate(claims):
        v = by_index.get(i, {})
        verdict = v.get("verdict")
        judged.append({
            **c,
            "verdict": verdict if verdict in VERDICTS else "unverifiable",
            "reason": str(v.get("reason", "")).strip() or ("判决缺失" if i not in by_index else ""),
        })
    return judged, None


def aggregate_verdicts(claims: list):
    """规则聚合:核心矛盾→0,仅细节矛盾→1,否则→2。确定性规则,不经过 LLM。"""
    contradicted = [c for c in claims if c["verdict"] == "contradicted"]
    core_bad = [c for c in contradicted if c["type"] == "core"]
    detail_bad = [c for c in contradicted if c["type"] == "detail"]
    n_sup = sum(1 for c in claims if c["verdict"] == "supported")
    n_unv = sum(1 for c in claims if c["verdict"] == "unverifiable")

    def _why(bad):
        parts = [f"{c['text']}({c['reason']})" for c in bad[:2]]
        return ";".join(parts)

    if core_bad:
        return 0, f"核心声明与事实矛盾:{_why(core_bad)}"
    if detail_bad:
        return 1, f"细节声明与事实矛盾:{_why(detail_bad)}"
    return 2, f"未发现事实矛盾({n_sup}条支持/{n_unv}条无法验证/共{len(claims)}条)"


def judge_decompose(item: dict, client: OpenAI) -> dict:
    qid = item.get("id", "unknown")

    claims, err = _extract_claims(item, client)
    if claims is None:
        # 拆解失败:降级为 v0 直接判断
        res = judge_direct(item, client)
        res["strategy"] = "decompose→direct(拆解失败降级)"
        res["analyse"] = f"拆解失败({err});降级直接判断:{res['analyse']}"
        return res
    if not claims:
        # 无可验证声明:回答是纯观点/套话,按任务定义无事实错误
        return {
            "id": qid, "label": 2, "strategy": "decompose",
            "claims": [], "analyse": "无可验证的事实声明(观点/套话)",
        }

    judged, err = _verify_claims(item, claims, client)
    if judged is None:
        res = judge_direct(item, client)
        res["strategy"] = "decompose→direct(判决失败降级)"
        res["analyse"] = f"逐条判决失败({err});降级直接判断:{res['analyse']}"
        return res

    label, analyse = aggregate_verdicts(judged)
    return {
        "id": qid, "label": label, "strategy": "decompose",
        "claims": judged, "analyse": analyse,
    }


# =============================================================================
# 主流程
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="factcheck-agent v1:事实拆解版核查引擎")
    parser.add_argument("--strategy", choices=["decompose", "direct"], default="decompose",
                        help="decompose=v1 拆解版(默认);direct=v0 纯 LLM 基线")
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
    judge_fn = judge_decompose if args.strategy == "decompose" else judge_direct
    print(f"引擎启动 v1-{args.strategy} | 模型 {MODEL_NAME} | 输入 {args.input} | 共 {len(items)} 条")

    results = []
    t_start = time.time()
    for i, item in enumerate(items, 1):
        t0 = time.time()
        res = judge_fn(item, client)
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
    degrade_cnt = sum(1 for r in results if r.get("strategy", "").startswith("decompose→"))
    print(f"\n{'=' * 60}")
    print(f"运行完成,耗时 {time.time() - t_start:.1f}s,已写入 {args.output}")
    print(f"标签分布:{' | '.join(f'{k}: {dist.get(k, 0)}' for k in (0, 1, 2))}"
          f"(兜底 {fallback_cnt} 条,降级 {degrade_cnt} 条)")


if __name__ == "__main__":
    main()
