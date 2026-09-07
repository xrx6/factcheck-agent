#!/usr/bin/env python3
"""
事实性校验引擎 —— 多 claim 核查，先核心后细节
版本：v5.5（2摘要500字符 + 详情800字符，总证据上限4500，详情页抓取重试上限3）
"""

import json
import os
import re
import time
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus

from openai import OpenAI
from playwright.sync_api import sync_playwright


# =============================================================================
# 全局配置
# =============================================================================
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "gpt-4.1")

# 搜索
BING_SEARCH_URL = "https://cn.bing.com/search?q="
MAX_SEARCH_RESULTS = 2               # 每个短语取 2 条摘要
SNIPPET_LIMIT = 400                  # 每条摘要统一截断至 400 字符
DETAIL_CHAR_LIMIT = 350              # 详情页内容截断长度
MAX_DETAIL_PAGES = 1                 # 每个短语最多成功抓取 1 个详情页
MAX_DETAIL_ATTEMPTS = 3              # 每个短语最多尝试抓取 3 次详情页

# 浏览器
HEADLESS = True
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
SEARCH_DELAY_MIN = 0.4
SEARCH_DELAY_MAX = 1.0
MAX_SEARCH_RETRIES = 2
BROWSER_TIMEOUT_MS = 15000

# 并发与输出
WORKERS = 2
RESULT_FILE = "output/results.json"
TOTAL_EVIDENCE_LIMIT = 3500          # 最终证据截断上限

# 虚构标记检测
MIN_ANSWER_LEN_QUICK = 300

# 日志
SEP_LINE = "─" * 70


# =============================================================================
# 调试日志
# =============================================================================
DEBUG = False
DEBUG_LOG_PATH = "debug.log"
_log_lock = threading.Lock()


def _log(message: str):
    if not DEBUG:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with _log_lock:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {message}\n")


# =============================================================================
# 虚构标记检测（快速路径）
# =============================================================================
def _has_fabrication_markers(answer: str) -> bool:
    markers = ["虚构", "编造", "不具备", "不可能", "不存在", "纯属虚构"]
    return any(m in answer for m in markers)


def _quick_check(item: dict):
    answer = item.get("answer", "")
    if len(answer) < MIN_ANSWER_LEN_QUICK:
        return None, None
    if _has_fabrication_markers(answer):
        return "0-主需存在事实错误", "回答自身含有虚构标记"
    return None, None


# =============================================================================
# 提示词
# =============================================================================
INTENT_CORE_PROMPT = """你是一个事实核查的预处理器。请根据用户问题和 AI 回答片段：
1. 用一句话概括用户核心意图。
2. 生成一个核心搜索短语（2‑12 字），用于验证核心论点。不要包含 AI 给出的具体数值。

输出纯 JSON：
{"intent": "用户意图", "core_phrase": "核心搜索短语"}
"""

DETAIL_EXTRACTOR_PROMPT = """你是一个事实核查助手。基于用户问题、AI 回答以及已获得的核心证据，找出 **1-2 条最可能出错的可查询细节**。
每条细节必须是包含具体数字、时间、人名或事件的独立事实，并为每条细节生成搜索短语（2‑12 字）。
即使核心证据看起来充分，也请尽量找出 1 条潜在的细节风险。如果确实没有，可返回空数组。

输出纯 JSON：
{"details": [{"claim": "细节1", "search_phrase": "搜索短语1"}, ...] 或 []}
"""

JUDGE_SCORING_PROMPT = """你是事实核查法官，按两步法判断 AI 回答的准确性。

示例（仅说明格式，不代表评判标准）：
用户问题：今天的澳门乒乓球比赛结果
AI 回答片段：覃予萱 3‑0 击败考夫蔓（11‑9,11‑8,11‑6）
核心证据：考夫蔓因伤退赛，覃予萱不战而胜
你的输出：{"question_type":"动态","intent":"澳门乒乓球比赛结果","main_correct":false,"main_reason":"证据显示对手退赛，比分虚构","minor_issues":["虚构具体比分"],"final_label":"0-主需存在事实错误","judgement":"核心事实虚构"}

### 第一步：判断核心论点
根据用户意图和核心搜索证据，判断核心主张是否与证据矛盾。
- 核心错误 → 直接评 0，结束。
- 核心正确（或无法验证但无明显矛盾）→ 进入第二步。

### 第二步：检查细节声明（如果有）
针对细节声明及其对应证据，检查是否存在证据明确矛盾。
- 核心正确 + 所有细节均无矛盾（或无细节）→ 2
- 核心正确 + 某条细节被明确证伪 → 1
- 细节中搜索不到证据不算错误，只算“无法验证”。

### 问题类型
- 动态：赛事、汇率、票房等实时变化问题
- 静态：历史事实、政策等固定信息
对于动态问题，若证据时间不匹配且无法确认矛盾，默认给 1（保守）。

### 输出格式（纯 JSON）
{
  "question_type": "动态/静态",
  "intent": "用户意图",
  "main_correct": true/false,
  "main_reason": "核心判断依据",
  "minor_issues": ["明确矛盾的细节错误"],
  "final_label": "0-主需存在事实错误 / 1-次需存在事实错误 / 2-无事实错误",
  "judgement": "综合理由"
}
"""


# =============================================================================
# 阶段 1：提取核心意图与搜索短语
# =============================================================================
def extract_core_intent(item, client):
    question = item.get("question", "")
    answer = item.get("answer", "")
    user_content = f"问题：{question}\nAI 回答片段：{answer[:800]}"
    messages = [
        {"role": "system", "content": INTENT_CORE_PROMPT},
        {"role": "user", "content": user_content}
    ]
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=512
        )
        raw = resp.choices[0].message.content.strip()
        _log(f"[核心提取] 耗时 {time.time()-t0:.1f}s，响应: {raw[:100]}")
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group(0))
        return {
            "intent": parsed.get("intent", ""),
            "core_phrase": parsed.get("core_phrase", question[:15])
        }
    except Exception as e:
        _log(f"[核心提取] 失败: {e}")
        return None


# =============================================================================
# 阶段 2：提取 1-2 条细节
# =============================================================================
def extract_details_if_needed(item, core_evidence, client):
    question = item.get("question", "")
    answer = item.get("answer", "")
    user_content = (
        f"问题：{question}\nAI 回答：{answer[:800]}\n"
        f"核心证据：{core_evidence[:500]}"
    )
    messages = [
        {"role": "system", "content": DETAIL_EXTRACTOR_PROMPT},
        {"role": "user", "content": user_content}
    ]
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=512
        )
        raw = resp.choices[0].message.content.strip()
        _log(f"[细节提取] 耗时 {time.time()-t0:.1f}s，响应: {raw[:100]}")
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group(0))
        details = parsed.get("details", [])
        valid = [d for d in details if d.get("claim") and d.get("search_phrase")]
        return valid[:2]  # 最多保留 2 条
    except Exception as e:
        _log(f"[细节提取] 失败: {e}")
        return []


# =============================================================================
# 浏览器与搜索基础函数
# =============================================================================
def _launch_browser():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=HEADLESS,
        args=[
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--no-sandbox'
        ]
    )
    ctx = browser.new_context(user_agent=BROWSER_UA)
    ctx.route("**/*", lambda route: (
        route.abort() if route.request.resource_type in ["image", "font"]
        else route.continue_()
    ))
    return pw, browser, ctx


def _fetch_article_text(page, url):
    """抓取详情页正文，并截断至 DETAIL_CHAR_LIMIT 字符。"""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=8000)
        page.wait_for_timeout(1000)
        content = ""
        for sel in ["article", "main", ".article-content", ".post-content", "#content", "body"]:
            el = page.query_selector(sel)
            if el:
                text = el.inner_text().strip()
                if len(text) > 50:
                    content = text
                    break
        if not content:
            return ""
        lines = [l.strip() for l in content.split('\n') if len(l.strip()) >= 10]
        lines = [l for l in lines if not any(
            kw in l for kw in ['广告', '推荐', '分享到', '关注', '微信扫一扫']
        )]
        full_text = '\n'.join(lines)
        return _trim_text(full_text, DETAIL_CHAR_LIMIT)
    except Exception as e:
        _log(f"[详情抓取] 异常 {url}: {e}")
        return ""


def _extract_snippet(elem, title="", href=""):
    """从 Bing 搜索结果节点提取干净摘要，并清理末尾多余省略号。"""
    for sel in [".b_caption p", ".b_snippet", "p"]:
        try:
            el = elem.query_selector(sel)
            if el:
                snip = el.inner_text().strip()
                if len(snip) >= 20:
                    return re.sub(r'[\s…\.]+$', '', snip).rstrip('…') or snip
        except:
            pass
    raw = (elem.inner_text() or "").strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    clean = []
    for line in lines:
        if title and line == title:
            continue
        if "http://" in line or "https://" in line:
            continue
        if "›" in line or re.search(r"\.(com|cn|org|net|gov|edu)\b", line, re.I):
            continue
        if len(line) < 10:
            continue
        clean.append(line)
    snippet = " ".join(clean).strip()
    snippet = re.sub(r'[\s…\.]+$', '', snippet).rstrip('…') or snippet
    return snippet


def _bing_search(page, query, time_hint=""):
    date_str = time_hint[:10] if time_hint else ""
    has_date = bool(
        re.search(r"\d{4}年\d{1,2}月\d{1,2}日", query) or
        re.search(r"\d{4}-\d{1,2}-\d{1,2}", query) or
        re.search(r"\d{1,2}月\d{1,2}日", query)
    )
    if date_str and not has_date and any(
        kw in query for kw in ["今天", "今日", "比赛", "比分", "汇率", "新闻"]
    ):
        search_q = f"{query} {date_str}"
    else:
        search_q = query

    for attempt in range(MAX_SEARCH_RETRIES):
        try:
            time.sleep(random.uniform(SEARCH_DELAY_MIN, SEARCH_DELAY_MAX))
            if "bing.com" not in page.url.lower():
                page.goto("https://www.bing.com/", timeout=BROWSER_TIMEOUT_MS, wait_until="domcontentloaded")
                page.wait_for_timeout(300)
            box = None
            for s in ["#sb_form_q", "textarea[name='q']", "input[name='q']"]:
                loc = page.locator(s).first
                if loc.count() > 0:
                    box = loc
                    break
            if not box:
                page.goto("https://www.bing.com/", timeout=BROWSER_TIMEOUT_MS, wait_until="domcontentloaded")
                page.wait_for_timeout(300)
                for s in ["#sb_form_q", "textarea[name='q']", "input[name='q']"]:
                    loc = page.locator(s).first
                    if loc.count() > 0:
                        box = loc
                        break
            if not box:
                page.goto(BING_SEARCH_URL + quote_plus(search_q), timeout=BROWSER_TIMEOUT_MS, wait_until="domcontentloaded")
            else:
                box.click()
                box.fill(search_q)
                page.wait_for_timeout(random.randint(120, 300))
                box.press("Enter")
            try:
                page.wait_for_url("**/search**", timeout=3500)
            except:
                pass
            try:
                page.wait_for_selector("#b_results li.b_algo:visible, .b_ans:visible", timeout=4500)
            except:
                pass
            if page.locator("#b_results li.b_algo:visible, .b_ans:visible").count() > 0:
                break
        except Exception as e:
            _log(f"[Bing] 异常: {e}")
            if attempt == MAX_SEARCH_RETRIES - 1:
                return []

    results = []
    try:
        card = page.query_selector(".b_ans, .b_focus, .b_card")
        if card and card.is_visible():
            txt = (card.inner_text() or "").strip()
            if not txt:
                txt = (card.text_content() or "").strip()
            if txt and len(txt) > 20:
                results.append({"title": "即时卡片", "url": page.url, "snippet": txt[:300]})
    except:
        pass
    try:
        items = page.query_selector_all("#b_results li.b_algo")
        vis = [it for it in items if it.is_visible() and it.bounding_box()]
        for it in vis:
            a_tag = None
            for s in ["h2 a", ".b_title a", "a[href]"]:
                cand = it.query_selector(s)
                if cand:
                    a_tag = cand
                    break
            if not a_tag:
                continue
            href = a_tag.get_attribute("href") or ""
            title = (a_tag.inner_text() or "").strip()
            if not href or "bing.com/search" in href or href.startswith("/"):
                continue
            snip = _extract_snippet(it, title=title, href=href)
            if len(snip) < 10:
                continue
            results.append({"title": title, "url": href, "snippet": snip[:SNIPPET_LIMIT]})
            if len(results) >= MAX_SEARCH_RESULTS:
                break
    except:
        pass
    return results


def _trim_text(text, max_len):
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    last_good = max(
        cut.rfind("。"), cut.rfind("！"), cut.rfind("？"),
        cut.rfind("\n"), cut.rfind("."), cut.rfind(";")
    )
    if last_good > max_len * 0.6:
        return cut[:last_good + 1].rstrip()
    return cut.rstrip()


# =============================================================================
# 证据收集（摘要 + 详情，带重试限制）
# =============================================================================
def _search_for_phrase(page, phrase, time_hint, label):
    short_label = label.split("\n")[0].strip()[:20]
    _log(f"[证据] {short_label} 搜索: {phrase}")
    t0 = time.time()
    hits = _bing_search(page, phrase, time_hint)
    elapsed = time.time() - t0
    if not hits:
        _log(f"[证据] {short_label} 无结果 ({elapsed:.1f}s)")
        return f"{label}：{phrase}\n无结果"

    _log(f"[证据] {short_label} 结果={len(hits)} ({elapsed:.1f}s)")

    parts = [f"{label}：{phrase}"]
    for i, h in enumerate(hits[:MAX_SEARCH_RESULTS], 1):
        parts.append(f"【结果{i}】{h['title']}\n摘要：{h['snippet']}")

    # 抓取详情页（最多尝试 MAX_DETAIL_ATTEMPTS 次，成功上限 MAX_DETAIL_PAGES）
    detail_cnt = 0
    attempt_cnt = 0
    for h in hits:
        if detail_cnt >= MAX_DETAIL_PAGES or attempt_cnt >= MAX_DETAIL_ATTEMPTS:
            break
        url = h.get("url", "")
        if not url or "bing.com" in url or "baidu.com" in url:
            continue
        attempt_cnt += 1
        dp = page.context.new_page()
        try:
            content = _fetch_article_text(dp, url)
            if content and len(content) > 100:
                parts.append(f"【详情】{h['title'][:30]}")
                parts.append(content)
                detail_cnt += 1
        except:
            pass
        finally:
            dp.close()
    _log(f"[证据] {short_label} 详情尝试{attempt_cnt}次，成功{detail_cnt}个")
    return "\n\n".join(parts)


def collect_all_evidence(page, claims_data, time_hint):
    core_phrase = claims_data["core_phrase"]
    details = claims_data.get("details", [])

    evidence_blocks = [
        _search_for_phrase(page, core_phrase, time_hint, "【核心核查】")
    ]

    for idx, detail in enumerate(details, 1):
        phrase = detail.get("search_phrase", "")
        claim_text = detail.get("claim", "")
        block = _search_for_phrase(
            page, phrase, time_hint,
            f"【细节{idx}】声明：{claim_text}\n搜索"
        )
        evidence_blocks.append(block)

    combined = "\n\n" + "=" * 40 + "\n\n".join(evidence_blocks)
    _log(f"[证据收集] 合并完成，总长度 {len(combined)} 字符")
    if len(combined) > TOTAL_EVIDENCE_LIMIT:
        combined = _trim_text(combined, TOTAL_EVIDENCE_LIMIT)
        _log(f"[证据收集] 截断至 {TOTAL_EVIDENCE_LIMIT} 字符")
    return combined


# =============================================================================
# 法官判定（直接取前 TOTAL_EVIDENCE_LIMIT 字符）
# =============================================================================
def evaluate(item, claims_data, evidence, client):
    question = item.get("question", "")
    answer = item.get("answer", "")
    time_str = item.get("time", "")

    evidence_for_judge = evidence[:TOTAL_EVIDENCE_LIMIT] if evidence else ""
    details_text = "\n".join(f"- {d['claim']}" for d in claims_data.get("details", []))

    evidence_guide = (
        "【证据说明】\n"
        "以下证据按“【核心核查】”和“【细节X】”分段。\n"
        "- 【核心核查】：用于验证 AI 回答的核心主张是否与事实矛盾。\n"
        "- 【细节X】：对应上方细节声明列表中的第 X 条细节，用于核查该细节是否有误。\n"
        "每一段内可能包含多条【结果N】（摘要）和可选的一条【详情】（网页正文截取）。\n"
        "请先依据【核心核查】判断核心主张，若核心正确，再逐个审查【细节X】是否被证据证伪。\n\n"
    )

    user_msg = f"""用户问题：{question}
提问时间：{time_str}
用户意图：{claims_data.get('intent', '')}
时间预警：{claims_data.get('time_alert', '无')}

AI 回答（截取）：
{answer[:1200]}

细节声明列表（需根据【细节X】证据核查）：
{details_text if details_text else '无'}

{evidence_guide}外部证据：
{evidence_for_judge if evidence_for_judge else '无证据'}

请按两步法判断：先核心，若核心正确再审查细节。"""

    # 打印真正送入法官的完整消息
    _log(f"[法官] 送入法官完整输入（长度{len(user_msg)}字符）:\n{user_msg}")

    messages = [
        {"role": "system", "content": JUDGE_SCORING_PROMPT},
        {"role": "user", "content": user_msg}
    ]
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.0, max_tokens=2048
        )
        raw = resp.choices[0].message.content.strip()
        _log(f"[法官] LLM耗时 {time.time()-t0:.1f}s，响应: {raw[:100]}")
        data = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group(0))
        label = data.get("final_label", "1-次需存在事实错误")
        reason = data.get("judgement", "")
        meta = {
            "main_correct": data.get("main_correct"),
            "minor_issues": data.get("minor_issues", []),
        }
        _log(f"[法官] 结果 → 标签: {label}，主正确: {meta['main_correct']}，次错误数: {len(meta['minor_issues'])}")
        return label, reason, meta
    except Exception as e:
        _log(f"[法官] 失败: {e}")
        return "1-次需存在事实错误", "法官调用失败", {}

# =============================================================================
# 后处理修正
# =============================================================================
def hard_fallback_if_no_evidence(evidence, label, reason):
    if "无搜索结果" in evidence[:100] or "无结果" in evidence[:100]:
        if label == "2-无事实错误":
            _log("[后处理] 核心证据缺失，强制降级 2 -> 1")
            return "1-次需存在事实错误", f"核心证据缺失，强制降级。{reason}"
    return label, reason


def align_label_with_meta(label, reason, meta):
    if not label or not reason or not meta:
        return label, reason

    main_ok = meta.get("main_correct")
    minors = meta.get("minor_issues", [])

    if main_ok is False:
        if label != "0-主需存在事实错误":
            _log(f"[后处理] main_correct=false，修正 {label} -> 0")
            return "0-主需存在事实错误", f"{reason} (修正：main_correct=false)"
    elif main_ok is True:
        if label == "0-主需存在事实错误":
            _log(f"[后处理] main_correct=true，修正 0 -> 1")
            return "1-次需存在事实错误", f"{reason} (修正：main_correct=true)"

    if label == "2-无事实错误" and minors:
        _log(f"[后处理] 存在次要错误，修正 2 -> 1")
        return "1-次需存在事实错误", f"{reason} (修正：存在minor_issues)"

    return label, reason


# =============================================================================
# 单条处理主流程（两阶段）
# =============================================================================
def process_one(item, client, page):
    qid = item.get("id", "unknown")
    _log(SEP_LINE)
    _log(f"📌 开始处理 {qid}")
    _log(SEP_LINE)
    t_start = time.time()

    label, reason = _quick_check(item)
    if label:
        _log(f"{qid} 快路径命中 → {label}，耗时 {time.time()-t_start:.1f}s")
        return {"id": qid, "label": label, "analyse": reason}

    core_data = extract_core_intent(item, client)
    if not core_data:
        _log(f"{qid} 核心提取失败，兜底判 1")
        return {"id": qid, "label": "1-次需存在事实错误", "analyse": "意图提取失败"}
    _log(f"{qid} 核心意图: {core_data['intent']}，短语: {core_data['core_phrase']}")

    core_evidence = _search_for_phrase(
        page, core_data["core_phrase"], item.get("time", ""), "【核心核查】"
    )
    _log(f"{qid} 核心证据长度: {len(core_evidence)}")

    details = extract_details_if_needed(item, core_evidence, client)
    if details:
        _log(f"{qid} 提取到 {len(details)} 条细节")

    claims_data = {
        "intent": core_data["intent"],
        "core_phrase": core_data["core_phrase"],
        "details": details,
        "time_alert": None
    }

    evidence = collect_all_evidence(page, claims_data, item.get("time", ""))
    _log(f"{qid} 最终证据长度: {len(evidence)}")

    t_judge = time.time()
    label, reason, meta = evaluate(item, claims_data, evidence, client)
    _log(f"{qid} 法官初判: {label}，耗时 {time.time()-t_judge:.1f}s")

    label, reason = hard_fallback_if_no_evidence(evidence, label, reason)
    label, reason = align_label_with_meta(label, reason, meta)

    _log(SEP_LINE)
    _log(f"🏁 {qid} 完成，最终标签: {label}，总耗时 {time.time()-t_start:.1f}s")
    _log(SEP_LINE + "\n")
    return {"id": qid, "label": label, "analyse": reason}


# =============================================================================
# 并行执行器
# =============================================================================
RESOURCE_LOCK = threading.Lock()
GLOBAL_RESOURCES = []


def main():
    print("🚀 启动事实性校验引擎（多claim核查）")
    if DEBUG:
        open(DEBUG_LOG_PATH, "w").close()
        _log("引擎启动")
    t_start = time.time()

    with open("data.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    local = threading.local()

    def _worker(entry):
        if not hasattr(local, "ctx"):
            pw, br, ctx = _launch_browser()
            local.pw = pw
            local.br = br
            local.ctx = ctx
            with RESOURCE_LOCK:
                GLOBAL_RESOURCES.append((pw, br, ctx))
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        pg = local.ctx.new_page()
        try:
            return process_one(entry, client, pg)
        except Exception as e:
            _log(f"任务异常: {e}")
            return {"id": entry.get("id"), "label": "1-次需存在事实错误", "analyse": f"异常: {str(e)[:50]}"}
        finally:
            pg.close()

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_map = {executor.submit(_worker, item): idx for idx, item in enumerate(dataset)}
            for fut in as_completed(future_map):
                idx = future_map[fut]
                try:
                    res = fut.result()
                    results.append((idx, res))
                    print(f"✅ 完成: {res['id']}")
                except Exception as e:
                    print(f"❌ 失败: {e}")
    finally:
        for pw, br, ctx in GLOBAL_RESOURCES:
            try:
                ctx.close()
            except:
                pass
            try:
                br.close()
            except:
                pass
            try:
                pw.stop()
            except:
                pass

    results.sort(key=lambda x: x[0])
    final = [r[1] for r in results]
    os.makedirs("output", exist_ok=True)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    dist = Counter(r["label"] for r in final)
    elapsed = time.time() - t_start
    print(f"\n{'='*60}\n运行完成，耗时 {elapsed:.1f}s")
    print("标签分布：")
    for l, c in sorted(dist.items()):
        print(f"  {l}: {c} ({c/len(final)*100:.1f}%)")
    _log("引擎结束")
    _log(f"标签分布: {dict(dist)}")


if __name__ == "__main__":
    main()