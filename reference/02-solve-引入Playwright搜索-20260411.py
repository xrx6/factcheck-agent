import json
import os
from openai import OpenAI
from urllib.parse import quote_plus
import time

# ====== Playwright 相关 ======
from playwright.sync_api import sync_playwright

SEARCH_ENGINE = "https://cn.bing.com/search?q="
MAX_SEARCH_RESULTS = 3
DETAIL_MAX_CHARS = 1000


def create_browser():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=True,
        args=[
            '--disable-blink-features=AutomationControlled',
            '--disable-dev-shm-usage',
            '--no-sandbox'
        ]
    )
    context = browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    return pw, browser, context


def search_bing(page, query):
    url = SEARCH_ENGINE + quote_plus(query)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            page.goto(url, timeout=30000, wait_until='commit')
            page.wait_for_timeout(3000)
            
            try:
                page.wait_for_selector("#b_results", timeout=10000)
            except:
                try:
                    page.wait_for_selector(".b_algo", timeout=5000)
                except:
                    page.wait_for_timeout(3000)
            
            if "captcha" in page.url.lower() or "challenge" in page.url.lower():
                print("检测到验证码，等待重试...")
                page.wait_for_timeout(5000)
                continue
                
            break
            
        except Exception as e:
            print(f"导航尝试 {attempt + 1} 失败: {e}")
            if attempt == max_retries - 1:
                return "搜索失败"
            page.wait_for_timeout(2000)
    
    results = []
    try:
        items = page.query_selector_all("#b_results > li.b_algo")
        if not items:
            items = page.query_selector_all(".b_algo")
            
        for item in items[:MAX_SEARCH_RESULTS]:
            try:
                title_el = item.query_selector("h2")
                title = title_el.inner_text() if title_el else ""
                
                snippet_el = item.query_selector(".b_caption p")
                if not snippet_el:
                    snippet_el = item.query_selector(".b_snippet")
                snippet = snippet_el.inner_text() if snippet_el else ""
                
                if title or snippet:
                    results.append(f"【标题】{title}\n【摘要】{snippet}")
            except:
                continue
    except Exception as e:
        print(f"提取搜索结果时出错: {e}")
    
    evidence_text = "\n\n".join(results)
    if not evidence_text:
        evidence_text = "未找到相关搜索结果"
    
    return evidence_text[:DETAIL_MAX_CHARS]


# ====== 新增：提取关键事实 ======
def extract_key_claims(client, question, answer, time="", session="", turn_number="", history=None):
    """
    用LLM从问答中提取需要验证的关键事实，转化为搜索关键词
    """
    prompt = f"""你是一个事实性校验助手。请从以下 AI 回答中提取需要验证的关键事实声明。

【对话上下文】
- 时间：{time}
- 会话编号：{session}
- 当前轮次：{turn_number}
- 历史问题：{history if history else "无"}

要求：
1. 提取 2-4 条最关键、最可能出错的事实声明
2. 每条声明应该是具体的、可搜索验证的（如具体时间、数字、人名、事件）
3. 将每条声明转化为适合搜索引擎查询的搜索词（简洁、精准，5-15个字）
4. ⚠️ 若是比赛、汇率、新闻类问题，必须结合“时间”判断是否已发生

输出格式（严格遵守，只输出JSON，不要任何解释）：
{{"claims": ["搜索词1", "搜索词2", "搜索词3"]}}

用户问题：{question}
AI回答：{answer}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        text = response.choices[0].message.content.strip()
        
        # 解析JSON
        try:
            result = json.loads(text)
            claims = result.get("claims", [])
            if claims and len(claims) > 0:
                return claims[:4]
        except:
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{[^{}]*"claims"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    claims = result.get("claims", [])
                    if claims:
                        return claims[:4]
                except:
                    pass
        
        # 兜底：返回问题本身
        return [question[:50]]
        
    except Exception as e:
        print(f"提取关键事实失败: {e}")
        return [question[:50]]


# ====== 判断函数（AI不联网，只基于证据） ======
def get_label(client, claim, evidence=""):
    prompt = f"""
你是一个专业的事实性校验助手。

【需要校验的内容】
{claim}

【搜索引擎证据（请基于此判断）】
{evidence if evidence else "未找到相关证据"}
【重要规则（最高优先级）】
##对于以下问题：
- 汇率（美元兑人民币等），重要：如果回答中的数据：没有事实错误，优先判为：2-无事实错误
- 金价、油价，重要：如果回答中的数据：没有事实错误，优先判为：2-无事实错误
-汇率，金价，油价不包括股票走势等市场分析，请严格按规则执行
- 若是询问比赛结果等事情，则需要具体看比赛是在当天的哪个时刻结束，要是回答存在还没发生的事情，优先判0
##关键区分原则：0 vs 1：
- 看错误是否影响用户对核心问题的理解：如果去掉这个错误，用户对事件的整体认知不变判1，如果核心认知被颠覆判0
- 不要过度严格：次要细节的小错误可以理解，只要核心正确就不判0
   
## 普通判定规则：
# 主要事实错误（核心事实错误）→ 输出：0-主需存在事实错误
- 回答的核心观点或关键事实明显错误，会从根本上改变用户对该事件的理解
- 非常重要的事件、时间或人物匹配错误，严重影响核心事实
- 用捏造的事实回答并严重影响核心事实

# 次要事实错误（细节错误、不确定）→ 输出：1-次需存在事实错误
- 存在非核心的细节偏差，但不会影响用户对事件整体情况的正确认识
- 表述存在问题，但不会影响用户对事件整体情况的正确认识

# 完全正确 → 输出：2-无事实错误
- 回答的内容完全符合事实，没有任何错误
- 若证据不足或无法验证，倾向于判为2

# 判断是否为预测性事情：
- 应结合询问内容判断

## 重要提示：
- 请严格基于上述"搜索引擎证据"进行判断
- 如果证据不足以证明错误，请判为2-无事实错误
- 不要自己编造或猜测证据之外的信息

## 输出格式（必须严格遵守）
只输出JSON，不要任何解释：

{{
  "label": "0-主需存在事实错误 / 1-次需存在事实错误 / 2-无事实错误",
  "analyse": "一句简要原因"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        text = response.choices[0].message.content.strip()

        # with open("debug.txt", "a", encoding="utf-8") as f:
        #     f.write("====== DEBUG ======\n")
        #     f.write(f"输入：{claim[:200]}\n")
        #     f.write(f"证据：{evidence[:300]}\n")
        #     f.write(f"输出：{text}\n")
        #     f.write("===================\n\n")

        try:
            result_json = json.loads(text)
            label = result_json.get("label", "2-无事实错误")
            analyse = result_json.get("analyse", "")
        except:
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{[^{}]*"label"[^{}]*\}', text, re.DOTALL)
            if json_match:
                try:
                    result_json = json.loads(json_match.group())
                    label = result_json.get("label", "2-无事实错误")
                    analyse = result_json.get("analyse", "")
                except:
                    label = "2-无事实错误"
                    analyse = "解析失败"
            else:
                label = "2-无事实错误"
                analyse = "JSON解析失败"

        return label, analyse

    except Exception as e:
        print("API错误:", e)
        return "1-次需存在事实错误", f"API异常: {str(e)[:50]}"


def solve():
    print("程序开始运行")

    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # ===== DeepSeek配置 =====
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1"
    )

    # ===== 启动浏览器 =====
    pw, browser, context = create_browser()
    page = context.new_page()

    results = []

    for idx, item in enumerate(data, 1):
        print(f"\n{'='*60}")
        print(f"处理第 {idx}/{len(data)} 条: {item.get('id')}")
        
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # ===== 步骤1：提取关键事实 =====
        print(f"  [1/3] 提取关键事实...")
        claims = extract_key_claims(
            client,
            question=question,
            answer=answer,
            time=item.get("time", ""),
            session=item.get("session", ""),
            turn_number=item.get("turn_number", ""),
            history=item.get("history_question", [])
        )
        print(f"  ✓ 提取到 {len(claims)} 个关键事实: {claims}")
        
        # ===== 步骤2：为每个关键事实搜索，收集证据 =====
        print(f"  [2/3] 搜索关键事实并收集证据...")
        all_evidence = []
        for i, claim_query in enumerate(claims, 1):
            print(f"    → 搜索 {i}/{len(claims)}: {claim_query[:40]}...")
            try:
                evidence = search_bing(page, claim_query)
                if evidence and evidence != "搜索失败" and evidence != "未找到相关搜索结果":
                    all_evidence.append(f"【搜索关键词 {i}】{claim_query}\n{evidence}")
                    print(f"      ✓ 找到证据 ({len(evidence)} 字符)")
                else:
                    print(f"      ✗ 未找到相关结果")
                time.sleep(1)  # 避免请求过快
            except Exception as e:
                print(f"      ✗ 搜索失败: {e}")
                continue
        
        # 合并所有证据
        if all_evidence:
            combined_evidence = "\n\n---\n\n".join(all_evidence)
        else:
            combined_evidence = "未找到相关搜索结果"
        
        # 构建完整的claim（保持原有格式）
        claim = f"""时间：{item.get('time')}
会话编号：{item.get('session')}
轮次：{item.get('turn_number')}
历史问题：{item.get('history_question')}

当前问题：{question}
回答：{answer}"""

        
        # ===== 步骤3：基于证据进行判断 =====
        print(f"  [3/3] 基于证据进行事实判断...")
        label, analyse = get_label(client, claim, combined_evidence)

        results.append({
            "id": item.get("id"),
            "label": label,
            "analyse": analyse
        })
        
        print(f"  ✓ 判断结果: {label}")
        print(f"  ✓ 判断理由: {analyse[:150]}...")

        # 增加间隔时间
        time.sleep(2)

    # 关闭浏览器
    context.close()
    browser.close()
    pw.stop()

    os.makedirs("output", exist_ok=True)

    with open("output/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("运行完成")
    print(f"结果已保存到 output/results.json")
    
    # 统计标签分布
    from collections import Counter
    label_counts = Counter(r["label"] for r in results)
    print("\n标签分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # 统计各标签占比
    total = len(results)
    print(f"\n总计: {total} 条")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count/total*100:.1f}%")


if __name__ == "__main__":
    solve()