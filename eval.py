#!/usr/bin/env python3
"""
评测:对比金标与 solve.py 输出

指标:总体 accuracy、macro-F1、混淆矩阵、三类各自的判对率与误判去向、
     标签分布(金标 vs 预测,防止引擎偷懒全猜一类)、兜底条目数。

用法:
    python eval.py                                        # 默认对比 data/dataset.json
    python eval.py --gold data/dataset.json --pred output/results.json
"""

import argparse
import json
import os
import sys
from collections import Counter

LABELS = (0, 1, 2)
LABEL_DESC = {0: "主需存在事实错误", 1: "次需存在事实错误", 2: "无事实错误"}
DEFAULT_GOLD = "data/dataset.json"
DEFAULT_PRED = "output/results.json"
DEFAULT_REPORT = "output/eval_report.json"


def load_items(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_ratio(num: int, den: int) -> str:
    return f"{num}/{den}" + (f"({num / den:.3f})" if den else "(--)")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="factcheck-agent 评测")
    parser.add_argument("--gold", default=DEFAULT_GOLD, help=f"金标文件(默认 {DEFAULT_GOLD})")
    parser.add_argument("--pred", default=DEFAULT_PRED, help=f"预测文件(默认 {DEFAULT_PRED})")
    parser.add_argument("--report", default=DEFAULT_REPORT, help=f"评测报告输出(默认 {DEFAULT_REPORT})")
    args = parser.parse_args()

    gold_items = load_items(args.gold)
    gold = {it["id"]: it["label"] for it in gold_items if it.get("label") in (0, 1, 2)}
    n_skipped = len(gold_items) - len(gold)  # label 为 null 的待补标条目

    pred_items = load_items(args.pred)
    preds = {}
    n_invalid_pred = 0
    for p in pred_items:
        if p.get("label") in (0, 1, 2):
            preds[p["id"]] = p
        else:
            n_invalid_pred += 1

    ids = [i for i in gold if i in preds]
    missing = [i for i in gold if i not in preds]
    extra = [i for i in preds if i not in gold]
    if not ids:
        sys.exit("错误:金标与预测没有可对比的条目(检查 id 是否对得上、金标 label 是否为 null)")

    n = len(ids)
    correct = sum(1 for i in ids if gold[i] == preds[i]["label"])
    accuracy = correct / n

    # 混淆矩阵 confusion[金标][预测]
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    for i in ids:
        confusion[gold[i]][preds[i]["label"]] += 1

    # 分类指标:recall 按「金标中出现过的类」算,precision 按「预测中出现过的类」算
    gold_dist = Counter(gold[i] for i in ids)
    pred_dist = Counter(preds[i]["label"] for i in ids)
    per_class = {}
    f1s = []
    for c in LABELS:
        tp = confusion[c][c]
        g_cnt, p_cnt = gold_dist.get(c, 0), pred_dist.get(c, 0)
        precision = tp / p_cnt if p_cnt else None
        recall = tp / g_cnt if g_cnt else None
        f1 = (2 * precision * recall / (precision + recall)) if tp else 0.0
        if g_cnt or p_cnt:
            f1s.append(f1)
        # 误判去向:该金标类被误判成哪些类、各多少条
        wrong_to = {p: cnt for p, cnt in confusion[c].items() if p != c and cnt}
        per_class[c] = {
            "desc": LABEL_DESC[c],
            "gold_count": g_cnt,
            "pred_count": p_cnt,
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "wrong_to": wrong_to,
        }
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    fallback_cnt = sum(1 for i in ids if preds[i].get("fallback"))

    # ---------------- 输出 ----------------
    print(f"{'=' * 60}")
    print(f"评测结果  对比 {n} 条"
          f"(金标 {len(gold_items)} 条,跳过无金标 {n_skipped} 条,"
          f"预测缺失 {len(missing)} 条,多余预测 {len(extra)} 条,非法预测 {n_invalid_pred} 条)")
    if missing:
        print(f"  预测缺失的 id:{', '.join(missing)}")
    if extra:
        print(f"  不在金标中的 id:{', '.join(extra)}")

    print(f"\n总体 accuracy : {fmt_ratio(correct, n)}")
    print(f"macro-F1      : {macro_f1:.3f}")

    header = "".join(f"{'pred=' + str(p):>10}" for p in LABELS)
    print(f"\n混淆矩阵(行=金标,列=预测):\n{'':>10}{header}")
    for g in LABELS:
        row = "".join(f"{confusion[g][p]:>10}" for p in LABELS)
        print(f"{'gold=' + str(g):<10}{row}")

    print("\n分类明细:")
    for c in LABELS:
        info = per_class[c]
        if not info["gold_count"] and not info["pred_count"]:
            continue
        prec = "无预测" if info["precision"] is None else f"{info['precision']:.3f}"
        rec = "--" if info["recall"] is None else f"{info['recall']:.3f}"
        line = (f"  {c} {info['desc']}: 判对 {info['correct']}/{info['gold_count'] or 0}"
                f"(precision {prec}, recall {rec})")
        if info["wrong_to"]:
            worst = max(info["wrong_to"].items(), key=lambda kv: kv[1])
            line += f" | 最常误判为 → {worst[0]}(×{worst[1]})"
        print(line)

    dist_line = " | ".join(
        f"{c}: 金标{gold_dist.get(c, 0)} vs 预测{pred_dist.get(c, 0)}" for c in LABELS
    )
    print(f"\n标签分布  {dist_line}")
    print(f"兜底条目  {fallback_cnt}(引擎调用失败硬编码的预测)")

    report = {
        "gold_file": args.gold,
        "pred_file": args.pred,
        "n_compared": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "confusion": {str(g): dict(confusion[g]) for g in LABELS},
        "per_class": {str(c): {k: v for k, v in per_class[c].items() if k != "desc"}
                      for c in LABELS},
        "gold_distribution": {str(c): gold_dist.get(c, 0) for c in LABELS},
        "pred_distribution": {str(c): pred_dist.get(c, 0) for c in LABELS},
        "fallback_count": fallback_cnt,
        "missing_pred_ids": missing,
        "extra_pred_ids": extra,
    }
    out_dir = os.path.dirname(os.path.abspath(args.report))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已写入 {args.report}")


if __name__ == "__main__":
    main()
