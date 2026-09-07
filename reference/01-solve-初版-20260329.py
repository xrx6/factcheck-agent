#!/usr/bin/env python3
import json, os

def classify(record):
    # 在此实现你的分类逻辑
    pass

def main():
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    os.makedirs("output", exist_ok=True)
    predictions = [{"id": r["id"], "label": classify(r)} for r in data]
    with open("output/results.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
