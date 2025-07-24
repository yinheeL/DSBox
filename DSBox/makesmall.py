import json
import random


with open("data/devign/function.json", "r", encoding="utf-8") as f:
    data = json.load(f)


n = 500
if len(data) <= n:
    small = data
else:
    small = random.sample(data, n)


with open("data/devign/function_small.json", "w", encoding="utf-8") as f:
    json.dump(small, f, ensure_ascii=False, indent=2)

print(f"save in function_small.json")
