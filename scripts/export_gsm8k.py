from datasets import load_dataset
import json
import os


def main():
    ds = load_dataset("openai/gsm8k", "main")
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "gsm8k_train.jsonl")
    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in ds["train"]:
            json.dump({
                "prompt": ex.get("question") or "",
                "target": ex.get("answer") or "",
            }, f, ensure_ascii=False)
            f.write("\n")
            count += 1
    print(f"Wrote {count} train examples to {out_path}")


if __name__ == "__main__":
    main()
