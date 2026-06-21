# scripts/class_balance.py
import os
from collections import Counter
import yaml

def check_balance(merged_dir):
    with open(f"{merged_dir}/data.yaml") as f:
        classes = yaml.safe_load(f)["names"]

    counter = Counter()
    for split in ["train", "valid"]:
        lbl_dir = f"{merged_dir}/{split}/labels"
        if not os.path.exists(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            with open(os.path.join(lbl_dir, fname)) as f:
                for line in f:
                    cls_id = int(line.split()[0])
                    counter[classes[cls_id]] += 1

    print(f"\n{merged_dir} class distribution:")
    for cls, count in counter.most_common():
        print(f"  {cls}: {count}")

if __name__ == "__main__":
    check_balance("data/helmet_triple_mobile/merged")
    check_balance("data/seatbelt/merged")