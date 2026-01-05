import json
import os

def save_code_label_pair(directory, codes, labels):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "source_codes.txt"), 'w', encoding='utf-8') as f_code, \
         open(os.path.join(directory, "source_labels.txt"), 'w', encoding='utf-8') as f_label:
        for c, l in zip(codes, labels):
            f_code.write(c + '\n')
            f_label.write(l + '\n')

def save_target_codes(directory, codes):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "target_codes.txt"), 'w', encoding='utf-8') as f_out:
        for c in codes:
            f_out.write(c + '\n')

def convert_codexglue_to_txt(train_file, valid_file):
    java_codes, java_labels = [], []
    c_codes, c_labels = [], []
    java_targets, c_targets = [], []

    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            code = data.get("func", "").replace('\n', ' ').strip()
            label = str(data.get("target", ""))
            lang = data.get("lang", "").lower() if "lang" in data else "java"

            if code and label:
                if "java" in lang:
                    java_codes.append(code)
                    java_labels.append(label)
                elif lang in {"c", "cpp", "c++"}:
                    c_codes.append(code)
                    c_labels.append(label)

    with open(valid_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            code = data.get("func", "").replace('\n', ' ').strip()
            lang = data.get("lang", "").lower() if "lang" in data else "java"

            if code:
                if "java" in lang:
                    java_targets.append(code)
                elif lang in {"c", "cpp", "c++"}:
                    c_targets.append(code)

    # 保存
    save_code_label_pair("data/java", java_codes, java_labels)
    save_code_label_pair("data/c", c_codes, c_labels)
    save_target_codes("data/java", java_targets)
    save_target_codes("data/c", c_targets)

    print("✅ 已完成语言划分数据生成：")
    print(f"  - Java：{len(java_codes)} 源样本，{len(java_targets)} 目标样本")
    print(f"  - C   ：{len(c_codes)} 源样本，{len(c_targets)} 目标样本")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    convert_codexglue_to_txt(
        train_file="train.jsonl",
        valid_file="valid.jsonl"
    )
