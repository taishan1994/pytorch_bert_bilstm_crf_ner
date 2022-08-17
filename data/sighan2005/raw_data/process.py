import json
from pprint import pprint

max_seq_len= 512
# -1：索引为0-511，-2：去掉CLS和SEP，+1：词索引到下一位
max_seq_len = max_seq_len - 1 - 2 + 1

with open("training.txt", "r", encoding="utf-8") as fp:
    data = fp.readlines()

res = []
i = 0
for d in data:
    d = d.strip().split("  ")
    dtype = "word"
    start = 0
    tmp = []
    labels = []
    j = 0
    for word in d:
        start = len("".join(tmp))
        tmp.append(word)
        end = start + len(word)
        labels.append(["T{}".format(j), dtype, start, end, word])
        j += 1
        if end > max_seq_len:
            sub_tmp = tmp[:-1]
            sub_labels = labels[:-1]
            end = start + len("".join(sub_tmp))
            text = "".join(sub_tmp)
            res.append({
                "id": i,
                "text": text,
                "labels": sub_labels
            })

            start = 0
            tmp = [word]
            end = len("".join(tmp))
            labels = [["T{}".format(0), dtype, 0, end, word]]
            i += 1

    if tmp:
        text = "".join(tmp)
        res.append({
            "id": i,
            "text": text,
            "labels": labels
        })
        i += 1

with open("../mid_data/train.json", 'w', encoding="utf-8") as fp:
    json.dump(res, fp, ensure_ascii=False)

labels = ["word"]
with open("../mid_data/labels.json", 'w', encoding="utf-8") as fp:
    json.dump(labels, fp, ensure_ascii=False)

nor_ent2id = {"O":0, "B-word":1, "I-word":2, "E-word":3, "S-word":4}
with open("../mid_data/nor_ent2id.json", 'w', encoding="utf-8") as fp:
    json.dump(nor_ent2id, fp, ensure_ascii=False)

with open("test.txt", "r", encoding="utf-8") as fp:
    data = fp.readlines()

res = []
i = 0
for d in data:
    d = d.strip().split("  ")
    dtype = "word"
    start = 0
    tmp = []
    labels = []
    j = 0
    for word in d:
        start = len("".join(tmp))
        tmp.append(word)
        end = start + len(word)
        labels.append(["T{}".format(j), dtype, start, end, word])
        j += 1
        if end > max_seq_len:
            sub_tmp = tmp[:-1]
            sub_labels = labels[:-1]
            end = start + len("".join(sub_tmp))
            text = "".join(sub_tmp)
            res.append({
                "id": i,
                "text": text,
                "labels": sub_labels
            })

            start = 0
            tmp = [word]
            end = len("".join(tmp))
            labels = [["T{}".format(0), dtype, 0, end, word]]
            i += 1

    if tmp:
        text = "".join(tmp)
        res.append({
            "id": i,
            "text": text,
            "labels": labels
        })
        i += 1

with open("../mid_data/test.json", 'w', encoding="utf-8") as fp:
    json.dump(res, fp, ensure_ascii=False)