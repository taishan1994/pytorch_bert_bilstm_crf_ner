import json


def get_data(path, mode="train"):
    with open(path, 'r', encoding="utf-8") as fp:
        data = fp.readlines()
    ents = set()
    res = []
    i = 0
    for d in data:
        d = json.loads(d)
        text = d["text"]
        if not text:
            continue
        entities = d["entity_list"]
        j = 0
        labels = []
        for entity in entities:
            entity_index = entity["entity_index"]
            start = entity_index["begin"]
            end = entity_index["end"]
            dtype = entity["entity_type"]
            ents.add(dtype)
            e = entity["entity"]
            labels.append([
                "T{}".format(j),
                dtype,
                start,
                end,
                e
            ])
            j += 1
        res.append({
            "id": i,
            "text": text,
            "labels": labels
        })
        i += 1

    with open("../mid_data/" + mode + ".json", "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False)

    if mode == "train":
        with open("../mid_data/" + "labels.json", "w", encoding="utf-8") as fp:
            json.dump(list(ents), fp, ensure_ascii=False)

        nor_ent2id = {"O": 0}
        i = 1
        for label in ents:
            nor_ent2id["B-" + label] = i
            i += 1
            nor_ent2id["I-" + label] = i
            i += 1
            nor_ent2id["E-" + label] = i
            i += 1
            nor_ent2id["S-" + label] = i
            i += 1

        with open("../mid_data/" + "nor_ent2id.json", "w", encoding="utf-8") as fp:
            json.dump(nor_ent2id, fp, ensure_ascii=False)


get_data("msra_train.txt", mode="train")
get_data("msra_1000.txt", mode="dev")