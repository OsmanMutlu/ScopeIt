import json

def read_file(file_name):
    with open(file_name, "r") as fi:
        docs = [ json.loads(l) for l in fi ]

    return docs

def group_set(_set, batch_size=32, doc=True):
    groups, labels = [], []
    urls = set(x["url"] for x in _set)
    for u in urls:
        sents = [ d for d in _set if d["url"] == u ][:batch_size]
        sents = sorted(sents, key=lambda x: x["sent_num"])
        groups.append([x["sentence"] for x in sents ])
        labels.append([ x["label"] for x in sents ])

        # doc label
        if doc:
            labels[-1].append(int(1 in labels[-1]))

    return groups, labels