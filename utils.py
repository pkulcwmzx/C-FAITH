import pandas as pd
import json
import os
import numpy as np
import csv
import math
import random

class data_item(object):
    def __init__(self, class1, class2, class3, class4, class5, risk_level, ent, src, knowledge):
        self.classes = []
        self.classes.append(class1)
        self.classes.append(class2)
        self.classes.append(class3)
        self.classes.append(class4)
        self.classes.append(class5)

        self.label = self.classes[-2]
        
        self.risk_level = risk_level
        self.ent = ent
        self.src = src
        self.knowledge = knowledge

    def to_dict(self):
        result_dict = {}
        result_dict["label"] = self.label
        result_dict["risk_level"] = self.risk_level
        result_dict["ent"] = self.ent
        result_dict["src"] = self.src
        result_dict["knowledge"] = self.knowledge
        return result_dict

# load and save data

def read_jsonl(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = []
        for line in f.readlines():
            json_data = json.loads(line)
            data.append(json_data)

    return data

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def to_jsonl(dicts, save_file):
    if not os.path.isdir(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with open(save_file, 'w', encoding='utf-8') as f:
        for line_dict in dicts:
            jsonl_line = f'{json.dumps(line_dict, cls = NpEncoder, ensure_ascii=False)}\n'
            f.write(jsonl_line)

def json2csv(path):
    data = read_jsonl(path)
    csv_path = 'data/llm_data.csv' # input csv data
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def csv2json(path):
    output_path  = "data/sample_data.jsonl" # output json data
    df = pd.read_csv(path, header=None, encoding="utf-8")
    idx_list, input_list, target_list = df[0].tolist(), df[1].tolist(), df[2].to_list()
    idx_list = idx_list[1:]
    input_list = input_list[1:]
    target_list = target_list[1:]
    results = []
    for i in range(len(idx_list)):
        new_dict = {}
        new_dict["idx"] = idx_list[i]
        new_dict["input"] = input_list[i]
        new_dict["target"] = target_list[i]
        results.append(new_dict)

    to_jsonl(results, output_path)

def xlsx2json(path):
    df = pd.read_excel(path, header=None)
    class1_list, class2_list, class3_list, class4_list, class5_list, risk_level_list, ent_list, src_list, knowledge_list = \
    df[0].tolist()[1:], df[1].tolist()[1:], df[2].tolist()[1:], df[3].tolist()[1:], df[4].tolist()[1:], df[5].tolist()[1:], df[6].tolist()[1:], df[7].tolist()[1:], df[8].tolist()[1:]
    
    class5_list = [x if isinstance(x, str) else "" for x in class5_list]
    risk_level_list = [x if isinstance(x, str) else "" for x in risk_level_list]
    data_list = []
    for i in range(len(class1_list)):
        data = data_item(class1_list[i], class2_list[i], class3_list[i], class4_list[i], class5_list[i], risk_level_list[i], ent_list[i], src_list[i], knowledge_list[i])
        data_list.append(data)

    return data_list

def sample_data(data):
    label_list = []
    for item in data:
        label = item.label
        label_list.append(label)

    label_list = list(set(label_list))
    label_list.remove("")

    sampled_labels = random.sample(label_list, 5)
    result_dicts = []
    for item in data:
        if item.label in sampled_labels:
            result_dicts.append(item.to_dict())

    return result_dicts

def get_ent_dict(data):
    ent2dict = {}
    deduplicated_data = []
    for item in data:
        assert len(item.classes) == 4 or len(item.classes) == 5
        item_dict = {}
        item_key = item.ent
        if item_key in ent2dict.keys():
            continue

        item_dict["class1"] = item.classes[0]
        item_dict["class2"] = item.classes[1]
        item_dict["class3"] = item.classes[2]
        item_dict["class4"] = item.classes[3]
        item_dict["class5"] = item.classes[4]
        
        ent2dict[item_key] = item_dict
        deduplicated_data.append(item)

    class2ent = {}
    for ent in ent2dict.keys():
        main_class = ent2dict[ent]["class4"]
        if main_class not in class2ent.keys():
            class2ent[main_class] = [ent]
        else:
            class2ent[main_class].append(ent)

    return [ent2dict, class2ent], deduplicated_data

def prepare_data(data):
    prepared_data = []
    for item in data:
        prepared_item = item.to_dict()
        prepared_data.append(prepared_item)

    return prepared_data

def sample_train_dev_data(class2ent, prepared_data, train_num=5, dev_num=2):
    train_ent = []
    dev_ent = []
    for item_class in class2ent.keys():
        item_ents = class2ent[item_class]
        sampled_ents = random.sample(item_ents, min(train_num + dev_num, len(item_ents)))
        train_ent += sampled_ents[:train_num]
        dev_ent += sampled_ents[train_num:]

    train_data = []
    dev_data = []
    for item_data in prepared_data:
        if item_data["ent"] in train_ent:
            train_data.append(item_data)
        elif item_data["ent"] in dev_ent:
            dev_data.append(item_data)
        else:
            continue

    return train_data, dev_data

if __name__ == '__main__':
    input_path="data/domain_data.xlsx"
    data = xlsx2json(input_path)
    output_path = "data/domain_data.jsonl"
    data_info, deduplicated_data = get_ent_dict(data)
    print(len(data_info[0]), len(data_info[1]))
    prepared_data = prepare_data(deduplicated_data)
    train_data, dev_data = sample_train_dev_data(data_info[1], prepared_data, train_num=3, dev_num=1)
    to_jsonl(prepared_data, output_path)
    to_jsonl(train_data, "data/train_data.jsonl")
    to_jsonl(dev_data, "data/dev_data.jsonl")