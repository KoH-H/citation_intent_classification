# -*- coding: utf-8 -*-
import jsonlines
from pathlib import Path
import numpy as np
import pandas as pd
import json
import collections

# root_path = Path('/content/citation_classification/dataset')
# section = root_path / 'sections-scaffold-train.jsonl'
#
# section_text = []
# section_name = []
# section_dict = {'introduction': 0, 'related work': 1, 'method': 2, 'experiments': 3, 'conclusion': 4}
# with jsonlines.open(section, mode='r') as reader:
#     for row in reader:
#         section_text.append(row['text'])  # 原文
#         section_name.append(section_dict[row['section_name']])
# print(section_name)
# print(collections.Counter(section_name))
# section_location = pd.DataFrame(columns=['citation_context', 'citation_class_label'])
# for i in range(len(section_name)):
#     section_location.loc[i] = {'citation_context': section_text[i],
#                                'citation_class_label': section_name[i]}
# section_location.to_csv('/content/citation_classification/dataset/section_name.csv', sep=',', index=False)
train_set = pd.read_csv('dataset/act/SDP_train.csv', sep=',')

sample_submission = pd.read_csv('dataset/act/sample_submission.csv', sep=',')

label_description = {0: 'The cited paper provides relevant Background information or is part of the body of literature',
                     1: 'The citing paper expresses similarities or differences to, or disagrees with, the cited paper.',
                     2: 'The citing paper extends the methods, tools or data etc. of the cited paper.',
                     3: 'The cited paper may be a potential avenue for future work.',
                     4: 'The citing paper is directly motivated by the cited paper.',
                     5: 'The citing paper uses the methodology or tools created by the cited paper.'}

label_deslist = []
for index, row in train_set.iterrows():
    # print(label_description[row['citation_class_label']])
    label_deslist.append(label_description[row['citation_class_label']])
train_set['label_description'] = label_deslist
train_set.to_csv('dataset/new_SDP_train.csv', sep=',', index=False, encoding='utf-8')

sampel_label_list = []

for index, row in sample_submission.iterrows():
    sampel_label_list.append(label_description[row['citation_class_label']])
print(sampel_label_list)
sample_submission['label_description'] = sampel_label_list
sample_submission.to_csv('dataset/new_sample_submission.csv', sep=',', index=False, encoding='utf-8')
print(sample_submission)