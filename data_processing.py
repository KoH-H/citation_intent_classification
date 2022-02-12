# -*- coding: utf-8 -*-
# import jsonlines
from pathlib import Path
import numpy as np
import pandas as pd
import re
# import json
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
train_set = pd.read_csv('dataset/act/SDP_test.csv', sep=',')
sam = pd.read_csv('dataset/act/sample_submission.csv', sep=',')
# print(train_set.shape)
# exit()


# for index, row in train_set.iterrows():
#     citation_text1 = re.sub(r'\[.*?\]', '', row['citation_context'])
#     citation_text2 = re.sub(r'\(.*?\)|\)|\.', '', citation_text1)
#     citation_text3 = re.sub(r'[0-9]+', '', citation_text2)
#     citation_text4 = nltk.word_tokenize(citation_text3)
#     citation_text = [word for word in citation_text4 if (word not in stop_words and len(word) > 1)]
#     if len(citation_text) == 0:
#         print(row['citation_context'])
#
# sample_submission = pd.read_csv('dataset/act/sample_submission.csv', sep=',')

label_description = {0: 'The cited paper provides relevant Background information or is part of the body of literature',
                     1: 'The citing paper expresses similarities or differences to, or disagrees with, the cited paper.',
                     2: 'The citing paper extends the methods, tools or data etc. of the cited paper.',
                     3: 'The cited paper may be a potential avenue for future work.',
                     4: 'The citing paper is directly motivated by the cited paper.',
                     5: 'The citing paper uses the methodology or tools created by the cited paper.'}

# 判断 是否 一篇文章有多个引用意图
# trai = train_set.drop_duplicates("cited_title", "first", inplace=True)
# print(trai)


value = dict()
resul = dict()
paper_list = []
for ind, row in train_set.iterrows():
    # print(row['citation_context'])
    res = re.findall(r"\[.*?\]", row['citation_context'])
    if len(res) != 0:
        num = 0
        list_v = None
        for i in res:
            if list_v is None:
                list_v = i.split(',')
            else:
                list_v.extend(i.split(','))
        # print(list_v)
        # print(len(list_v))
        listlen = len(list_v)
        if listlen == 15:
            paper_list.append(13)
        elif listlen == 17:
            paper_list.append(14)
        else:
            paper_list.append(listlen)

        continue

        # if len(res) > 1 | (len(res[0].split(',')) > 1):
        #     paper_list.append(len(res))
        #     resul[''.join(res)] = row['citation_class_label']
        #     if value.__contains__(row['citation_class_label']):
        #         value[row['citation_class_label']] = value.get(row['citation_class_label']) + 1
        #     else:
        #         value[row['citation_class_label']] = 1
        #     continue

    res1 = re.findall(r"\(.*?\)", row['citation_context'])
    if len(res1) > 0:
        # paper_list.append(len(ress))
        # print(res1)
        num = 0
        for i in res1:
            ress = re.findall("[0-9]{4}", i)
            num = num + len(ress)
        if num != 0:
            if num == 15:
                paper_list.append(13)
            elif num == 17:
                paper_list.append(14)
            else:
                paper_list.append(num)
            continue
        # print(num)
    paper_list.append(1)

print(len(paper_list))
sam['paper_list'] = paper_list
# print(train_set)
print(list(set(paper_list)))
sam.to_csv('dataset/act/citednum_sam.csv', sep=',', index=False, encoding='utf-8')
    # res1 = re.findall(r"\(.*?\)", row['citation_context'])
    # if len(res1) > 0:
    #     # paper_list.append(len(ress))
    #     print(res1)
    #     for i in res1:
    #         ress = re.findall("[0-9]{4}", i)
    #         print(ress)
    #         if len(ress) > 1:
    #             # paper_list.append(len(ress))
    #             resul[''.join(ress)] = row['citation_class_label']
    #             if value.__contains__(row['citation_class_label']):
    #                 value[row['citation_class_label']] = value.get(row['citation_class_label']) + 1
    #             else:
    #                 value[row['citation_class_label']] = 1
    #     continue
# print(value)
# train_set['paper_list'] = paper_list
# print(paper_list)
# print(train_set)
# print(len(resul))

# ress = re.findall("[0-9]{4}", res[0])
# label_deslist = []
# for index, row in train_set.iterrows():
#     context = row['citation_context']
#     # print(context)
#     label_des = label_description.get(row['citation_class_label'])
#     context = context + ' ' + label_des
#     # print(context)
#     train_set['citation_context'].loc[index] = context
#
#     # print(label_description[row['citation_class_label']])
#     # label_deslist.append(label_description[row['citation_class_label']])
# # train_set['label_description'] = label_deslist
# train_set.to_csv('dataset/act/new_SDP_train.csv', sep=',', index=False, encoding='utf-8')
#
# sampel_label_list = []
#
# for index, row in sample_submission.iterrows():
#     sampel_label_list.append(label_description[row['citation_class_label']])
# print(sampel_label_list)
# sample_submission['label_description'] = sampel_label_list
# sample_submission.to_csv('dataset/new_sample_submission.csv', sep=',', index=False, encoding='utf-8')
# print(sample_submission)

# def clear_section():
#     train_set = pd.read_csv('dataset/section_name.csv', sep=',')
#     index_list = []
#     for index, row in train_set.iterrows():
#         citation_text1 = re.sub(r'\[.*?\]', '', row['citation_context'])
#         citation_text2 = re.sub(r'\(.*?\)|\)|\.', '', citation_text1)
#         citation_text3 = re.sub(r'[0-9]+', '', citation_text2)
#         citation_text4 = nltk.word_tokenize(citation_text3)
#         citation_text = [word for word in citation_text4 if (word not in stop_words and len(word) > 1)]
#         if len(citation_text) == 0:
#             index_list.append(index)
#             print("index-------------", row['citation_context'])
#     new_set = train_set.drop(index=index_list)
#     print("line".center(20, "*"))
#     for index, row in new_set.iterrows():
#         citation_text1 = re.sub(r'\[.*?\]', '', row['citation_context'])
#         citation_text2 = re.sub(r'\(.*?\)|\)|\.', '', citation_text1)
#         citation_text3 = re.sub(r'[0-9]+', '', citation_text2)
#         citation_text4 = nltk.word_tokenize(citation_text3)
#         citation_text = [word for word in citation_text4 if (word not in stop_words and len(word) > 1)]
#         if len(citation_text) == 0:
#             print(row['citation_context'])
#     new_set.to_csv('dataset/new_section_name.csv', sep=',', index=False, encoding='utf-8')