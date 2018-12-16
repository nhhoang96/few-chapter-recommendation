#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from skm3 import CSOClassifier as CSO
from sys import argv
import re, math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import csv
import collections

WORD = re.compile(r'\w+')
csv_output = open('Summary_Few_Chapter_' + argv[1] +'.csv', 'w')
csv_writer = csv.writer(csv_output)
csv_writer.writerow(['Chapter Included', "Number of topics", 'Ranked', 'Jaccard', 'Cosine'])

# Compute Jaccard similarity from 2 vectors
def compute_jaccard_similarity( list1, list2):
    intersection = float(len(list(set(list1).intersection(list2))))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)

# Compute Cosine Similarity
def compute_cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# Compute Ranked Semantic Matrix between
def calculate_rank_matrix(standard_topics, compared_topics, extended_topics, standard_topic_weights):
    rank_matrix = np.zeros((len(standard_topics), len(compared_topics)), dtype=np.float64)

    for j in range (rank_matrix.shape[0]):
        rank_matrix[:, j] = standard_topic_weights

    for column in range(rank_matrix.shape[1]):
        for row in range(rank_matrix.shape[0]):
            if (standard_topics[row] == compared_topics[column]):
                rank_matrix[row, column] *= 1.0
            elif ((standard_topics[row] != compared_topics[column]) and (compared_topics[column] in extended_topics)):
                rank_matrix[row, column] *= 0.5
            else:
                rank_matrix[row, column] *= -0.1

    sum_val = rank_matrix.sum(axis = 0).reshape(-1,len(compared_topics))
    min_val = float(np.min(sum_val))
    max_val = float(np.max(sum_val))

    for i in range(sum_val.shape[1]):
        sum_val[:,i] = (sum_val[:,i] - min_val) / ((max_val - min_val) + 1e-19)

    rank_matrix = np.append(arr= rank_matrix,values=sum_val, axis = 0)

    rank_feature = {}
    for i in range(len(compared_topics)):
        rank_feature[compared_topics[i]] = float(sum_val[:,i])

    return rank_matrix, rank_feature

# Compute similarity_score using rank_matrix
def use_rank_matrix(testing_topic_list, rank_feature):
    rank_score = 0.0;
    for topic in testing_topic_list:
        if (topic in rank_feature.keys()):
            rank_score += rank_feature[topic]
    return rank_score

# Compute topics for conferences and extended chapters of the books
# If original_topics = None, it means we are looking at
# the original book topics (few-chapter books)
def compute_topics(products, clf, original_topics =None):
    full_product_topics_list = []
    full_product_topics = set()
    related_topics = set()
    # Create new product list and product set
    if (original_topics != None):
        for product in original_topics:
            full_product_topics.add(product)

        for product in full_product_topics:
            full_product_topics_list.append(product)

    # Find topics from the new set of products
    for product in products:
        product_topics = clf.classify(product, num_narrower=1, min_similarity=0.9, climb_ont='jfb', verbose=False)
        for item, value in product_topics.items():
            for v in value:
                topic_name = v.replace('-', '_').lower().strip(' ')
                full_product_topics.add(topic_name)
                related_topics.add(topic_name)
    full_product_topics_list = []

    for topic in full_product_topics:
        full_product_topics_list.append(topic)

    if (original_topics == None):
        return full_product_topics, full_product_topics_list
    return full_product_topics, full_product_topics_list, related_topics


# Convert sentences to vectors to get prepared
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

clf = CSO(version=2)
clf.load_cso()

#------------------------------ Obtain Conference topics ------------------------------------
f = open('iswc_topic', 'r')

conf_topics = ' '
for line in f.readlines():
    conf_topics +=(line.strip('\n').lower().strip(' ').replace('-', '_')) + ' '

conf = {
    'title':'International Semantic Web Conference',
    'abstract': conf_topics,
    'keywords': ''
}

#------------------------------ Extract conference topics ------------------------------------=#
conf_list = [conf]
full_conf_topics, full_conf_topics_list = compute_topics(conf_list, clf, None)

# ------------ Obtain set of topics from the book itself --------------------------------------#
book_list =[]
xlabels = []
with open('short_book.json') as f:
    data = json.load(f, object_pairs_hook=collections.OrderedDict)
    for key, value in data.items():
        xlabels.append(key.split('_')[1])
        book_list.append(value)

ranked_similarities = []
cosine_similarities = []
jaccard_similarities = []
num_topics =[]
book = []

# Go through each section/ chapter for the few-chapter book
for index in range (8):
    if (argv[1] != 'cumulative'):
        book = []
    book.append(book_list[index])
    book_topics_map = {}
    for chap in book:
        result = clf.classify(chap, num_narrower=1, min_similarity=0.9, climb_ont='jfb', verbose=False)
        for item, value in result.items():
            for v in value:
                topic_name = v.replace('-', '_').lower()
                if (topic_name in book_topics_map):
                    book_topics_map[topic_name] += 1
                else:
                    book_topics_map[topic_name] = 1

    book_topics = set(book_topics_map.keys())
    book_topic_weights = []
    book_topics_list = []

    for weight in book_topics_map.values():
        book_topic_weights.append(float(weight))

    for topic in book_topics_map.keys():
        book_topics_list.append(str(topic))

    num_topics.append(len(book_topics))
    #-----------------------------------------------------------------------------------------#
    related_topics = set()
    conf_rank_matrix, rank_feature = calculate_rank_matrix(book_topics_list, book_topics_list, related_topics, book_topic_weights)
    rank_score = use_rank_matrix(full_conf_topics_list, rank_feature)
    rank_dem_score = use_rank_matrix(book_topics_list, rank_feature)
    ranked_similarity = rank_score / rank_dem_score

    conf_vec = text_to_vector(' '.join(full_conf_topics_list))
    book_vec = text_to_vector(' '.join(book_topics_list))
    cosine_simlarity = compute_cosine_similarity(conf_vec, book_vec)
    jaccard_similarity = compute_jaccard_similarity(conf_vec, book_vec)

    print ("Considered chapter: ", xlabels[index])
    print("Number of up-to-date chapters: ", len(book_topics))
    print ("Jaccard: ", jaccard_similarity)
    print ("Cosine: ", cosine_simlarity)
    print ("Ranked: ", ranked_similarity)
    print ("\n")
    ranked_similarities.append(ranked_similarity)
    jaccard_similarities.append(jaccard_similarity)
    cosine_similarities.append(cosine_simlarity)

    csv_writer.writerow([xlabels[index], len(book_topics), ranked_similarity, jaccard_similarity, cosine_simlarity])

ind = np.arange(len(xlabels))
width = 0.27
fig, ax1 = plt.subplots()
ax1.bar(ind, ranked_similarities, width, color ='r', align='center', alpha=0.5)
ax1.bar(ind + width, jaccard_similarities, width, color='y', align='center', alpha=0.5)
ax1.bar(ind + width * 2, cosine_similarities, width, color ='g', align='center', alpha=0.5)

ax1.set_xlabel('Included chapter')
ax1.set_ylabel('Similarity score', color = 'r')
ax1.set_xticks(ind + width)
ax1.set_xticklabels(xlabels)
ax1.tick_params('y', labelcolor = 'r')
ax1.legend(['Ranked', 'Jaccard', 'Cosine'], loc='upper right')

ax2 = ax1.twinx()
ax2.set_ylabel('Number of total topics', color = 'b')
ax2.tick_params('y', labelcolor ='b')
ax2.plot(num_topics, color = 'b')
ax2.legend([' Num topics'], loc='upper left')

plt.show()
print ("Results and plots are saved in your current directory now")
fig.savefig('Similarity scores for different ' + argv[1] + ' chapters.png')