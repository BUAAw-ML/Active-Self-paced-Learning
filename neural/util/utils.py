from __future__ import print_function
import torch
import os
import re
import codecs
import copy
import numpy as np

import random
random.seed(0)

from collections import Counter


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file

    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def tag_mapping(dataset):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [s[2] for s in dataset]
    dico = Counter(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(dataset, word_to_id, tag_to_id):

    def f(x): return x.lower()

    data = []
    for s in dataset:
        str_words_q = [w for w in s[0].split()]
        str_words_a = [w for w in s[1].split()]
        words_q = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']for w in str_words_q]
        words_a = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']for w in str_words_a]
        tag = tag_to_id[s[2]]
        data.append({
            'str_words_q': str_words_q,
            'str_words_a': str_words_a,
            'words_q': words_q,
            'words_a': words_a,
            'tag': tag,
        })
    return data


def pad_seq(seq, max_length, PAD_token=0):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def create_batches(dataset, batch_size, order, labeled_tag=False):

    newdata = copy.deepcopy(dataset)
    if order == 'sort':
        newdata.sort(key=lambda x: len(x['words']))

    elif order == 'random':
        random.shuffle(newdata)


    newdata = np.array(newdata)
    batches = []
    num_batches = np.ceil(len(dataset) / float(batch_size)).astype('int')

    for i in range(num_batches):

        batch_data = newdata[(i * batch_size):min(len(dataset), (i + 1) * batch_size)]

        words_seqs_q = [itm['words_q'] for itm in batch_data]
        words_seqs_a = [itm['words_a'] for itm in batch_data]
        target_seqs = [itm['tag'] for itm in batch_data]

        if 'weight' in batch_data[0]:
            weight_seqs = [itm['weight'] for itm in batch_data]

        else:
            weight_seqs = [1 for itm in batch_data]

        str_words_seqs_q = [itm['str_words_q'] for itm in batch_data]
        str_words_seqs_a = [itm['str_words_a'] for itm in batch_data]
        labeled = None
        if labeled_tag:
            labeled = [itm['labeled'] for itm in batch_data]

        '''
        words_seqs = [#单词的索引
            [2,1],
            [5,6,9]
        ]
        target_seqs = [0,1]#标签的id
        str_words_seqs = [
            ['hello','world'],
            ['good','morning','dear']
        ]
        zip(words_seqs, target_seqs, str_words_seqs, range(len(words_seqs)))
        =>
        [
          ([2,1],11,['hello','world'],0)
          ([5,6,9],22,['good','morning','dear'],1)
        ]
        '''

        sort_info = range(len(words_seqs_q))

        words_lengths_q = np.array([len(s) for s in words_seqs_q])
        words_lengths_a = np.array([len(s) for s in words_seqs_a])
        words_padded_q = np.array([pad_seq(s, np.max(words_lengths_q)) for s in words_seqs_q])
        words_padded_a = np.array([pad_seq(s, np.max(words_lengths_a)) for s in words_seqs_a])
        words_mask_q = (words_padded_q != 0).astype('int')
        words_mask_a = (words_padded_a != 0).astype('int')

        outputdict = {'words_q': words_padded_q,'words_a': words_padded_a,
                      'tags': target_seqs,'weight':weight_seqs, 'labeled': labeled,
                      'wordslen_q': words_lengths_q,'wordslen_a': words_lengths_a,
                      'tagsmask_q': words_mask_q,'tagsmask_a': words_mask_a,
                      'str_words_q': str_words_seqs_q,'str_words_a': str_words_seqs_a,
                      'sort_info': sort_info}

        batches.append(outputdict)

    return batches


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu) ** 2 / (2 * sigma ** 2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu) ** 2 / (2 * torch.exp(logsigma) ** 2)


def bayes_loss_function(l_pw, l_qw, l_likelihood, n_batches, batch_size):
    return ((1. / n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)

