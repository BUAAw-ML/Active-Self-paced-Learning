import torch
from torch.autograd import Variable
import numpy as np
import time, json, copy, math
from scipy import stats
from scipy.spatial import distance_matrix
from neural.util.utils import *

from sklearn.metrics.pairwise import cosine_similarity

from neural.models.cnn import CNN

import math
import copy


class Acquisition(object):

    def __init__(self, train_data,seed=0, usecuda = True, answer_count = 5):
        self.answer_count = answer_count  #answer number for each question
        self.questions_num = int(len(train_data) / answer_count) #question number in the dataset
        self.train_index = set()  #the index of the labeled samples
        self.pseudo_train_data = []

        self.npr = np.random.RandomState(seed)
        self.usecuda = usecuda

    #-------------------------Random sampling-----------------------------
    def get_random(self, dataset, num_questions, returned = False):

            question_indices = [self.answer_count * x for x in range(self.questions_num)]
            random_indices = self.npr.permutation(self.questions_num)
            random_question_indices = [question_indices[x] for x in random_indices]

            cur_indices = set()
            sample_q_indices = set()
            i = 0
            while len(cur_indices) < num_questions * self.answer_count:
                if random_question_indices[i] not in self.train_index:
                    sample_q_indices.add(random_question_indices[i])
                    for k in range(self.answer_count):
                        cur_indices.add(random_question_indices[i] + k)
                i += 1
            if not returned:
                self.train_index.update(cur_indices)
            else:
                return sample_q_indices

    #--------------------------some related active learning methods: var，margin，entropy，me-em，lc
    def get_sampling(self, dataset, model_path, num_questions, batch_size = 1000, nsamp = 100, model_name = '',
                quota = 'me-em',
                _reverse = False,
                deterministic = False, #whether adopt bayesian neural network
                returned = False
                ):

        if quota == 'me-em' or quota == 'entropy'\
                or quota == 'mstd' or\
                quota == 'mstd-unregluar' or quota == 'mstd-regluar':
            _reverse = True

        model = torch.load(model_path)
        if deterministic:
            model.train(False)
            nsamp = 1
        else:
            model.train(True)
        tm = time.time()

        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]
        new_question_points = [new_datapoints[x * self.answer_count] for x in range(int(len(new_datapoints)/self.answer_count))]

        print("sample remaining in the pool:%d" % len(new_datapoints))

        data_batches = create_batches(new_dataset, batch_size = batch_size, order='no')

        pt = 0
        _delt_arr = []
        for data in data_batches:

            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda()
                words_a = Variable(torch.LongTensor(words_a)).cuda()
            else:
                words_q = Variable(torch.LongTensor(words_q))
                words_a = Variable(torch.LongTensor(words_a))

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            ###
            sort_info = data['sort_info']

            tag_arr = []
            score_arr = []
            real_tag_arr = []

            for itr in range(nsamp):
                if model_name == 'CNN':
                    score, tag = model.predict(words_q,
                                           words_a,
                                           usecuda=self.usecuda,
                                           scoreonly=False)
                elif model_name == 'BiLSTM':
                    score, tag = model.predict(words_q,
                                           words_a,
                                           wordslen_q, wordslen_a,
                                           usecuda=self.usecuda,
                                           scoreonly=False)


                st = sorted(zip(sort_info, score, tag, data['tags']),key=lambda p: p[0])
                _, origin_score, origin_tag, real_tag = zip(*st)

                tag_arr.append(list(origin_tag))
                score_arr.append(list(origin_score))
                real_tag_arr.append(list(real_tag))

            for i in range(len(score_arr)):
                for j in range(len(score_arr[i])):
                    if int(tag_arr[i][j]) == 0:
                        score_arr[i][j] = 1 - score_arr[i][j]

            new_score_seq = []
            for m in range(len(words_q)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            entropy_mean = []
            for i in range(int(len(new_score_seq) / self.answer_count)):
                all = 0
                for j in range(nsamp):
                    temp = []
                    for k in range(self.answer_count):
                        temp.append(new_score_seq[i * self.answer_count + k][j])
                    temp = np.array(temp)
                    temp = (temp + 1e-8) / np.sum(temp + 1e-8)
                    em = -np.sum((temp + 1e-8) * np.log2(temp + 1e-8))
                    all += em

                entropy_mean.append(float(float(all)/float(nsamp)))

            var_mean = []
            if quota == 'mstd-unregluar':
                for i in range(int(len(new_score_seq) / self.answer_count)):
                    all = 0
                    for j in range(self.answer_count):
                        temp = []
                        for k in range(nsamp):
                            temp.append(new_score_seq[i * self.answer_count + j][k])
                        temp = np.array(temp)
                        # temp = (temp + 1e-8) / np.sum(temp + 1e-8)
                        all += temp.std()
                    var_mean.append(float(float(all) / float(self.answer_count)))
            elif quota == 'mstd-regluar':
                for i in range(len(new_score_seq) // self.answer_count):
                    all = []
                    for j in range(nsamp):
                        temp = []
                        for k in range(self.answer_count):
                            temp.append(new_score_seq[i * self.answer_count + k][j])
                        temp = np.array(temp)
                        temp = (temp + 1e-8) / np.sum(temp + 1e-8)
                        all.append(temp)
                    all = np.array(all).T
                    assert all.shape == (self.answer_count, 100)
                    var_mean.append(np.mean(np.std(all, axis=1)))

            mean_score = []
            for i in range(len(new_score_seq) // self.answer_count):
                all = np.zeros(shape=(self.answer_count))
                for j in range(nsamp):
                    temp = []
                    for k in range(self.answer_count):
                        temp.append(new_score_seq[i * self.answer_count + k][j])
                    temp = np.array(temp)
                    temp = (temp + 1e-8) / np.sum(temp + 1e-8)
                    all += temp
                all /= nsamp
                mean_score.extend(all.tolist())

            un_regular_mean_score = []
            for arr in new_score_seq:
                all = 0
                for sc in arr:
                    all += sc
                un_regular_mean_score.append(float(float(all) / float(nsamp)))

            cutted_mean_score = []
            for i in range(int(len(mean_score)/self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(mean_score[i * self.answer_count + j])
                temp = sorted(temp, reverse = True)
                cutted_mean_score.append(temp)

            un_regular_cutted_mean_score = []
            for i in range(int(len(un_regular_mean_score) / self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(un_regular_mean_score[i * self.answer_count + j])
                un_regular_cutted_mean_score.append(temp)

            for i in range(len(cutted_mean_score)):
                item = cutted_mean_score[i]
                if quota == 'var':
                    _delt = np.array(item).var()
                elif quota == 'margin':
                    _delt = item[0] - item[1]
                elif quota == 'lc':
                    _delt = item[0]
                elif quota == "entropy":
                    item = np.array(item)
                    item = (item + 1e-8) / np.sum(item + 1e-8)
                    _delt = -np.sum((item + 1e-8) * np.log2(item + 1e-8))
                elif quota == "me-em":
                    item = np.array(item)
                    item = (item + 1e-8) / np.sum(item + 1e-8)
                    me = -np.sum((item + 1e-8) * np.log2(item + 1e-8))
                    _delt = me - entropy_mean[i]

                obj = {}
                obj["q_id"] = pt
                obj["real_id"] = new_question_points[pt]
                obj["delt"] = _delt
                obj["origin_score"] = un_regular_cutted_mean_score[i]
                _delt_arr.append(obj)

                pt += 1

        _delt_arr = sorted(_delt_arr, key=lambda o: o["delt"], reverse=_reverse)

        cur_indices = set()
        sample_q_indices = set()
        i = 0
        while len(cur_indices) < num_questions * self.answer_count:
            sample_q_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            for k in range(self.answer_count):
                cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + k)
            i += 1
        if  not returned:
            self.train_index.update(cur_indices)
            print ('time consuming： %d seconds:' % (time.time() - tm))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, sample_q_indices

    def get_BALD(self, dataset, model_path,
                 num_questions, batch_size=1000,
                 nsamp=100, model_name='',
                 top_1=True,
                 returned=False
                 ):

        model = torch.load(model_path)
        model.train(True)
        tm = time.time()

        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]
        new_question_points = [new_datapoints[x * self.answer_count] for x in
                               range(int(len(new_datapoints) / self.answer_count))]

        print("sample remaining in the pool:%d" % len(new_datapoints))

        data_batches = create_batches(new_dataset, batch_size=batch_size, order='no')

        pt = 0
        _delt_arr = []
        for data in data_batches:

            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda()
                words_a = Variable(torch.LongTensor(words_a)).cuda()
            else:
                words_q = Variable(torch.LongTensor(words_q)).cuda()
                words_a = Variable(torch.LongTensor(words_a)).cuda()

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            ###
            sort_info = data['sort_info']

            tag_arr = []
            score_arr = []
            real_tag_arr = []

            for itr in range(nsamp):

                if model_name == 'CNN':
                    score, tag = model.predict(words_q,
                                               words_a,
                                               usecuda=self.usecuda,
                                               scoreonly=False)
                elif model_name == 'BiLSTM':
                    score, tag = model.predict(words_q,
                                               words_a,
                                               wordslen_q, wordslen_a,
                                               usecuda=self.usecuda,
                                               scoreonly=False)

                st = sorted(zip(sort_info, score, tag, data['tags']), key=lambda p: p[0])
                _, origin_score, origin_tag, real_tag = zip(*st)

                tag_arr.append(list(origin_tag))
                score_arr.append(list(origin_score))
                real_tag_arr.append(list(real_tag))

            for i in range(len(score_arr)):
                for j in range(len(score_arr[i])):
                    if int(tag_arr[i][j]) == 0:
                        score_arr[i][j] = 1 - score_arr[i][j]

            new_score_seq = []
            for m in range(len(words_q)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            cutted_score_seq = []
            for i in range(int(len(new_score_seq) / self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(new_score_seq[i * self.answer_count + j])
                cutted_score_seq.append(temp)

            for item in cutted_score_seq:
                tp1 = np.transpose(np.array(item))
                _index = np.argsort(tp1, axis=1).tolist()

                if top_1:  # Only consider the first item in the rank
                    _index = np.argmax(tp1, axis=1)
                    _delt = stats.mode(_index)[1][0]
                else:
                    for i in range(len(_index)):
                        _index[i] = 10000 * _index[i][0] + 1000 * _index[i][1] + 100 * _index[i][2] + 10 * _index[i][
                            3] + _index[i][4]

                    _delt = stats.mode(np.array(_index))[1][0]

                obj = {}
                obj["q_id"] = pt
                obj["delt"] = _delt
                _delt_arr.append(obj)

                pt += 1

        _delt_arr = sorted(_delt_arr, key=lambda o: o["delt"])

        cur_indices = set()
        sample_q_indices = set()
        i = 0

        while len(cur_indices) < num_questions * self.answer_count:
            sample_q_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + 1)
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + 2)
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + 3)
            cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + 4)
            i += 1

        if not returned:
            self.train_index.update(cur_indices)
            print('time consuming： %d seconds:' % (time.time() - tm))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, sample_q_indices

    def get_DAL(self, dataset, model_path,
                   num_questions, batch_size=1000,
                   nsamp=100, model_name='',
                   returned = False,
                   ):

        model = torch.load(model_path)
        model.train(True)
        tm = time.time()

        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]
        new_question_points = [new_datapoints[x * self.answer_count] for x in
                               range(int(len(new_datapoints) / self.answer_count))]

        data_batches = create_batches(new_dataset, batch_size=batch_size, order='no')

        pt = 0
        _delt_arr = []

        for data in data_batches:

            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda()
                words_a = Variable(torch.LongTensor(words_a)).cuda()
            else:
                words_q = Variable(torch.LongTensor(words_q))
                words_a = Variable(torch.LongTensor(words_a))

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            sort_info = data['sort_info']

            tag_arr = []
            score_arr = []
            real_tag_arr = []

            for itr in range(nsamp):

                if model_name == 'CNN':
                    score, tag = model.predict(words_q,
                                               words_a,
                                               usecuda=self.usecuda,
                                               scoreonly=False)
                elif model_name == 'BiLSTM':
                    score, tag = model.predict(words_q,
                                               words_a,
                                               wordslen_q, wordslen_a,
                                               usecuda=self.usecuda,
                                               scoreonly=False)

                st = sorted(zip(sort_info, score, tag, data['tags']), key=lambda p: p[0])
                _, origin_score, origin_tag, real_tag = zip(*st)

                tag_arr.append(list(origin_tag))
                score_arr.append(list(origin_score))
                real_tag_arr.append(list(real_tag))

            for i in range(len(score_arr)):
                for j in range(len(score_arr[i])):
                    if int(tag_arr[i][j]) == 0:
                        score_arr[i][j] = 1 - score_arr[i][j]

            new_score_seq = []
            for m in range(len(words_q)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            cutted_score_seq = []
            for i in range(int(len(new_score_seq) / self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(new_score_seq[i * self.answer_count + j])
                cutted_score_seq.append(temp)

            for item in cutted_score_seq:

                def rankedList(rList):
                    rList = np.array(rList)
                    gain = 2 ** rList - 1
                    discounts = np.log2(np.arange(len(rList)) + 2)
                    return np.sum(gain / discounts)

                tp1 = np.transpose(
                    np.array(item)).tolist()
                dList = []
                for i in range(len(tp1)):
                    rL = sorted(tp1[i], reverse=True)
                    dList.append(rankedList(rL))

                t = np.mean(2 ** np.array(item) - 1, axis=1)

                rankedt = sorted(t.tolist(), reverse=True)
                d = rankedList(rankedt)

                obj = {}
                obj["q_id"] = pt
                obj["el"] = np.mean(np.array(dList)) - d

                if obj["el"] < 0:
                    print("elo error")
                    exit()

                _index = np.argmax(tp1, axis=1)

                obj["predict"] = stats.mode(_index)[0][0]
                obj["predict_confidence"] =  stats.mode(_index)[1][0]

                _delt_arr.append(obj)

                pt += 1

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True)

        cur_indices = set()
        sample_q_indices = set()
        i = 0

        while len(cur_indices) < num_questions * self.answer_count:
            sample_q_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            for k in range(self.answer_count):
                cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + k)
            i += 1

        if not returned:
            print("Active")
            self.train_index.update(cur_indices)
            print('time consuming： %d seconds:' % (time.time() - tm))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, sample_q_indices


    def get_DASL(self, dataset, model_path,
                   num_questions, batch_size=1000,
                   nsamp=100, model_name='',
                   returned = False,
                   noActive = False
                   ):

        model = torch.load(model_path)
        model.train(True)
        tm = time.time()

        new_dataset = [datapoint for j, datapoint in enumerate(dataset) if j not in list(self.train_index)]
        new_datapoints = [j for j in range(len(dataset)) if j not in list(self.train_index)]
        new_question_points = [new_datapoints[x * self.answer_count] for x in
                               range(int(len(new_datapoints) / self.answer_count))]

        data_batches = create_batches(new_dataset, batch_size=batch_size, order='no')

        pt = 0
        _delt_arr = []

        for data in data_batches:

            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda()
                words_a = Variable(torch.LongTensor(words_a)).cuda()
            else:
                words_q = Variable(torch.LongTensor(words_q))
                words_a = Variable(torch.LongTensor(words_a))

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            sort_info = data['sort_info']

            tag_arr = []
            score_arr = []
            real_tag_arr = []

            for itr in range(nsamp):

                if model_name == 'CNN':
                    score, tag = model.predict(words_q,
                                               words_a,
                                               usecuda=self.usecuda,
                                               scoreonly=False)
                elif model_name == 'BiLSTM':
                    score, tag = model.predict(words_q,
                                               words_a,
                                               wordslen_q, wordslen_a,
                                               usecuda=self.usecuda,
                                               scoreonly=False)

                st = sorted(zip(sort_info, score, tag, data['tags']), key=lambda p: p[0])
                _, origin_score, origin_tag, real_tag = zip(*st)

                tag_arr.append(list(origin_tag))
                score_arr.append(list(origin_score))
                real_tag_arr.append(list(real_tag))


            for i in range(len(score_arr)):
                for j in range(len(score_arr[i])):
                    if int(tag_arr[i][j]) == 0:
                        score_arr[i][j] = 1 - score_arr[i][j]

            new_score_seq = []
            for m in range(len(words_q)):
                tp = []
                for n in range(nsamp):
                    tp.append(score_arr[n][m])
                new_score_seq.append(tp)

            cutted_score_seq = []
            for i in range(int(len(new_score_seq) / self.answer_count)):
                temp = []
                for j in range(self.answer_count):
                    temp.append(new_score_seq[i * self.answer_count + j])
                cutted_score_seq.append(temp)

            for item in cutted_score_seq:

                def rankedList(rList):
                    rList = np.array(rList)
                    gain = 2 ** rList - 1
                    discounts = np.log2(np.arange(len(rList)) + 2)
                    return np.sum(gain / discounts)

                tp1 = np.transpose(
                    np.array(item)).tolist()
                dList = []
                for i in range(len(tp1)):
                    rL = sorted(tp1[i], reverse=True)
                    dList.append(rankedList(rL))

                t = np.mean(2 ** np.array(item) - 1, axis=1)

                rankedt = sorted(t.tolist(), reverse=True)
                d = rankedList(rankedt)

                obj = {}
                obj["q_id"] = pt
                obj["el"] = np.mean(np.array(dList)) - d

                if obj["el"] < 0:
                    print("elo error: acqusition_new: 927")
                    exit()

                _index = np.argmax(tp1, axis=1)

                obj["predict"] = stats.mode(_index)[0][0]
                obj["predict_confidence"] = stats.mode(_index)[1][0]

                _delt_arr.append(obj)

                pt += 1

        _delt_arr = sorted(_delt_arr, key=lambda o: o["el"], reverse=True)

        data_el = []
        for item in _delt_arr:
            data_el.append(item["el"])

        cur_indices = set()
        sample_q_indices = set()
        i = 0
        while len(cur_indices) < num_questions * self.answer_count:
            sample_q_indices.add(new_question_points[_delt_arr[i]["q_id"]])
            for k in range(self.answer_count):
                cur_indices.add(new_question_points[_delt_arr[i]["q_id"]] + k)
            i += 1

        self.pseudo_train_data = []
        i = 1
        while i <= len(_delt_arr) - num_questions:
            if _delt_arr[len(_delt_arr)-i]["predict_confidence"] == 100:
                for k in range(self.answer_count):
                     data = copy.deepcopy(dataset[new_question_points[_delt_arr[len(_delt_arr)-i]["q_id"]] + k])
                     data['tag'] = 1 if k == _delt_arr[len(_delt_arr) - i]["predict"] else 0
                     data['el'] = _delt_arr[len(_delt_arr)-i]["el"]
                     self.pseudo_train_data.append(data)

            i += 1

        if noActive:
            print("DASL：self-pace learning")
            return

        if not returned:
            self.train_index.update(cur_indices)
            print('time consuming： %d seconds:' % (time.time() - tm))
        else:
            sorted_cur_indices = list(cur_indices)
            sorted_cur_indices.sort()
            dataset_pool = []
            for m in range(len(sorted_cur_indices)):
                item = dataset[sorted_cur_indices[m]]
                item["index"] = sorted_cur_indices[m]
                dataset_pool.append(item)

            return dataset_pool, sample_q_indices


    def coreset_sample(self, data, num_questions, model_path='', model_name='', feature_type='query'):
        def greedy_k_center(labeled, unlabeled, amount):

            greedy_indices = []

            # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
            min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            for j in range(1, labeled.shape[0], 100):
                if j + 100 < labeled.shape[0]:
                    dist = distance_matrix(labeled[j:j + 100, :], unlabeled)
                else:
                    dist = distance_matrix(labeled[j:, :], unlabeled)
                min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
                min_dist = np.min(min_dist, axis=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))

            # iteratively insert the farthest index and recalculate the minimum distances:
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)
            for i in range(amount - 1):
                dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1, unlabeled.shape[1])), unlabeled)
                min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
                min_dist = np.min(min_dist, axis=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))
                farthest = np.argmax(min_dist)
                greedy_indices.append(farthest)

            return np.array(greedy_indices)

        sample_feature = self.getSimilarityMatrix(data, model_path, model_name, type=feature_type, feature_only=True)
        unlabel = [id for id in range(len(data)) if id not in self.train_index][::self.answer_count]
        unlabel = [id // self.answer_count for id in unlabel]
        labeled = sorted(list(self.train_index))[::self.answer_count]
        labeled = [id // self.answer_count for id in labeled]
        # print('In coreset_sample, labeled size: {}, unlabeled size:{}'.format(len(labeled), len(unlabel)))
        # print('In coreset_sample, sample_feature shape: {}'.format(sample_feature.shape))
        labeled_feature = sample_feature[labeled]
        unlabel_feature = sample_feature[unlabel]
        sel_indices = greedy_k_center(labeled_feature, unlabel_feature, num_questions)
        sel_indices = np.array(unlabel)[sel_indices].tolist()
        cur_indices = set([ind * self.answer_count + k for ind in sel_indices for k in range(self.answer_count)])
        self.train_index.update(cur_indices)


        ##############################
    def getSimilarityMatrix(self, dataset, model_path='', model_name='', batch_size=1000, type='query',
                            feature_only=False):
        '''
        :param dataset:
        :param model_path:
        :param model_name:
        :param batch_size:
        :param type:   1.query   2. q-a-concat   3.q-a-concat-mean    4.mean-var
        :param feature_only: Whether to return the feature or similarity matrix
        :return:
        '''
        model = torch.load(model_path)
        model.train(False)
        tm = time.time()

        data_batches = create_batches(dataset, batch_size=batch_size, order='no')

        temp_q = []
        temp_a = []
        for data in data_batches:

            words_q = data['words_q']
            words_a = data['words_a']

            if self.usecuda:
                words_q = Variable(torch.LongTensor(words_q)).cuda()
                words_a = Variable(torch.LongTensor(words_a)).cuda()
            else:
                words_q = Variable(torch.LongTensor(words_q))
                words_a = Variable(torch.LongTensor(words_a))

            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            if model_name == 'CNN':
                q_f, a_f = model.predict(words_q,
                                         words_a,
                                         usecuda=self.usecuda,
                                         scoreonly=False,
                                         encoder_only=True)
            elif model_name == 'BiLSTM':
                q_f, a_f = model.predict(words_q,
                                         words_a,
                                         wordslen_q, wordslen_a,
                                         usecuda=self.usecuda,
                                         scoreonly=False,
                                         encoder_only=True)
            temp_q.extend(list(q_f))
            temp_a.extend(list(a_f))

        q_features = temp_q[::5]
        q_features = np.stack(q_features, axis=0)
        a_features = np.stack(temp_a, axis=0)

        if type == 'query':
            sample_feature = q_features
        if type == 'q-a-concat':
            a_features = a_features.reshape(-1, 5 * a_features.shape[1])
            sample_feature = np.concatenate((q_features, a_features), axis=1)
        elif type == 'q-a-concat-mean':
            q_features = q_features.reshape(q_features.shape[0], 1, q_features.shape[1])
            a_features = a_features.reshape(-1, 5, a_features.shape[1])
            sample_feature = np.mean(np.concatenate((q_features, a_features), axis=1), axis=1)
            assert sample_feature.shape == (q_features.shape[0], q_features.shape[2])
        elif type == 'mean-var':
            a_shape = (a_features.shape[0] // 5, 5, a_features.shape[1])
            a_features = np.reshape(a_features, a_shape)
            mean_feature = np.mean(a_features, axis=1)
            var_feature = np.var(a_features, axis=1)
            sample_feature = np.concatenate((mean_feature, var_feature), axis=1)

        if feature_only:
            return sample_feature

        similarity = cosine_similarity(sample_feature) + 1
        return similarity


    #——————————————————————————————Invoking a sampling strategy to obtain data————————————————————————————————————————————
    def obtain_data(self, data, model_path=None, model_name=None, acquire=2,
                    method='random', sub_method=''):

        print("sampling method：" + sub_method)

        num_questions = acquire  # Number of questions to sample

        if model_path == "":
            print("First round of sampling")
            self.get_random(data,num_questions)
        else:
            if method == 'random':
                self.get_random(data, num_questions)
            elif method == 'dete':
                if sub_method == 'coreset':
                    self.coreset_sample(data, num_questions, model_path=model_path, model_name=model_name)
                else:
                    self.get_sampling(data, model_path,
                                      num_questions, model_name=model_name, quota=sub_method, deterministic=True)
            elif method == 'no-dete':  # Bayesian neural network based method
                if sub_method == 'BALD':
                    self.get_BALD(data, model_path, num_questions, model_name=model_name)
                elif sub_method == 'DASL':
                    self.get_DASL(data, model_path, num_questions, model_name=model_name)
                elif sub_method == 'DAL':
                    self.get_DAL(data, model_path, num_questions, model_name=model_name)
                else:
                    self.get_sampling(data, model_path, num_questions, model_name=model_name, quota=sub_method,
                                      deterministic=False)
            else:
                raise NotImplementedError()

        return 1


