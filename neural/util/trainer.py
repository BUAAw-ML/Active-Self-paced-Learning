from __future__ import print_function
from torch.autograd import Variable
import time
from .evaluator import Evaluator
from .utils import *
import sys
import os
import numpy as np

import torch
import torch.nn as nn


class Trainer(object):

    def __init__(self, model, result_path, model_name, tag_to_id,
                 eval_every=1, usecuda=True, answer_count = 5):
        self.model = model
        self.eval_every = eval_every
        self.model_name = os.path.join(result_path, model_name)
        self._model_name = model_name
        self.usecuda = usecuda
        self.tagset_size = len(tag_to_id)

        self.lossfunc = nn.CrossEntropyLoss()

        self.evaluator = Evaluator(result_path, model_name, answer_count = answer_count).evaluate_rank


    ##############################################################################
    ################# training only by supervised learning        ################
    def train_supervisedLearning(self, num_epochs, train_data, val_data, test_data, learning_rate, checkpoint_folder='.',
                     plot_every=20, batch_size=50):

        losses = []
        lossD = 0.0
        best_test_mrr = -1.0
        count = 0
        batch_count = 0

        self.model.train(True)
        print('********Training Start*******')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(1, num_epochs + 1):
            t = time.time()

            train_batches = create_batches(train_data['labeled_train_data'],
                                           batch_size=batch_size, order='random')

            for i, index in enumerate(np.random.permutation(len(train_batches))):

                data = train_batches[index]
                self.model.zero_grad()

                words_q = data['words_q']
                words_a = data['words_a']
                tags = data['tags']

                words_q = Variable(torch.LongTensor(words_q)).cuda()
                words_a = Variable(torch.LongTensor(words_a)).cuda()
                tags = Variable(torch.LongTensor(tags)).cuda()

                wordslen_q = data['wordslen_q']
                wordslen_a = data['wordslen_a']

                if self._model_name in ['BiLSTM']:
                    output = self.model(words_q, words_a, tags, wordslen_q, wordslen_a, usecuda=self.usecuda)
                elif self._model_name in ['CNN']:
                    output = self.model(words_q, words_a, tags, usecuda=self.usecuda)

                loss = self.lossfunc(output, tags)
                lossD += loss.item() / len(wordslen_q)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                count += 1
                batch_count += len(wordslen_q)

                if count % plot_every == 0:
                    lossD /= plot_every
                    #print(batch_count, ': ', lossD)
                    if losses == []:
                        losses.append(lossD)
                    losses.append(lossD)
                    lossD = 0.0

            ####################################### Validation ###########################################
            if epoch % self.eval_every == 0:

                best_test_mrr, new_test_mrr, save = self.evaluator(self.model, val_data, best_test_mrr, model_name = self._model_name)

                print('*'*50)
                print("Validation：best_mrr：%f new_mrr:%f " % (best_test_mrr, new_test_mrr))

                if save:
                    print('Saving Best Weights')
                    torch.save(self.model, os.path.join(self.model_name, checkpoint_folder, 'modelweights'))

                sys.stdout.flush()

            print('Epoch %d Complete: Time Taken %d' % (epoch, time.time() - t))

        _, test_mrr, _ = self.evaluator(torch.load(os.path.join(self.model_name, checkpoint_folder, 'modelweights')),
                                        test_data,
                                        model_name=self._model_name)

        return test_mrr



    ##############################################################################
    ################# training  by self-Paced learning        ####################
    def train_selfPacedLearning(self, noActiveTrain, num_epochs, train_data, val_data, test_data, mu, learning_rate, checkpoint_folder='.',
                    plot_every=20, batch_size=50):

        losses = []
        lossD = 0.0
        best_test_mrr = -1.0
        count = 0
        batch_count = 0

        self.model.train(True)
        print('********Training Start*******')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        PLdata = train_data['pseudo_train_data']

        baseValue_index = int(0.1 * len(PLdata))
        ld = PLdata[baseValue_index]['el']

        iter = 0
        best_iter = iter

        while ld <= PLdata[-1]['el']:

            pseudoData_end = 0
            while PLdata[pseudoData_end]['el'] <= ld:
                pseudoData_end += 1

            print('*' * 63)
            print("current iter: {}, pseudoData_end: {}".format(iter, pseudoData_end))

            for epoch in range(1, num_epochs // 2 + 1):
                t = time.time()

                ####################### train with pseudo-labeled data ###################
                train_batches = create_batches(PLdata[:pseudoData_end],
                                               batch_size=batch_size, order='random')

                for i, index in enumerate(np.random.permutation(len(train_batches))):

                    data = train_batches[index]
                    self.model.zero_grad()

                    words_q = data['words_q']
                    words_a = data['words_a']
                    tags = data['tags']

                    words_q = Variable(torch.LongTensor(words_q)).cuda()
                    words_a = Variable(torch.LongTensor(words_a)).cuda()
                    tags = Variable(torch.LongTensor(tags)).cuda()

                    wordslen_q = data['wordslen_q']
                    wordslen_a = data['wordslen_a']

                    if self._model_name in ['BiLSTM']:
                        output = self.model(words_q, words_a, tags, wordslen_q, wordslen_a, usecuda=self.usecuda)
                    elif self._model_name in ['CNN']:
                        output = self.model(words_q, words_a, tags, usecuda=self.usecuda)

                    loss = self.lossfunc(output, tags)
                    lossD += loss.item() / len(wordslen_q)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    count += 1
                    batch_count += len(wordslen_q)

                    if count % plot_every == 0:
                        lossD /= plot_every
                        #print(batch_count, ': ', lossD)
                        if losses == []:
                            losses.append(lossD)
                        losses.append(lossD)
                        lossD = 0.0

                ####################### train with labeled data ###################
                train_batches = create_batches(train_data['labeled_train_data'],
                                               batch_size=batch_size, order='random')

                for i, index in enumerate(np.random.permutation(len(train_batches))):  # [(0,2),(1,3),(2,1),(3,0)]


                    data = train_batches[index]
                    self.model.zero_grad()

                    words_q = data['words_q']  # [[1,2,3,4,9],[4,3,5,0,0]]
                    words_a = data['words_a']  # [[1,2,3,4,9],[4,3,5,0,0]]
                    tags = data['tags']

                    words_q = Variable(torch.LongTensor(words_q)).cuda()
                    words_a = Variable(torch.LongTensor(words_a)).cuda()
                    tags = Variable(torch.LongTensor(tags)).cuda()

                    wordslen_q = data['wordslen_q']
                    wordslen_a = data['wordslen_a']

                    if self._model_name in ['BiLSTM']:
                        output = self.model(words_q, words_a, tags, wordslen_q, wordslen_a, usecuda=self.usecuda)
                    elif self._model_name in ['CNN']:
                        output = self.model(words_q, words_a, tags, usecuda=self.usecuda)

                    loss = self.lossfunc(output, tags)
                    lossD += loss.item() / len(wordslen_q)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                    optimizer.step()

                    count += 1
                    batch_count += len(wordslen_q)

                    if count % plot_every == 0:
                        lossD /= plot_every
                        #print(batch_count, ': ', lossD)
                        if losses == []:
                            losses.append(lossD)
                        losses.append(lossD)
                        lossD = 0.0

                ####################################### Validation ###########################################
                if epoch % self.eval_every == 0:

                    best_test_mrr, new_test_mrr, save = self.evaluator(self.model, val_data, best_test_mrr,
                                                                       model_name=self._model_name)

                    print('*' * 50)
                    print("Validation：best_mrr：%f new_mrr:%f " % (best_test_mrr, new_test_mrr))

                    if save:
                        print('Saving Best Weights')
                        torch.save(self.model, os.path.join(self.model_name, checkpoint_folder, 'modelweights'))
                        best_iter = iter

                    sys.stdout.flush()

                print('Epoch %d Complete: Time Taken %d' % (epoch, time.time() - t))

            if iter != best_iter:
                break

            noActiveTrain["acquisition_function"].get_DASL(noActiveTrain["train_data"],
                                                           noActiveTrain["model_path"],
                                                           noActiveTrain["acquire"],
                                                           model_name=noActiveTrain["model_name"],
                                                           noActive=True)

            PLdata = noActiveTrain["acquisition_function"].pseudo_train_data

            if len(PLdata) == 0:
                break

            baseValue_index = int(0.1 * len(PLdata))
            ld = PLdata[baseValue_index]['el'] * mu

            mu *= mu
            iter += 1

        print("the best iter: {0}".format(best_iter))

        _, test_mrr, _ = self.evaluator(torch.load(os.path.join(self.model_name, checkpoint_folder, 'modelweights')),
                                        test_data,
                                        model_name=self._model_name)

        return test_mrr
