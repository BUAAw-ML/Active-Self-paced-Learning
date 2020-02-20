import os
import codecs
import torch
from .utils import *
import torch
from torch.autograd import Variable

# np.set_printoptions(threshold=np.inf)


class Evaluator(object):
    def __init__(self, result_path, model_name, usecuda=True, answer_count = 5):
        self.result_path = result_path
        self.model_name = model_name
        self.usecuda = usecuda
        self.answer_count = answer_count


    def evaluate_rank(self, model, dataset, best_mrr = 0.0, model_name='CNN'):

        model.train(False)

        RR = []

        save = False

        #get RR value for a prediction
        def getRR(predicted, ground_truth):
            zipped = zip(predicted, ground_truth)
            temp = sorted(zipped, key=lambda p: p[0], reverse=True)
            _, ground_truth_ids = zip(*temp)
            for i in range(len(ground_truth_ids)):
                if ground_truth_ids[i] == 1:
                    return (1.0 / float(i + 1))

        data_batches = create_batches(dataset, batch_size = 1000, order='no')

        for data in data_batches:

            predicted_scores = []
            predicted_ids = []
            ground_truth_ids = []

            words_q = data['words_q']  # [[1,2,3,4,9],[4,3,5,0,0]]
            words_a = data['words_a']  # [[1,2,3,4,9],[4,3,5,0,0]]

            words_q = Variable(torch.LongTensor(words_q)).cuda()
            words_a = Variable(torch.LongTensor(words_a)).cuda()

            # 句子长度
            wordslen_q = data['wordslen_q']
            wordslen_a = data['wordslen_a']

            if model_name == 'BiLSTM':
                score, out = model.predict(words_q, words_a,
                                           wordslen_q, wordslen_a,
                                           usecuda=self.usecuda)
            elif model_name == 'CNN':
                score, out = model.predict(words_q, words_a,
                                       usecuda=self.usecuda)


            ground_truth_ids.extend(data['tags'])
            predicted_ids.extend(out)
            predicted_scores.extend(score)


            for i in range(len(predicted_scores)):
                if int(predicted_ids[i]) == 0:
                    predicted_scores[i] = 1 - predicted_scores[i]


            for i in range(int(len(data["words_q"]) / self.answer_count)):
                temp_1 = []
                temp_2 = []
                for j in range(self.answer_count):
                    temp_1.append(predicted_scores[i * self.answer_count + j])
                    temp_2.append(ground_truth_ids[i * self.answer_count + j])
                RR.append(getRR(temp_1, temp_2))

        # MRR
        new_mrr = round(sum(RR) / len(RR), 4)

        if new_mrr > best_mrr:
            best_mrr = new_mrr
            save = True

        model.train(True)

        return best_mrr, new_mrr, save