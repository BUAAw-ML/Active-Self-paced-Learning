# coding=utf-8
from __future__ import print_function
import numpy as np
import torch

import os
import pickle as pkl
import argparse

from neural.util import Trainer, Loader
from neural.models import BiLSTM
from neural.models import CNN


from active_learning.acquisition import Acquisition
from active_learning.chartTool import *

torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')

    parser.add_argument('--answer_count', type=int, default=5, help='the amount of answer for each quesiton')
    parser.add_argument('--num_epochs', type=int, default=12, help='training epoch')
    parser.add_argument('--use_pretrained_word_embedding', type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=50, help='')
    parser.add_argument('--with_sim_feature', type=bool, default=True, help='whether use sim_feature in deep model')
    parser.add_argument('--double_embedding', type=bool, default=False, help='whether use two kinds of word embedding')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='')
    parser.add_argument('--pretrained_word_embedding', default="data/pretrained-word-embedding/glove.6B.300d.txt", help='')
    parser.add_argument('--dropout', type=float, default=0.5, help='')
    parser.add_argument('--dropout2', type=float, default=0.5, help='')
    parser.add_argument('--word_hidden_dim', type=int, default=75, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--target_size', type=int, default=2, help='')
    parser.add_argument('--word_out_channels', type=int, default=200, help='')
    parser.add_argument('--result_path', default="result/YahooCQA/",help='')
    parser.add_argument('--mu', type=int, default=1.2, help='')

    args = parser.parse_args()
    return args


####################################################################################################
#############################              active learning               ###########################
def main(args):

    task_seq = [

        # The config for a task:
        # acquire_method(sub_acquire_method): random(""), no-dete("DASL","DAL","BALD"), dete("coreset","entropy",...)
        {
            "model_name": "BiLSTM",
            "group_name": "[2.18-?]BiLSTM+FD+MRR+200+200",
            "max_performance": 0.80,
            "data_path": "data/YahooCQA/data-FD/",
            "acquire_method": "no-dete",
            "sub_acquire_method": "DASL",
            "num_acquisitions_round": 37,
            "init_question_num": 40,
            "acquire_question_num_per_round": 40,
            "warm_start_random_seed": 16,
            "sample_method": "No-Deterministic+DASL2+seed16",
        },
    ]

    allMethods_results = []   #Record the performance results of each method during active learning

    for config in task_seq:

        print("-----------------------{}-{}-----------------------".format(config["group_name"], config["sample_method"]))

        ####################################### initial setting ###########################################
        data_path = config["data_path"] if "data_path" in config else "data/YahooCQA/data-FD/"
        model_name = config["model_name"] if "model_name" in config else 'CNN'
        num_acquisitions_round = config["num_acquisitions_round"]
        acquire_method = config["acquire_method"]
        sub_acquire_method = config["sub_acquire_method"]
        init_question_num = config["init_question_num"] if "init_question_num" in config else 160 # number of initial training samples
        acquire_question_num_per_round = config["acquire_question_num_per_round"] if "acquire_question_num_per_round" in config else 20 #Number of samples collected per round
        warm_start_random_seed = config["warm_start_random_seed"]  # the random seed for selecting the initial training set
        sample_method = config["sample_method"]


        loader = Loader()

        print('model:', model_name)
        print('dataset:', data_path)
        print('acquisition method:', acquire_method, "+", sub_acquire_method)

        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)

        if not os.path.exists(os.path.join(args.result_path, model_name)):
            os.makedirs(os.path.join(args.result_path, model_name))

        if not os.path.exists(os.path.join(args.result_path, model_name, 'active_checkpoint', acquire_method)):
            os.makedirs(os.path.join(args.result_path, model_name, 'active_checkpoint', acquire_method))

        #### If the data is not compiled, compile; otherwise load directly
        if (os.path.exists(os.path.join(data_path, 'mappings.pkl')) and
            os.path.exists(os.path.join(data_path, 'train.pkl')) and
            os.path.exists(os.path.join(data_path, 'val.pkl')) and
            os.path.exists(os.path.join(data_path, 'test.pkl'))
        ):
            mappings = pkl.load(open(os.path.join(data_path, 'mappings.pkl'), 'rb'))
            train_data = pkl.load(open(os.path.join(data_path, 'train.pkl'), 'rb'))
            val_data = pkl.load(open(os.path.join(data_path, 'val.pkl'), 'rb'))
            test_data = pkl.load(open(os.path.join(data_path, 'test.pkl'), 'rb'))
        else:
            train_data, val_data, test_data, mappings = loader.load_yahoo(data_path,
                                                                args.pretrained_word_embedding,
                                                                args.word_embedding_dim,
                                                                args.answer_count)
            pkl.dump(train_data, open(os.path.join(data_path, 'train.pkl'), 'wb'))
            pkl.dump(val_data, open(os.path.join(data_path, 'val.pkl'), 'wb'))
            pkl.dump(test_data, open(os.path.join(data_path, 'test.pkl'), 'wb'))
            pkl.dump(mappings, open(os.path.join(data_path, 'mappings.pkl'), 'wb'))


        #word embedding
        word_to_id = mappings['word_to_id']
        tag_to_id = mappings['tag_to_id']
        word_embeds = mappings['word_embeds'] if args.use_pretrained_word_embedding else None

        word_vocab_size = len(word_to_id)

        total_sentences = len(train_data)  # Total number of training samples (number of question answer pair)

        print('After training data is loaded, the total amount of training data： %d' % total_sentences)

        acquisition_function = Acquisition(train_data,
                                            seed=warm_start_random_seed,
                                            answer_count = args.answer_count
                                            )

        method_result = []  # Record the performance results of each method during active learning
        ####################################### acquire data and retrain ###########################################
        for i in range(num_acquisitions_round):

            print("current round：{}".format(i))

            #-------------------acquisition---------------------
            if i == 0:#first round
                acq = init_question_num
                a_m = "random"
                m_p = ""
                acquisition_function.obtain_data(train_data, model_path="", acquire=init_question_num, method="random")
            else:
                acq = acquire_question_num_per_round
                a_m = acquire_method
                m_p = os.path.join(checkpoint_path, 'modelweights')
                acquisition_function.obtain_data(model_path = m_p,
                                                model_name = model_name,
                                                data = train_data,
                                                acquire = acq,
                                                method = a_m,
                                                sub_method = sub_acquire_method
                                                )

            # -------------------prepare training data---------------------
            '''
            train_data的每个元素格式：
            {
                'str_words_q': str_words_q,  # question word segmentation
                'str_words_a': str_words_a,  # answer word segmentation
                'words_q': words_q,  # question word id
                'words_a': words_a,  # answer word id
                'tag': tag,  # sample tag id
            }
            '''

            new_train_index = (acquisition_function.train_index).copy()
            sorted_train_index = list(new_train_index)
            sorted_train_index.sort()
            labeled_train_data = [train_data[i] for i in sorted_train_index]

            active_train_data = dict()
            active_train_data['labeled_train_data'] = labeled_train_data
            active_train_data['pseudo_train_data'] = acquisition_function.pseudo_train_data


            print("Labeled training samples: {}".format(len(acquisition_function.train_index)))
            print("Unlabeled sample remaining: {}".format(len(train_data) - len(acquisition_function.train_index)))

            # -------------------------------------train--------------------------------------
            checkpoint_folder = os.path.join('active_checkpoint', acquire_method, "fixed")
            checkpoint_path = os.path.join(args.result_path, model_name, checkpoint_folder)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            print('.............Recreate the model...................')
            if model_name == 'BiLSTM':
                    model = BiLSTM(word_vocab_size,
                                   args.word_embedding_dim,
                                   args.word_hidden_dim,
                                   args.target_size,
                                   pretrained=word_embeds,
                                   with_sim_features=args.with_sim_feature,
                                   double_embedding=args.double_embedding
                                   )
            if model_name == 'CNN':
                    model = CNN(word_vocab_size,
                                args.word_embedding_dim,
                                args.word_out_channels,
                                args.target_size,
                                pretrained=word_embeds,
                                double_embedding=args.double_embedding
                                )

            model.cuda()

            trainer = Trainer(model,
                                args.result_path,
                                model_name,
                                tag_to_id,
                                answer_count=args.answer_count)

            if active_train_data['pseudo_train_data']:

                noActiveTrain = {"acquisition_function": acquisition_function,
                                 "model_path": m_p,
                                 "model_name": model_name,
                                 "train_data": train_data,
                                 "acquire": acq,
                                 "method": a_m,
                                 "sub_method": sub_acquire_method}

                test_performance = trainer.train_selfPacedLearning(noActiveTrain,args.num_epochs,
                                                                    active_train_data,
                                                                    val_data,
                                                                    test_data,
                                                                    args.mu,
                                                                    args.learning_rate,
                                                                    checkpoint_folder=checkpoint_folder,
                                                                    batch_size=args.batch_size
                                                                    )
            else:
                test_performance = trainer.train_supervisedLearning(args.num_epochs,
                                                                    active_train_data,
                                                                    val_data,
                                                                    test_data,
                                                                    args.learning_rate,
                                                                    checkpoint_folder=checkpoint_folder,
                                                                    batch_size=args.batch_size
                                                                    )

            print('*' * 50)
            print("Test performance: {}".format(test_performance))
            print('-' * 80)


            #--------------------------Send data for a visual web page------------------------------
            max_performance = config["max_performance"] if "max_performance" in config else 0

            if "group_name" in config:
                updateLineChart(str(test_performance), sample_method, gp_name = config["group_name"], max = max_performance)
            else:
                updateLineChart(str(test_performance), sample_method, max = max_performance)

        #     method_result.append(test_performance)
        #
        # print("acquire_method: {}，sub_acquire_method: {}, warm_start_random_seed{}"
        #       .format(acquire_method, sub_acquire_method, warm_start_random_seed))
        # print(method_result)
        # allMethods_results.append(method_result)




if __name__ == "__main__":
    args = parse_args()

    main(args)
