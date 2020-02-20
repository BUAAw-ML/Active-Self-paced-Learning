import torch
import copy
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

import neural
from neural.util import Initializer
from neural.util import Loader
from neural.modules import EncoderRNN

class BiLSTM(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_hidden_dim, output_size, 
                 pretrained=None,
                 n_layers = 1,
                 bidirectional = True,
                 dropout_p = 0.5,
                 with_sim_features = True,
                 double_embedding = False):

        super(BiLSTM, self).__init__()

        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        
        self.initializer = Initializer()
        self.loader = Loader()

        self.with_sim_features = with_sim_features

        self.embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))

        #Q_LSTM
        self.question_encoder = EncoderRNN(word_vocab_size, word_embedding_dim, word_hidden_dim,
                                       n_layers = n_layers, bidirectional = bidirectional)
        #A_LSTM
        self.answer_encoder = EncoderRNN(word_vocab_size, word_embedding_dim, word_hidden_dim,
                                       n_layers = n_layers, bidirectional = bidirectional)

        hidden_size = 2 * (2 * n_layers * word_hidden_dim if bidirectional else n_layers * word_hidden_dim)

        if self.with_sim_features:
            word_out_dim = 2*n_layers*word_hidden_dim if bidirectional else n_layers*word_hidden_dim
            self.interaction = nn.Parameter(torch.FloatTensor(word_out_dim, word_out_dim).uniform_(0, .1))
            hidden_size += 1

        self.dropout = nn.Dropout(p=dropout_p)

        self.linear = nn.Linear(hidden_size, output_size)

        
    def forward(self, questions, answers, tags, wordslen_q, wordslen_a,  usecuda=True):

        questions_embedded = self.embedding(questions)
        answers_embedded = self.embedding(answers)

        question_features = self.question_encoder(questions_embedded, wordslen_q)
        answer_features = self.answer_encoder(answers_embedded, wordslen_a)

        if self.with_sim_features:
            i_question_features = torch.matmul(question_features, self.interaction)
            i_feature = torch.sum(i_question_features * answer_features, dim=1, keepdim=True)

            join_features = torch.cat((question_features, i_feature, answer_features), dim=1);
        else:
            join_features = torch.cat((question_features, answer_features), dim=1);

        join_features = self.dropout(join_features)


        output = self.linear(join_features)
        
        return output
    
    def predict(self, questions, answers, wordslen_q, wordslen_a,  scoreonly=False, usecuda=True, encoder_only = False):

        questions_embedded = self.embedding(questions)
        answers_embedded = self.embedding(answers)

        question_features = self.question_encoder(questions_embedded, wordslen_q)
        answer_features = self.answer_encoder(answers_embedded, wordslen_a)

        if encoder_only:
            return question_features.data.cpu().numpy(), answer_features.data.cpu().numpy(),

        if self.with_sim_features:
            i_question_features = torch.matmul(question_features, self.interaction)
            i_feature = torch.sum(i_question_features * answer_features, dim=1, keepdim=True)

            join_features = torch.cat((question_features, i_feature, answer_features), dim=1);
        else:
            join_features = torch.cat((question_features, answer_features), dim=1);

        join_features = self.dropout(join_features)
        output = self.linear(join_features)


        #属于该类别的概率
        scores = torch.max(F.softmax(output, dim =1), dim=1)[0].data.cpu().numpy()
        if scoreonly:
            return scores
        
        prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
        return scores, prediction
