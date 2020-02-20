import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

import neural
from neural.util import Initializer
from neural.util import Loader
from neural.modules import EncoderCNN
from neural.modules import EncoderCNN_Pair

class CNN(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_out_channels, output_size, 
                 dropout_p = 0.5, pretrained=None, double_embedding = False):
        
        super(CNN, self).__init__()
        
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_out_channels = word_out_channels
        
        self.initializer = Initializer()
        self.loader = Loader()

        self.embedding = nn.Embedding(word_vocab_size, word_embedding_dim)

        if pretrained is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))

        #Q_CNN
        self.question_encoder = EncoderCNN(word_vocab_size, word_embedding_dim, word_out_channels)

        #A_CNN
        self.answer_encoder = EncoderCNN(word_vocab_size, word_embedding_dim, word_out_channels)

        #相似度特征
        self.interaction = nn.Parameter(torch.FloatTensor(word_out_channels, word_out_channels).uniform_(0, .1))

        self.dropout = nn.Dropout(p=dropout_p)
        
        hidden_size = word_out_channels * 2 + 1
        self.linear = nn.Linear(hidden_size, output_size)
        
        #self.lossfunc = nn.CrossEntropyLoss()
        
    def forward(self, questions, answers, tags, usecuda=True):

        # 2019-4-6
        questions_embedded = self.embedding(questions)
        answers_embedded = self.embedding(answers)

        question_features = self.question_encoder(questions_embedded)
        answer_features = self.answer_encoder(answers_embedded)

        i_question_features = torch.matmul(question_features, self.interaction)
        i_feature = torch.sum(i_question_features * answer_features, dim=1, keepdim=True)

        # 问题向量和答案向量拼接, join layer
        join_features = torch.cat((question_features, i_feature, answer_features), dim=1);

        join_features = self.dropout(join_features)
        output = self.linear(join_features)
        #loss = self.lossfunc(output, tags)
        
        return output#loss


    def predict(self, questions, answers, scoreonly=False, usecuda=True, encoder_only = False):

        questions_embedded = self.embedding(questions)
        answers_embedded = self.embedding(answers)

        # 2019-4-6
        question_features = self.question_encoder(questions_embedded)
        answer_features = self.answer_encoder(answers_embedded)

        #2019-5-11
        if encoder_only:
            return question_features.data.cpu().numpy(), answer_features.data.cpu().numpy(),

        i_question_features = torch.matmul(question_features, self.interaction)
        i_feature = torch.sum(i_question_features * answer_features, dim=1, keepdim=True)

        # 问题向量和答案向量拼接, join layer
        join_features = torch.cat((question_features, i_feature, answer_features), dim=1);

        join_features = self.dropout(join_features)
        output = self.linear(join_features)

        scores = torch.max(F.softmax(output, dim =1), dim=1)[0].data.cpu().numpy()
        if scoreonly:
            return scores

        prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
        return scores, prediction

    def predict_score(self, questions, answers):
        questions_embedded = self.embedding(questions)
        answers_embedded = self.embedding(answers)

        # 2019-4-6
        question_features = self.question_encoder(questions_embedded)
        answer_features = self.answer_encoder(answers_embedded)
        i_question_features = torch.matmul(question_features, self.interaction)
        i_feature = torch.sum(i_question_features * answer_features, dim=1, keepdim=True)

        # 问题向量和答案向量拼接, join layer
        join_features = torch.cat((question_features, i_feature, answer_features), dim=1);

        join_features = self.dropout(join_features)
        output = self.linear(join_features)
        return F.softmax(output, dim=1)