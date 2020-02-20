import torch
import torch.nn as nn

from torch.autograd import Variable

from .baseRNN import baseRNN

class EncoderRNN(baseRNN):

    def __init__(self, vocab_size, embedding_size ,hidden_size= 200, input_dropout_p=0,
                 output_dropout_p=0, n_layers=1, bidirectional=True, rnn_cell='lstm'):
        
        super(EncoderRNN, self).__init__(vocab_size, hidden_size, input_dropout_p, 
                                             output_dropout_p, n_layers, rnn_cell)


        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 bidirectional=bidirectional, dropout=output_dropout_p,
                                 batch_first=True)



    def forward(self, words, input_lengths):
        
        batch_size = words.size()[0]#每batch多少样本

        input_lengths = torch.LongTensor(input_lengths)

        # sorted_indices是针对未排序之前的序列，从大到小排序获得排好的序列
        _, sorted_indices = torch.sort(input_lengths, dim=0, descending=True)
        # unsort_indices是针对排好序的序列，找回未排序之前的序列
        _, unsort_indices = torch.sort(sorted_indices, dim=0)

        sorted_lengths = list(input_lengths[sorted_indices])
        sorted_indices = Variable(sorted_indices).cuda()
        unsort_indices = Variable(unsort_indices).cuda()

        # 对words按有效长度从大到小排序
        words = words.index_select(dim = 0, index = sorted_indices)

        embedded = words
        embedded = self.input_dropout(embedded)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths, batch_first = True)
        _, output = self.rnn(embedded)

        '''
        output[0]:  (num_layers * num_directions, batch, hidden_size): 
                    tensor containing the hidden state for t = seq_len.
        最后一个时间步所有隐藏层节点的状态组合成一个向量作为Encoder编码的中间向量（语义编码），共batch_size个
        每个语义编码的长度是：隐藏层数目*LSTM方向数（单向、双向）*隐藏层节点个数
        [
            句子1的语义编码，
            句子2的语义编码，
            ...
            句子n的语义编码
        ]
        '''
        output = output[0].transpose(0,1).contiguous().view(batch_size, -1)#-1代表会自动推算该参数的大小

        # 恢复原始顺序
        output = output.index_select(0, unsort_indices)

        return output