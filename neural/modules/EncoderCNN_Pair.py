import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN_Pair(nn.Module):

    def __init__(self, vocab_size, embedding_size, out_channels=100, dropout_p=0.5):
        super(EncoderCNN_Pair, self).__init__()

        self.out_channels = out_channels
        print('dropout_p:', dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        self.cnn1 = nn.Conv2d(1, out_channels, kernel_size=(2, embedding_size))
        self.cnn2 = nn.Conv2d(1, out_channels, kernel_size=(3, embedding_size))

    def forward(self, words, input_lengths=None):

        seq_len = words.size(1)
        feature = words.unsqueeze(1)
        feature1 = F.relu(self.cnn1(feature))
        feature2 = F.relu(self.cnn2(feature))
        feature1 = F.max_pool2d(feature1, kernel_size=(seq_len - 1, 1)).squeeze()
        feature2 = F.max_pool2d(feature2, kernel_size=(seq_len - 2, 1)).squeeze()
        feature1 = self.dropout(feature1)
        feature2 = self.dropout(feature2)
        return torch.cat((feature1, feature2), dim=1)
