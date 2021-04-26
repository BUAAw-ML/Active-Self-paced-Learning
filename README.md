# DASL

## Abstract
Answer Selection is an important subtask of Question Answering tasks. For this learning-to-rank problem, deep learning
methods have outperformed traditional methods. To train a highquality deep answer selection model, it often requires large amounts of labeled data, which is a costly and noise-prone process. Active learning and semi-supervised learning are usually applied in the modelling training procedure to achieve optimal accuracy with fewer labeled training samples. However, traditional active learning methods rely on good uncertainty estimates that are hard to obtain with standard neural networks. And the performance of semi-supervised learning methods are always affected adversely by the quality of the pseudo-labeled data. In this work, we propose a new framework integrating active learning and self-paced learning in training deep answer selection models. This framework proposes an uncertainty quantification method based on Bayesian neural network, which can guide active learning and self-paced learning in the same iterative process of model training. 

## Publication
This is the codebase for our 2020 ECAI paper:
[Combination of Active Learning and Self-Paced Learning for Deep Answer Selection with Bayesian Neural Network](http://ecai2020.eu/papers/449_paper.pdf).

```
@incollection{wang2020combination,
  title={Combination of Active Learning and Self-Paced Learning for Deep Answer Selection with Bayesian Neural Network},
  author={Wang, Qunbo and Wu, Wenjun and Qi, Yuxing and Xin, Zhimin},
  booktitle={ECAI 2020},
  pages={1587--1594},
  year={2020},
  publisher={IOS Press}
}
```

## Requirements: 
- Python3;
- Pytorch;
- numpy;
- sklearn;

In addition, anyone who want to run these codes should download the word embedding 'glove.6B.300d.txt' from https://nlp.stanford.edu/projects/glove/. The file should be placed at './datasets/word_embedding/glove.6B.300d.txt'.

## Datasets:
We upload a subset of the dataset YahooCQA to run these codes. You can download other datasets of Community-based Question Answering (CQA) and place it at './datasets'.

## Train and Test
python run_active_task_seq.py

- The options of the active learning method can be set in 'run_active_task_seq.py'. 