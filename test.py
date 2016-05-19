# coding: utf-8
"""
test lda model using sogo mini news: Sample
"""
import os
from collections import defaultdict
from gensim import corpora, models
import LdaModel
import utils


def get_data():
    """
    获取语料库
    """
    texts = []
    subfolds = os.listdir("../Sample")
    for subfold in subfolds:
        subdir = "../Sample/{}".format(subfold)
        if os.path.isdir(subdir):
            files = os.listdir(subdir)
            for file in files:
                text = open("{}/{}".format(subdir, file)).read()
                text = utils.preprocess(text)
                text = utils.getWordlist(text)
                texts.append(text)

    # 去除只出现一次的词
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]

    return texts


def test_original_gibbs(texts):
    """
    测试自己写的lda模型
    """
    K = 20  # topic个数
    alpha = 50.0 / K  # 超参数alpha
    beta = 0.01  # 超参数beta
    iterations = 1000  # 迭代次数
    saveStep = 100  # 保存的间隔
    beginSaveIters = 500  # 第一开始保存的迭代次数

    ldamodel = LdaModel.LdaModel(texts, alpha, beta, iterations, K, saveStep, beginSaveIters)
    ldamodel.trainModel()


def test_gensim(texts):
    """
    测试gensim模块中的lda模型
    """
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=100)
