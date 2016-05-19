# coding: utf-8
"""
传统吉布斯采样实现lda
"""

import random
import pickle
from gensim import corpora


class LdaModel:
    def __init__(self, texts, alpha, beta, iterations, K, saveStep, beginSaveIters):
        """
        :param texts: 分词后的语料库
        :param alpha: 文档中topic的dir分布的超参数
        :param beta: topic中word的dir分布的超参数
        :param iterations: 迭代次数
        :param K: topic的个数
        :param saveStep: 保存间隔
        :param beginSaveIters: 开始保存的位置
        """
        print "init..."
        self.dic = corpora.Dictionary(texts)
        self.doc = [[self.dic.token2id[w] for w in d] for d in texts]  # word index array
        self.V = len(self.dic.keys())  # vocabulary size
        self.K = K  # topic number
        self.M = len(texts)  # document number
        self.z = {}  # topic assignment for each word
        self.alpha = alpha
        self.beta = beta
        self.nmk = {m: {k: 0 for k in range(self.K)} for m in
                    range(self.M)}  # given document m, count times of topic k  M*K
        self.nkt = {k: {t: 0 for t in range(self.V)} for k in
                    range(self.K)}  # given topic k, count times of word t  k*V
        self.nmkSum = {m: 0 for m in range(self.M)}  # Sum for each row in nmk
        self.nktSum = {k: 0 for k in range(self.K)}  # Sum for each row in nkt
        self.phi = {k: {t: 0 for t in range(self.V)} for k in
                    range(self.K)}  # Parameters for topic-word distribution K*V
        self.theta = {m: {k: 0 for k in range(self.K)} for m in
                      range(self.M)}  # Parameters for doc-topic distribution M*K
        self.iterations = iterations  # Times of iterations
        self.saveStep = saveStep  # The number of iterations between two saving
        self.beginSaveIters = beginSaveIters  # Begin save model at this iteration

        # initialize topic lable z for each word
        for m in range(self.M):
            self.z[m] = {}
            N = len(self.doc[m])
            for n in range(N):
                initTopic = random.randint(0, K - 1)  # 随机选取一个topic id
                self.z[m][n] = initTopic
                self.nmk[m][initTopic] += 1
                self.nkt[initTopic][self.doc[m][n]] += 1
                self.nktSum[initTopic] += 1
            self.nmkSum[m] = N

    def trainModel(self):
        if self.iterations < self.saveStep + self.beginSaveIters:
            print "Error: the number of iterations should be larger than saveStep + beginSaveIters"
            exit()

        for i in range(1, self.iterations + 1):
            print "Iteration", i
            if i >= self.beginSaveIters and (((i - self.beginSaveIters) % self.saveStep) == 0):
                print "Saving model..."
                self.updateEstimatedParameters()
                self.saveIteratedModel(i)

            # 用吉布斯采样更新z
            for m in range(self.M):
                N = len(self.doc[m])
                for n in range(N):
                    newTopic = self.sampleTopicZ(m, n)
                    self.z[m][n] = newTopic

        # 存储topic model
        with open("lda.phi", 'wb') as f:
            pickle.dump(self.phi, f)

        # 存储字典
        with open("lda.dic", 'wb') as f:
            pickle.dump(self.dic, f)

    def updateEstimatedParameters(self):
        for k in range(self.K):
            for t in range(self.V):
                self.phi[k][t] = (self.nkt[k][t] + self.beta) / (self.nktSum[k] + self.V * self.beta)

        for m in range(self.M):
            for k in range(self.K):
                self.theta[m][k] = (self.nmk[m][k] + self.alpha) / (self.nmkSum[m] + self.K * self.alpha)

    def sampleTopicZ(self, m, n):
        oldTopic = self.z[m][n]
        self.nmk[m][oldTopic] -= 1
        self.nkt[oldTopic][self.doc[m][n]] -= 1
        self.nmkSum[m] -= 1
        self.nktSum[oldTopic] -= 1

        p = {}  # 选择每个topic的概率
        for k in range(self.K):
            p[k] = (self.nkt[k][self.doc[m][n]] + self.beta) / (self.nktSum[k] + self.V * self.beta) * (
                self.nmk[m][k] + self.alpha) / (self.nmkSum[m] + self.K * self.alpha)

        # Compute cumulated probability for p
        for k in range(self.K):
            if k != 0:
                p[k] += p[k - 1]

        u = random.random() * p[self.K - 1]
        for newTopic in range(self.K):
            if u < p[newTopic]:
                break

        self.nmk[m][newTopic] += 1
        self.nkt[newTopic][self.doc[m][n]] += 1
        self.nmkSum[m] += 1
        self.nktSum[newTopic] += 1
        return newTopic

    def saveIteratedModel(self, iters):
        model = "lda_%s" % iters
        dictionary = {k: v for v, k in self.dic.token2id.items()}
        # 存储此时的topic model
        with open(model, 'a') as f:
            for k in range(self.K):
                f.write(str(k) + '\t')
                termPros = sorted(self.phi[k].items(), key=lambda d: d[1], reverse=True)
                for i in range(20):  # 取前10个词
                    f.write(str(termPros[i][1]) + '*' + dictionary[termPros[i][0]].encode('utf-8') + '\t')
                f.write('\n')


def inferenceModelToNewText(text, dic, phi, K, iterations, alpha):
    '''
    利用训练好的lda model，得到新来的文章的topic分布
    :param text:  输入的新文档
    :param dic:  输入字典
    :param phi: topic-word分布矩阵，也就是训练好的topic model
    :param K:  topic的个数
    :param iterations: 迭代次数
    :param alpha: 超参数
    :return: 返回这篇文章在topic上的分布
    '''
    # init
    N = len(text)
    z = {}
    nk = {k: 0 for k in range(K)}
    for n in range(N):
        initTopic = random.randint(0, K - 1)  # 随机选取一个topic id
        z[n] = initTopic  # 随机给每个词分配topic
        nk[initTopic] += 1.0  # 在该文档中统计topic出现的次数

    # iteration Gibbs Sampling
    for i in range(iterations):
        for n in range(N):
            oldTopic = z[n]
            nk[oldTopic] -= 1

            ni = dic[text[n]]  # 获取词的id
            p = {}
            for k in range(K):
                p[k] = phi[k][ni] * (nk[k] + alpha) / (sum(nk) + K * alpha)
            for k in range(K):
                if k != 0:
                    p[k] += p[k - 1]

            u = random.random() * p[K - 1]
            for newTopic in range(K):
                if u < p[newTopic]:
                    break
            z[n] = newTopic
            nk[newTopic] += 1

    theta = {k: (nk[k] + alpha) / (sum(nk) + K * alpha) for k in nk}
    return theta
