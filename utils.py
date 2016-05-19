# coding: utf-8
"""
a package of functions for common use
"""
import codecs
import re
import jieba.posseg as pseg


def preprocess(text):
    """
    文本预处理
    :param text: 输入文本
    :return: 返回处理后的文本
    """
    text = re.sub("<[^>]*?>", "", text)  # 过滤html标签
    text = re.sub(r'http://[\x21-\x7e]+', '', text)  # 过滤网址
    text = text.replace("\n", "")  # 去除换行符
    return text


def getWordlist(text):
    """
    分词去停用词，返回词表
    :param text:
    :return: 返回这段文本的word list
    """
    stopwords = [line[:-1] for line in codecs.open("stopwords", 'r', 'utf-8')]  # 停用词表
    word_list = [w.word for w in pseg.cut(text) if
                 w.flag in ['n', 'nr', 'nrt', 'ns', 'nt', 'nz', 'nl', 'ng', 't', 'v', 'vn'] if w.word not in stopwords
                 if w.word != ' ']
    return word_list
