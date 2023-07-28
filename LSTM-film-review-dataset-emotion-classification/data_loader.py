# coding: UTF-8
import numpy as np
import tensorflow as tf
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# 加载数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
# 数据集形状
print(x_train.shape)
# 读进来的数据是已经转换成ID映射的，而一般的数据读进来都是词语，都需要手动转换成ID映射的
# print(x_train[0]) # 部分运行结果： [ 1, 14, 22, 16,,,]
# 分类
np.unique(y_train)

# 加载单词对照的索引
_word2idx = tf.keras.datasets.imdb.get_word_index()
# 将整体单词对应的索引都加 3 ，为下面三个字符腾出三个索引 0，1，2
word2idx = {w: i + 3 for w, i in _word2idx.items()}
# 空格对应 0
word2idx['<pad>'] = 0
# 每个影评开始对应 1
word2idx['<start>'] = 1
# 在加载的词汇表中找不到影评中的某些字符，都对应 2
word2idx['unk'] = 2
# 创建id与单词映射表
idx2word = {i: w for w, i in word2idx.items()}


# 按文本长度大小进行排序
def sort_by_len(x, y):
    x, y = np.asarray(x), np.asarray(y)
    idx = sorted(range(len(x)), key=lambda i: len(x[i]))
    return x[idx], y[idx]


# 对数据重新排序
x_train, y_train = sort_by_len(x_train, y_train)
x_test, y_test = sort_by_len(x_test, y_test)


def write_file(f_path, xs, ys):
    with open(f_path, 'w', encoding='utf-8') as f:
        for x, y in zip(xs, ys):
            f.write(str(y)+'\t'+' '.join([idx2word[i] for i in x][1:]) + '\n') # 从 1：开始，是因为前面有个start


# 将数据写入文件
write_file('./data/train.txt', x_train, y_train)
write_file('./data/test.txt', x_test, y_test)

# 构建语料表，基于词频进行统计
counter = Counter()
with open('./data/train.txt', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        label, words = line.split('\t')
        words = words.split(' ')
        counter.update(words)

words = ['<pad>'] + [w for w, freq in counter.most_common() if freq >= 10]
print('Vocab size:', len(words))

Path('./data/vocab').mkdir(exist_ok=True)

with open('./data/vocab/word.txt', mode='w', encoding='utf-8') as f:
    for w in words:
        f.write(w + '\n')

word2id = {}
with open('./data/vocab/word.txt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.rstrip()
        word2id[line] = i        

# 将已经训练好的词向量进行导入，创建embedding
# 做一个大表，里面有20598个不同的词，以及1个对应找不到的单词用的<UNK>(20599*50)
embedding = np.zeros((len(word2id) + 1, 50))
with open('./data/glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
    count = 0
    for i, line in enumerate(f):
        if i % 10000 == 0:
            print(f'- At line {i}')
        line = line.rstrip()
        sp = line.split(' ')
        word, vec = sp[0], sp[1]
        if word in word2id:
            count += 1
            embedding[word2id[word]] = np.asarray(vec, dtype='float32') # 将词转换成对应的向量
print(f"{count} / {len(word2id)} words have found pre-trained values")
np.save("./data/vocab/word.npy", embedding)
print("Save ./data/vocab/word.npy")
