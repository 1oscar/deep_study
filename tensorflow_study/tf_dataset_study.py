#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import  numpy as np

#batch 学习


BATCH_SIZE = 4
x = np.random.sample((10,2))
print(x,'\n==')
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE)
iter = dataset.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
    i=0
    while True:
        flag=1
        i+=1
        try:
            print(i,'\n ',sess.run(el))
        except Exception as e :
            flag=0
        if not flag:
            break



#repeat study
''':arg
repeat(1)：表示不重复，repeat(2): 表示对数据集重复一次，即迭代器，全部数据集第一次迭代完后，再来一次整体的迭代。 
每一次迭代输出的数据集数目就是batch_size大小。 
'''

# BATCHING
BATCH_SIZE = 4
x = np.random.sample((10,2))
print(x,'\n==')
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x).batch(BATCH_SIZE).repeat(1)
iter = dataset.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
    i=0
    while True:
        flag=1
        i+=1
        try:
            print(i,'\n ',sess.run(el))
        except Exception as e :
            flag=0
        if not flag:
            break


#shuffle
''':arg
shuffle the dataset is very important to avoid overfitting
随机打乱数据的顺序
buffer_size: A tf.int64 scalar tf.Tensor, representing the maximum number elements that will be buffered when prefetching.
就是把多少样本放入缓存中。 

'''


''':arg
shuffle,batch,repeat三者合起来使用
'''

import numpy as np
import tensorflow as tf
np.random.seed(0)
x = np.random.sample((11,2))
# make a dataset from a numpy array
print(x)

dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.shuffle(5)  # 将数据打乱，数值越大，混乱程度越大
dataset = dataset.batch(4)  # 按照顺序取出4行数据，最后一次输出可能小于batch
dataset = dataset.repeat()  # 数据集重复了指定次数
# repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
#为了配合输出次数，一般默认repeat()空

# create the iterator
iter = dataset.make_one_shot_iterator()
el = iter.get_next()

with tf.Session() as sess:
    for i in range(6):
        value = sess.run(el)
        print(i,' ',value)

