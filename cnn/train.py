#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  train.py
#       Author @  LiuSong
#  Create date @  2018/12/19 9:35
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************

import tensorflow as tf
from sklearn.model_selection import train_test_split
import data_processor
import numpy as np

#嵌入矩阵的维度
embed_dim = 32
#用户ID个数
uid_max = 6041
#性别个数
gender_max =2
#年龄类别个数
age_max = 7
#职业个数
job_max = 21
#电影ID个数
movie_id_max =3953
#电影类型个数
movie_categories_max =19
#电影名单词个数
movie_title_max =5216


#电影名长度
sentences_size =15
#文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
#文本卷积核数量
filter_num = 8

# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256
dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20


#————————————————————————————————————————————
#—————————————----—网络结构------------------———————————————
#————————————————————————————-————————————————
#定义输入的占位符
def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
    movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.placeholder(tf.float32, name="LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob

#定义User的嵌入矩阵
def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name="uid_embed_matrix")
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")

        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1),
                                          name="gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")

        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")

        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), name="job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer

#将User的嵌入矩阵一起全连接生成User的特征
def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope("user_fc"):
        # 第一层全连接
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name="gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name="job_fc_layer", activation=tf.nn.relu)

        # 第二层全连接
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  # (?, 1, 128)
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
    return user_combine_layer, user_combine_layer_flat

#定义Movie ID的嵌入矩阵
def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1), name = "movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name = "movie_id_embed_layer")
    return movie_id_embed_layer

#对电影类型的多个嵌入向量做平均
def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1), name = "movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories, name = "movie_categories_embed_layer")   #(?,18,32)
        movie_categories_embed_layer = tf.reduce_mean(movie_categories_embed_layer, axis=1, keepdims=True)     #(?,1,32)
    return movie_categories_embed_layer



#Movie Title的文本卷积网络实现¶
def get_movie_cnn_layer(movie_titles):
    # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1),
                                               name="movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name="movie_title_embed_layer") #(?,15,32)
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)   #{?,15,32,1}

    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                         name="filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                      name="conv_layer")   #(?,15-2+1,1,8)
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1],
                                           padding="VALID", name="maxpool_layer")  #(?,1,1,8)
            pool_layer_lst.append(maxpool_layer)

    # Dropout层
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")  #(-1,1,1,32)
        max_num = len(window_sizes) * filter_num       #32
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")   #(?,1,32)

        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
    return pool_layer_flat, dropout_layer

#将Movie的各个层一起做全连接¶
def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    with tf.name_scope("movie_fc"):
        # 第一层全连接
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name="movie_id_fc_layer",
                                            activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim,
                                                    name="movie_categories_fc_layer", activation=tf.nn.relu)

        # 第二层全连接
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer, movie_combine_layer_flat



#——————————————————————————————————————————————
#———————————————构建计算图———————————————————————————
#——————————————————————————————————————————————
#获取输入占位符
uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = get_inputs()
#获取User的4个嵌入向量
uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender, user_age, user_job)
#得到用户特征
user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer)
#获取电影ID的嵌入向量
movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
#获取电影类型的嵌入向量
movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
#获取电影名的特征向量
pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles)
#得到电影特征
movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                                movie_categories_embed_layer,
                                                                                dropout_layer)
#计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
with tf.name_scope("inference"):
    # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
    #inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  # (?, 400)
    #inference = tf.layers.dense(inference_layer, 1,
    #                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
    #                                     kernel_regularizer=tf.nn.l2_loss, name="inference")

    inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1) #(?,)
    inference = tf.expand_dims(inference, axis=1)   #(?,1)
with tf.name_scope("loss"):
    # MSE损失，将计算值回归到评分
    cost = tf.losses.mean_squared_error(targets, inference )
    loss = tf.reduce_mean(cost)
    tf.summary.scalar('loss', loss)

# 优化损失
global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss,global_step)  #cost

#定义准确率
# 合并所有的summary
merged = tf.summary.merge_all()

def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]
#————————————————————————————————————————————
#——————————————开始训练--------------------------—————————————
#————————————————————————————-————-———————————


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/cnn/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/cnn/test', sess.graph)

    saver = tf.train.Saver()
    #获取特征和标签
    features,targets_values=data_processor.process('../data/users.dat','../data/movies.dat','../data/ratings.dat')
    train_X, test_X, train_y, test_y = train_test_split(features,targets_values,test_size=0.2,random_state=0)

    for epoch_i in range(num_epochs):
        train_batches = get_batches(train_X, train_y, batch_size)
        test_batches = get_batches(test_X, test_y, batch_size)
        for batch_i in range(len(train_X) // batch_size):
            X_batch, y_batch = next(train_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = X_batch.take(6, 1)[i]

            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = X_batch.take(5, 1)[i]
            # 训练集
            feed = {
                uid:np.reshape(X_batch.take(0,axis=1),[batch_size,1]),
                user_gender:np.reshape(X_batch.take(1,axis=1),[batch_size,1]),
                user_age:np.reshape(X_batch.take(2,axis=1),[batch_size,1]),
                user_job:np.reshape(X_batch.take(3,axis=1),[batch_size,1]),
                movie_id:np.reshape(X_batch.take(4,axis=1),[batch_size,1]),
                movie_categories:categories,
                movie_titles:titles,
                targets:np.reshape(y_batch, [batch_size, 1]),
                dropout_keep_prob: dropout_keep,  # dropout_keep
                lr: learning_rate
            }
            train_loss,summary,step,_ = sess.run([loss, merged,global_step,optimizer], feed_dict=feed)
            # 记录训练集计算的参数
            train_writer.add_summary(summary, step)
            if batch_i % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))

        #测试集上的表现

        for batch_i  in range(len(test_X) // batch_size):

            X_batch_test, y_batch_test = next(test_batches)
            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = X_batch_test.take(6, 1)[i]

            titles = np.zeros([batch_size, sentences_size])
            for i in range(batch_size):
                titles[i] = X_batch_test.take(5, 1)[i]
            feed = {
                uid: np.reshape(X_batch_test.take(0, axis=1), [batch_size, 1]),
                user_gender: np.reshape(X_batch_test.take(1, axis=1), [batch_size, 1]),
                user_age: np.reshape(X_batch_test.take(2, axis=1), [batch_size, 1]),
                user_job: np.reshape(X_batch_test.take(3, axis=1), [batch_size, 1]),
                movie_id: np.reshape(X_batch_test.take(4, axis=1), [batch_size, 1]),
                movie_categories: categories,
                movie_titles: titles,
                targets: np.reshape(y_batch_test, [batch_size, 1]),
                dropout_keep_prob: 1.0,  # dropout_keep
                lr: learning_rate
            }
            test_loss,step,summary= sess.run([loss, global_step,merged], feed_dict=feed)
            test_writer.add_summary(summary, step)
            if batch_i % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    (len(test_X) // batch_size),
                    test_loss))
    #Save Model
    saver.save(sess, "../data/save")
    print('Model Trained and Saved')