#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  data_processor.py
#       Author @  LiuSong
#  Create date @  2018/12/19 16:10
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import pandas as pd
import user_processor
import movie_processor
import numpy as np


def process(users_input_file,movies_data_file,ratings_data_file):
    users_df=user_processor.process(users_input_file)
    movies_df=movie_processor.get_input(movies_data_file)
    movies_df=movie_processor.process(movies_df)
    rating_list=['UserID','MovieID','ratings','timestamps']
    ratings_df=pd.read_csv(ratings_data_file,sep='::',header=None,names=rating_list,engine='python')
    ratings_df=ratings_df.filter(items=['UserID','MovieID','ratings'])
    data=pd.merge(pd.merge(users_df,ratings_df),movies_df)
    #将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    #dataframe转化为ndarray
    return features_pd.values,targets_pd.values


if __name__ == "__main__":
    users_input_file='../data/users.dat'
    movies_data_file='../data/movies.dat'
    ratings_data_file='../data/ratings.dat'
    features, targets=process(users_input_file,movies_data_file,ratings_data_file)
    #返回的是个list
    print(features.take(6, 1)[0])
    categories = np.zeros([10, 18])
    for i in range(10):
        categories[i] = features.take(6, 1)[i]
    print(categories.shape)


