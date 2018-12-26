#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  user_processor.py
#       Author @  LiuSong
#  Create date @  2018/12/18 17:46
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import pandas as pd


def process(input_file):
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users_df = pd.read_csv(input_file, sep="::", header=None, names=users_title, engine="python")
    gender_map = {'F':0, 'M':1}
    #对性别的处理
    users_df['Gender']=users_df['Gender'].map(gender_map)
    origin_dict=users_df['Age'].value_counts().to_dict()
    dict = {}
    index = 0
    for zuhe in sorted(origin_dict.items(), key=lambda d: d[1], reverse=False):
        dict[zuhe[0]] = index
        index += 1
    #对年龄的处理
    users_df['Age']=users_df['Age'].map(dict)
    users_df=users_df.filter(items=['UserID','Gender','Age','JobID'])
    return users_df



if __name__ == "__main__":
    users_df=process("../data/users.dat")
    print(users_df)