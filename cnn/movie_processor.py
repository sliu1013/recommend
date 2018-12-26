#!/usr/bin/env python
#coding=utf-8
# *******************************************************************
#     Filename @  movie_processor.py
#       Author @  LiuSong
#  Create date @  2018/12/18 19:51
#        Email @  
#  Description @  
#      license @ (C) Copyright 2015-2018, DevOps Corporation Limited.
# ********************************************************************
import pandas as pd
import re

def get_input(movie_input_file):
    """
    :param movie_input_file:
    :return:
    """
    movie_titles=['MovieID', 'Title', 'Genres']
    movies_df=pd.read_csv(movie_input_file,sep='::',header=None,names=movie_titles,engine='python')
    return movies_df


def get_info(movies_df):
    genres_map = dict()
    genres_index = 0
    for val in movies_df['Genres']:
        for item in val.split('|'):
            if (item not in genres_map):
                genres_map[item] = genres_index
                genres_index += 1

    title_map = dict()
    title_index = 0
    max_title_length = 0
    for title in movies_df['Title']:
        # 去除title中的年份
        title = re.sub(r'\(\d+\)', '', title).strip()
        titles = title.split(" ")
        if len(titles) > max_title_length:
            max_title_length = len(titles)
        for val in titles:
            if val not in title_map:
                title_map[val] = title_index
                title_index += 1
    return genres_map,title_map,max_title_length


def process_genres_feature(genre,genres_map):
    # 不足位数的用<PAD>填充，PAD取值为len(genres_map)
    genres = [len(genres_map)] * len(genres_map)
    flag = 0
    for key in genre.split('|'):
        genres[flag] = genres_map[key]
        flag += 1
    return genres


def process_title_feature(title,title_map,max_title_length):
    title_list=[len(title_map)]*max_title_length
    flag = 0
    for key in re.sub(r'\(\d+\)', '', title).strip().split(" "):
        title_list[flag] = title_map[key]
        flag += 1
    return title_list



def process(movies_df):
    genres_map, title_map, max_title_length = get_info(movies_df)
    movies_df['Genres']=movies_df['Genres'].apply(process_genres_feature,args=(genres_map,))
    movies_df['Title']=movies_df['Title'].apply(process_title_feature,args=(title_map,max_title_length,))
    return movies_df

if __name__ == "__main__":
    movies_df=get_input("../data/movies.dat")
    movies_df=process(movies_df)
    print(movies_df)
