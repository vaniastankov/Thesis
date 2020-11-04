import random
import pandas as pd
import time

"""
This part of code is responsible for reshaping, slicing and modifying data
mostly ratings.
"""


def set_divide(ratings, l):
    """
    This function divides original data in test and train subsets
    ratings - original
    l - defines ratio between test/ratings
    each time random indexes are selected and dropped
    """
    train = ratings.copy()
    test = pd.DataFrame(columns=ratings.columns)
    user_list = list(pd.Series(ratings["userId"]).drop_duplicates())
    start = time.time()
    for i in user_list:
        watched = list(ratings[ratings.userId==i]['movieId'])
        random.shuffle(watched)
        amount = len(watched)
        watched = watched[0:int(amount*l)]
        row = train.loc[train["userId"] == i].loc[train["movieId"].isin(watched)]
        row_index = row.index
        test = pd.concat([test,row])
        train.drop(row_index, inplace=True)
        test.reset_index(drop=True)
        train.reset_index(drop=True)
    endd = time.time()
    print("1 ", endd-start)
    return test, train


def cut_train(test, predicted):
    """
    Taking predicted data, and, assuming that it also includes test data results - cutting test samples
    """
    cut_predicted = pd.merge(test[['userId', 'movieId']],predicted, on=['userId', 'movieId'])
    return cut_predicted

def reshape(dataframe):
    """
    Creating a pivot table out of dataframe(that was in form of ratings)
    Warning!!! There are 610 users, but only 9k movies, so that you cant iterate over movie id
    (Do something better instead)
    """
    return dataframe.pivot_table(index='movieId', columns='userId', values='rating')
