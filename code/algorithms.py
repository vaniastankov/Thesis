import pandas as pd
import similarities as sim
import numpy as np
import math
import scipy
from surprise import SlopeOne
from surprise import Dataset
from surprise import Reader
import time


def random_filter(test,train,all):
    start = time.time()
    mean = train["rating"].mean()
    adjusted_ratings = train["rating"]
    adjusted_ratings = (adjusted_ratings - mean)**2
    sigma = np.sqrt(adjusted_ratings.mean())
    user_list = pd.Series(test["userId"]).drop_duplicates()
    np_results = []
    fit = time.time()
    fit_time = fit - start
    for u in user_list:
        watched = list(test[test.userId == u]['movieId'])
        for m in watched:
            rating = np.random.normal(mean, sigma)
            if (rating >5) or (rating<0):
                rating = mean
            np_results.append([u,m,rating])
    res = pd.DataFrame(np_results,columns=test.columns)
    predict_time = time.time() - fit
    overall= predict_time+fit - start
    return res, [fit_time,predict_time,overall]


def random_user_filter(test,train,all):
    start = time.time()
    # mean = train["rating"].mean()
    # adjusted_ratings = train["rating"]
    # adjusted_ratings = (adjusted_ratings - mean)**2
    # sigma = np.sqrt(adjusted_ratings.mean())
    user_list = pd.Series(test["userId"]).drop_duplicates()
    np_results = []
    fit = time.time()
    fit_time = fit - start
    for u in user_list:
        ur = train.loc[train.userId == u]['rating']
        um = ur.mean()
        adj = (ur - um)**2
        sgm = np.sqrt(adj.mean())
        watched = list(test[test.userId == u]['movieId'])
        for m in watched:
            rating = np.random.normal(um, sgm)
            if (rating >5) or (rating<0):
                rating = um
            np_results.append([u,m,rating])
    res = pd.DataFrame(np_results,columns=test.columns)
    predict_time = time.time() - fit
    overall= predict_time+fit - start
    return res, [fit_time,predict_time,overall]


def random_item_filter(test,train,all):
    start = time.time()
    # mean = train["rating"].mean()
    # adjusted_ratings = train["rating"]
    # adjusted_ratings = (adjusted_ratings - mean)**2
    # sigma = np.sqrt(adjusted_ratings.mean())
    user_list = pd.Series(test["userId"]).drop_duplicates()
    np_results = []
    fit = time.time()
    fit_time = fit - start
    for u in user_list:
        # ur = train.loc[all.userId == u]['rating']
        # um = ur.mean()
        # adj = (ur - um)**2
        # sgm = np.sqrt(adj.mean())
        watched = list(test[test.userId == u]['movieId'])
        for m in watched:
            ir = train.loc[train.movieId == m]['rating']
            im = ir.mean()
            adj = (ir - im) ** 2
            sgm = np.sqrt(adj.mean())
            rating = np.random.normal(im, sgm)
            if (rating >5) or (rating<0):
                rating = im
            np_results.append([u,m,rating])
    res = pd.DataFrame(np_results,columns=test.columns)
    predict_time = time.time() - fit
    overall= predict_time+fit - start
    return res, [fit_time,predict_time,overall]


def simple_user_based_filter(test,train,all):

    def rate(mid, geeks, ratings,mean):
        row = ratings[mid][ratings.index.isin(geeks)]
        ln=len(row)
        for i in row.values:
            if i==0:
                ln = ln-1
        if ln !=0:
            return row.sum()/ln
        else:
            return mean

    start = time.time()
    accelerated = train.copy()
    mean = train["rating"].mean()
    pivot = pd.pivot_table(accelerated, values='rating', index='userId', columns='movieId').apply(
        lambda row: row.fillna(0), axis=1)
    all_pivot = pd.pivot_table(all, values='rating', index='userId', columns='movieId').apply(
        lambda row: row.fillna(0),axis=1)
    similarities = sim.pearson(pivot)
    np.fill_diagonal(similarities, 0)
    fit = time.time()
    fit_time = fit - start
    user_list = pd.Series(test["userId"]).drop_duplicates()
    results =[]
    for u in user_list:
        geeks = pd.Series(similarities[u-1])
        geeks.sort_values(ascending = False,inplace= True)
        geeks = geeks.index[:30]
        watched = list(test[test.userId == u]['movieId'])
        for m in watched:
            res = rate(m,geeks,all_pivot,mean)
            results.append([u,m,res])
    out = pd.DataFrame(results, columns=test.columns, index =test.index)
    predict_time = time.time() - fit
    overall = predict_time + fit - start
    return out, [fit_time,predict_time,overall]


def simple_item_based_filter(test, train, all):
    start = time.time()
    accelerated = train.copy()
    mean = train["rating"].mean()
    pivot = pd.pivot_table(accelerated, values='rating', index='movieId', columns='userId').apply(
        lambda row: row.fillna(0), axis=1)
    similarities = sim.cosine(pivot)
    np.fill_diagonal(similarities, 0)
    user_list = list(pd.Series(accelerated["userId"]).drop_duplicates())
    movie_list = list(pd.Series(accelerated["movieId"]).drop_duplicates())
    test_users = list(pd.Series(test["userId"]).drop_duplicates())
    labeled_sim = pd.DataFrame(similarities, index=movie_list, columns=movie_list)
    out = []
    fit = time.time()
    fit_time = fit - start
    for u in test_users:
        test_movies = list(test.loc[test.userId == u]['movieId'])
        if u in user_list:
            avg = train.loc[all.userId == u]['rating'].mean()
            for m in test_movies:
                if m in movie_list:
                    similar = pd.Series(labeled_sim.loc[m])
                    similar.sort_values(ascending = False,inplace= True)
                    similar = similar.index[:400]
                    rating = accelerated.loc[accelerated.userId == u].loc[accelerated.movieId.isin(similar)]['rating'].mean()
                    if math.isnan(rating):
                        rating=mean
                    out.append([u, m, rating])
                else:
                    out.append([u,m,avg])
    out = pd.DataFrame(out, columns=test.columns, index=test.index)
    predict_time = time.time() - fit
    overall = predict_time + fit - start
    return out, [fit_time,predict_time,overall]


def svd_filter(test,train,all):
    feature_num = 30

    def svd(matrix, k):
        U, s, V = np.linalg.svd(matrix, full_matrices=False)
        s = np.diag(s)
        s = s[0:k, 0:k]
        U = U[:, 0:k]
        V = V[0:k, :]
        s_root = scipy.linalg.sqrtm(s)
        Usk = np.dot(U, s_root)
        skV = np.dot(s_root, V)
        UsV = np.dot(Usk, skV)
        UsV = UsV
        return UsV

    start = time.time()
    accelerated = train.copy()
    pivot = pd.pivot_table(accelerated, values='rating', index='userId', columns='movieId')
    matrix = pivot.values
    user_num, movie_num = matrix.shape
    item_means = []
    user_means = []
    for u in range(user_num):
        user_scores = matrix[u]
        user_not_nan_scores = user_scores[~np.isnan(user_scores)]
        user_means.append(np.mean(user_not_nan_scores))
    for m in range(movie_num):
        item_scores = matrix[:, m]
        item_not_nan_scores = item_scores[~np.isnan(item_scores)]
        item_means.append(np.mean(item_not_nan_scores))
    for m in range(movie_num):
        for u in range(user_num):
            if np.isnan(matrix[u][m]):
                matrix[u][m] = item_means[m]
    x = np.tile(item_means, (matrix.shape[0], 1))
    matrix = matrix - x
    user_list = list(pd.Series(accelerated["userId"]).drop_duplicates())
    user_dict = {user_list[i]: i for i in range(user_num)}
    movie_list = list(pd.Series(accelerated["movieId"]).drop_duplicates())
    movie_dict = {movie_list[i]: i for i in range(movie_num)}
    SVD = svd(matrix, feature_num) + x
    test_users = list(pd.Series(test["userId"]).drop_duplicates())
    out =[]
    fit = time.time()
    fit_time = fit - start
    for u in test_users:
        test_movies = list(test.loc[test.userId == u]['movieId'])
        if u in user_list:
            for m in test_movies:
                if m in movie_list:
                    out.append([u,m,SVD[user_dict[u]][movie_dict[m]]])
                else:
                    out.append([u, m, user_means[user_dict[u]]])
    out = pd.DataFrame(out, columns=test.columns, index=test.index)
    predict_time = time.time() - fit
    overall = predict_time + fit - start
    return out, [fit_time,predict_time,overall]

def slope_one(test,train,all):
    start = time.time()
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)
    test_data = Dataset.load_from_df(test[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    testset = test_data.build_full_trainset().build_testset()
    algo = SlopeOne()
    algo.fit(trainset)
    fit = time.time()
    fit_time = fit - start
    predictions = algo.test(testset)
    uid = []
    mid = []
    rate =[]
    for i in range(len(predictions)):
        uid.append(predictions[i].uid)
        mid.append(predictions[i].iid)
        rate.append(predictions[i].est)
    out = {'userId': uid, 'movieId': mid, 'rating':rate}
    out = pd.DataFrame.from_dict(out)
    predict_time = time.time() - fit
    overall = predict_time + fit - start
    return out, [fit_time,predict_time,overall]


def simple_ubf_var_sim(test,train,all, sim_func):

    def rate(mid, geeks, ratings,mean):
        row = ratings[mid][ratings.index.isin(geeks)]
        ln=len(row)
        for i in row.values:
            if i==0:
                ln = ln-1
        if ln !=0:
            return row.sum()/ln
        else:
            return mean

    start = time.time()
    accelerated = train.copy()
    mean = train["rating"].mean()
    pivot = pd.pivot_table(accelerated, values='rating', index='userId', columns='movieId').apply(
        lambda row: row.fillna(0), axis=1)
    all_pivot = pd.pivot_table(all, values='rating', index='userId', columns='movieId').apply(
        lambda row: row.fillna(0),axis=1)
    similarities = sim_func(pivot)
    print("f")
    np.fill_diagonal(similarities, 0)
    fit = time.time()
    fit_time = fit - start
    user_list = pd.Series(test["userId"]).drop_duplicates()
    results =[]
    for u in user_list:
        geeks = pd.Series(similarities[u-1])
        geeks.sort_values(ascending = False,inplace= True)
        geeks = geeks.index[:30]
        watched = list(test[test.userId == u]['movieId'])
        for m in watched:
            res = rate(m,geeks,all_pivot,mean)
            results.append([u,m,res])
    out = pd.DataFrame(results, columns=test.columns, index =test.index)
    predict_time = time.time() - fit
    overall = predict_time + fit - start
    return out, [fit_time,predict_time,overall]


def simple_ubf_var_k(test,train,all, vss):

    def rate(mid, geeks, ratings,mean):
        row = ratings[mid][ratings.index.isin(geeks)]
        ln=len(row)
        for i in row.values:
            if i==0:
                ln = ln-1
        if ln !=0:
            return row.sum()/ln
        else:
            return mean

    start = time.time()
    accelerated = train.copy()
    mean = train["rating"].mean()
    pivot = pd.pivot_table(accelerated, values='rating', index='userId', columns='movieId').apply(
        lambda row: row.fillna(0), axis=1)
    all_pivot = pd.pivot_table(all, values='rating', index='userId', columns='movieId').apply(
        lambda row: row.fillna(0),axis=1)
    similarities = sim.cosine(pivot)
    print("f")
    np.fill_diagonal(similarities, 0)
    fit = time.time()
    fit_time = fit - start
    user_list = pd.Series(test["userId"]).drop_duplicates()
    results =[]
    for u in user_list:
        geeks = pd.Series(similarities[u-1])
        geeks.sort_values(ascending = False,inplace= True)
        geeks = geeks.index[:vss]
        watched = list(test[test.userId == u]['movieId'])
        for m in watched:
            res = rate(m,geeks,all_pivot,mean)
            results.append([u,m,res])
    out = pd.DataFrame(results, columns=test.columns, index =test.index)
    predict_time = time.time() - fit
    overall = predict_time + fit - start
    return out, [fit_time,predict_time,overall]


def simple_ibf_var_k(test, train, all, vss):
    start = time.time()
    accelerated = train.copy()
    mean = train["rating"].mean()
    pivot = pd.pivot_table(accelerated, values='rating', index='movieId', columns='userId').apply(
        lambda row: row.fillna(0), axis=1)
    similarities = sim.cosine(pivot)
    np.fill_diagonal(similarities, 0)
    user_list = list(pd.Series(accelerated["userId"]).drop_duplicates())
    movie_list = list(pd.Series(accelerated["movieId"]).drop_duplicates())
    test_users = list(pd.Series(test["userId"]).drop_duplicates())
    labeled_sim = pd.DataFrame(similarities, index=movie_list, columns=movie_list)
    out = []
    fit = time.time()
    fit_time = fit - start
    for u in test_users:
        test_movies = list(test.loc[test.userId == u]['movieId'])
        if u in user_list:
            avg = train.loc[all.userId == u]['rating'].mean()
            for m in test_movies:
                if m in movie_list:
                    similar = pd.Series(labeled_sim.loc[m])
                    similar.sort_values(ascending = False,inplace= True)
                    similar = similar.index[:vss]
                    rating = accelerated.loc[accelerated.userId == u].loc[accelerated.movieId.isin(similar)]['rating'].mean()
                    if math.isnan(rating):
                        rating=mean
                    out.append([u, m, rating])
                else:
                    out.append([u,m,avg])
    out = pd.DataFrame(out, columns=test.columns, index=test.index)
    predict_time = time.time() - fit
    overall = predict_time + fit - start
    return out, [fit_time,predict_time,overall]


def svd_k(test,train,all,kdd):
    feature_num = kdd

    def svd(matrix, k):
        U, s, V = np.linalg.svd(matrix, full_matrices=False)
        s = np.diag(s)
        s = s[0:k, 0:k]
        U = U[:, 0:k]
        V = V[0:k, :]
        # s_root = scipy.linalg.sqrtm(s)
        # Usk = np.dot(U, s_root)
        # skV = np.dot(s_root, V)
        # UsV = np.dot(Usk, skV)
        # UsV = UsV
        UsV = np.dot(np.dot(U,s),V)
        return UsV

    start = time.time()
    accelerated = train.copy()
    pivot = pd.pivot_table(accelerated, values='rating', index='userId', columns='movieId')
    matrix = pivot.values
    user_num, movie_num = matrix.shape
    item_means = []
    user_means = []
    for u in range(user_num):
        user_scores = matrix[u]
        user_not_nan_scores = user_scores[~np.isnan(user_scores)]
        user_means.append(np.mean(user_not_nan_scores))
    for m in range(movie_num):
        item_scores = matrix[:, m]
        item_not_nan_scores = item_scores[~np.isnan(item_scores)]
        item_means.append(np.mean(item_not_nan_scores))
    for m in range(movie_num):
        for u in range(user_num):
            if np.isnan(matrix[u][m]):
                matrix[u][m] = item_means[m]
    x = np.tile(item_means, (matrix.shape[0], 1))
    matrix = matrix - x
    user_list = list(pd.Series(accelerated["userId"]).drop_duplicates())
    user_dict = {user_list[i]: i for i in range(user_num)}
    movie_list = list(pd.Series(accelerated["movieId"]).drop_duplicates())
    movie_dict = {movie_list[i]: i for i in range(movie_num)}
    SVD = svd(matrix, feature_num) + x
    test_users = list(pd.Series(test["userId"]).drop_duplicates())
    out =[]
    fit = time.time()
    fit_time = fit - start
    for u in test_users:
        test_movies = list(test.loc[test.userId == u]['movieId'])
        if u in user_list:
            for m in test_movies:
                if m in movie_list:
                    out.append([u,m,SVD[user_dict[u]][movie_dict[m]]])
                else:
                    out.append([u, m, user_means[user_dict[u]]])
    out = pd.DataFrame(out, columns=test.columns, index=test.index)
    predict_time = time.time() - fit
    overall = predict_time + fit - start
    return out, [fit_time,predict_time,overall]

