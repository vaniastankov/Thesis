import pandas as pd
from pathlib import Path
import algorithms
import tests
import numpy as np
import similarities

import extract
import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
data_folder = Path("D:\Python Projects\\thesis\data\ml-latest-small")

data_split = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
rates = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
ks = [1,5,10,15,20]
algorithm_list = [[algorithms.random_filter, "Random Filter"],
             [algorithms.simple_item_based_filter, "Simple Item Based Filter"],
             [algorithms.simple_user_based_filter, "Simple User Based Filter"],
             [algorithms.svd_filter, "SVD Filter"],
             [algorithms.slope_one, "Slope One"]]

rand_list = [[algorithms.random_filter, "Usual Random Filter"],
             [algorithms.random_user_filter, "User Mean Random Filter"],
             [algorithms.random_item_filter, "Item Mean Random Filter"]]

if __name__ == "__main__":
    ratings = pd.read_csv(data_folder / "ratings.csv").drop(columns="timestamp")
    #test, train = extract.set_divide(ratings, 0.3)
    tests.prk_plots(algorithm_list,ks,ratings)
    #tests.tq_plots(algorithm_list,data_split,ratings)




    # #SVD VAR K
    # ubfs = [algorithms.svd_k(test,train,ratings, 10)[0],
    #        algorithms.svd_k(test,train,ratings, 30)[0],
    #        algorithms.svd_k(test,train,ratings, 100)[0]]
    # for res in ubfs:
    #     mae, rmse, rmsle = metrics.accuracy_metrics(test,res)
    #     p,re,f1 = metrics.order_dependent_metrics(test,res,3.5,10)
    #     print(mae,rmse,rmsle,p,f1)


    # UBF VAR K
    # ubfs = [algorithms.simple_ibf_var_k(test,train,ratings, 100)[0],
    #        algorithms.simple_ibf_var_k(test, train, ratings, 400)[0],
    #        algorithms.simple_ibf_var_k(test, train, ratings, 800)[0]]
    # for res in ubfs:
    #     mae, rmse, rmsle = metrics.accuracy_metrics(test,res)
    #     p,re,f1 = metrics.order_dependent_metrics(test,res,3.5,10)
    #     print(mae,rmse,rmsle,p,f1)
    #
    #


    # UBF VAR K
    # ubfs = [algorithms.simple_ubf_var_k(test,train,ratings, 5)[0],
    #        algorithms.simple_ubf_var_k(test, train, ratings, 30)[0],
    #        algorithms.simple_ubf_var_k(test, train, ratings, 100)[0]]
    # for res in ubfs:
    #     mae, rmse, rmsle = metrics.accuracy_metrics(test,res)
    #     p,re,f1 = metrics.order_dependent_metrics(test,res,3.5,10)
    #     print(mae,rmse,rmsle,p,f1)

    # UBF VAR SIM
    # ubfs = [algorithms.simple_ubf_var_sim(test,train,ratings, similarities.cosine_similarity)[0],
    #        algorithms.simple_ubf_var_sim(test, train, ratings, similarities.pearson)[0],
    #        algorithms.simple_ubf_var_sim(test, train, ratings, similarities.msd)[0]]
    # for res in ubfs:
    #     mae, rmse, rmsle = metrics.accuracy_metrics(test,res)
    #     p,re,f1 = metrics.order_dependent_metrics(test,res,3.5,10)
    #     print(mae,rmse,rmsle,p,f1)


    ## Testing Random filters
    # rand_list = [[algorithms.random_filter, "Usual Random Filter"],
    #              [algorithms.random_user_filter, "User Mean Random Filter"],
    #              [algorithms.random_item_filter, "Item Mean Random Filter"]]
    # for r,n in rand_list:
    #     res = r(test,train,ratings)[0]
    #     mae, rmse, rmsle = metrics.accuracy_metrics(test,res)
    #     p,re,f1 = metrics.order_dependent_metrics(test,res,3.5,10)
    #     print(n, mae,rmse,rmsle,p,f1)
