import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random


def order_dependent_metrics(test,predicted,thrshld, k):

    def ak(actual, predicted,k):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(k):
            if actual[i]:
                if predicted[i]:
                    tp = tp+1
                else:
                    fn = fn+1
            else:
                if predicted[i]:
                    fp = fp+1
                else:
                    tn = tn+1

        if ((tp + fp) ==0) or ((tp + fn) == 0):
            pr = 0
            rc =0
        else:
            pr = tp/(tp+fp)
            rc = tp/(tp+fn)
        return pr,rc

    data = test.copy()
    data = data.rename(columns={"rating": "actual"})
    merged = pd.merge(data, predicted, on=['userId', 'movieId'])
    threshold = thrshld
    merged["hit"] = merged["actual"] >= threshold
    merged["predicted"] = merged["rating"] >= threshold
    user_list = list(pd.Series(test["userId"]).drop_duplicates())
    user_num = len(user_list)
    mp = 0
    mr = 0
    for u in user_list:
        actual = list(merged.loc[merged["userId"] == u].sort_values(by=['rating'], ascending=False)['hit'])
        predicted = list(merged.loc[merged["userId"] == u].sort_values(by=['rating'], ascending=False)['predicted'])
        if (k > 0) and (len(actual) >= k):
            res = ak(actual[:k], predicted[:k], k)
            mp = mp + res[0]
            mr = mr + res[1]
            if (res[0] == 0) or (res[1] == 0):
                user_num = user_num-1
        else:
            user_num = user_num-1

    mp = mp/max(user_num,1)
    mr = mr/max(user_num,1)
    f1 = 2*mp*mr/max((mp+mr),1)
    return mp, mr, f1


def accuracy_metrics(test, predicted):

    df = pd.merge(test, predicted, on=["userId", "movieId"])
    df['mae'] = abs(df['rating_y'] - df['rating_x'])
    df['rmse'] = (df['rating_y'] - df['rating_x'])**2
    df['rmsle'] = (np.log(df['rating_y']+1) - np.log(df['rating_x'] + 1)) ** 2

    mae = df['mae'].mean()
    rmse = np.sqrt(df['rmse'].mean())
    rmsle = np.sqrt(df['rmsle'].mean())
    #print(mae, rmse, rmsle)
    return mae, rmse, rmsle


def data_plot(data):
    mean = data["rating"].mean()
    adjusted_ratings = data["rating"]
    adjusted_ratings = (adjusted_ratings - mean) ** 2
    sigma = np.sqrt(adjusted_ratings.mean())
    import scipy.stats as stats
    plt.hist(list(data["rating"]), density=True, bins=10, label="Data")
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 100)
    plt.plot(kde_xs, stats.norm.pdf(kde_xs, mean, sigma), label="Expected Distribution")
    plt.plot(kde_xs, stats.norm.pdf(kde_xs, 3, sigma*0.7), label="Ideal Distribution")
    plt.legend(loc="upper right")
    plt.ylabel('probability')
    plt.xlabel('rating')
    plt.title("Data Distribution")
    group_folder = "D:/Python Projects/thesis/results\group/"
    plt.savefig("{}dataplot.pdf".format(group_folder))
    plt.show()


def accuracy_metrics_plot():
    mae = []
    rmse = []
    rmsle = []
    size = 10
    for i in np.linspace(0, size, 100):
        x = math.sin(i*4)+0.1
        # y = math.sqrt(i)
        y = 5*math.sqrt(math.cos(i)+i)
        mae.append(x-y)
        rmse.append((x-y)**2)
        rmsle.append((np.log(x+1) - np.log(y+1))**2)

    mmae = 0
    rrmse = 0
    rrmsle = 0
    for j in range(100):
        mmae = mmae + mae[j]
        rrmse = rrmse + rmse[j]
        rrmsle = rrmsle + rmsle[j]

    print(mmae/size, np.sqrt(rrmse/size), np.sqrt(rrmsle/size))
    ox = list(np.linspace(1, size+1, 100))
    plt.plot(ox, mae, label="Error")
    plt.plot(ox, rmse, label="Squared Error")
    plt.plot(ox, rmsle, label="Squared Log Error")
    plt.legend(loc="upper right")
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title("Error Plot")
    group_folder = "D:/Python Projects/thesis/results\group/"
    plt.savefig("{}Errrdrrr22.pdf".format(group_folder))
    plt.show()

