import algorithms
import extract
import metrics
import numpy as np
import similarities
import matplotlib.pyplot as plt


def tq_plots(algo,l_list, ratings):
    group_folder = "D:/Python Projects/thesis/results\group/"

    def select_m(data, a, metric):
        out = []
        for i in range(len(l_list)):
            out.append(data[i][a][metric])
        return out

    time_measures = []
    quality_measures =[]
    for i in l_list:
        tl = []
        ql = []
        test, train = extract.set_divide(ratings, i)
        for algorithm, name in algo:
            predictions, times = algorithm(test,train, ratings)
            tl.append(times)
            mae, rmse, rmsle = metrics.accuracy_metrics(test,predictions)
            pr, rc,f1 = metrics.order_dependent_metrics(test, predictions,3.5,10)
            ql.append([mae,rmse,rmsle,pr,rc,f1])
        time_measures.append(tl)
        quality_measures.append(ql)
    x = l_list

    time_plt1 = plt.figure(1)
    print("Train")
    for a in range(len(algo)):
        plt.plot(x, select_m(time_measures,a, 0), label=algo[a][1])
        print(algo[a][1], " ", select_m(time_measures,a, 0))
    plt.legend(loc="upper right")
    plt.ylabel('time')
    plt.xlabel('l')
    plt.title("Train Time")
    plt.savefig("{}Time{}.pdf".format(group_folder,1))
    time_plt2 = plt.figure(2)
    print("Predict")
    for a in range(len(algo)):
        plt.plot(x, select_m(time_measures,a, 1),label = algo[a][1])
        print(algo[a][1], " ", select_m(time_measures,a, 1))
    plt.legend(loc="upper right")
    plt.ylabel('time')
    plt.xlabel('l')
    plt.title("Predict Time")
    plt.savefig("{}Time{}.pdf".format(group_folder,2))
    time_plt3 = plt.figure(3)
    print("All")
    for a in range(len(algo)):
        plt.plot(x, select_m(time_measures, a, 2), label = algo[a][1])
        print(algo[a][1], " ", select_m(time_measures,a, 2))
    plt.legend(loc="upper right")
    plt.ylabel('time')
    plt.xlabel('l')
    plt.title("Total Time")
    plt.savefig("{}Time{}.pdf".format(group_folder,3))
    time_plt4 = plt.figure(4)
    print("MAE")
    for a in range(len(algo)):
        plt.plot(x, select_m(quality_measures, a, 0), label=algo[a][1])
        print(algo[a][1], " ", select_m(quality_measures, a, 0))
    plt.legend(loc="upper right")
    plt.ylabel('y')
    plt.xlabel('l')
    plt.title("Mean Average Error")
    plt.savefig("{}Accuracy{}.pdf".format(group_folder, 1))
    time_plt5 = plt.figure(5)
    print("RMSE")
    for a in range(len(algo)):
        plt.plot(x, select_m(quality_measures, a, 1), label=algo[a][1])
        print(algo[a][1], " ", select_m(quality_measures, a, 1))
    plt.legend(loc="upper right")
    plt.ylabel('y')
    plt.xlabel('l')
    plt.title("Root Mean Squared Error")
    plt.savefig("{}Accuracy{}.pdf".format(group_folder, 2))
    time_plt6 = plt.figure(6)
    print("RMSLE")
    for a in range(len(algo)):
        plt.plot(x, select_m(quality_measures, a, 2), label=algo[a][1])
        print(algo[a][1], " ", select_m(quality_measures, a, 2))
    plt.legend(loc="upper right")
    plt.ylabel('y')
    plt.xlabel('l')
    plt.title("Root Mean Squared Log Error")
    plt.savefig("{}Accuracy{}.pdf".format(group_folder, 3))
    time_plt7 = plt.figure(7)
    print("Precission@10")
    for a in range(len(algo)):
        plt.plot(x, select_m(quality_measures, a, 3), label=algo[a][1])
        print(algo[a][1], " ", select_m(quality_measures, a, 3))
    plt.legend(loc="upper right")
    plt.ylabel('y')
    plt.xlabel('l')
    plt.title("Precission@10")
    plt.savefig("{}Accuracy{}.pdf".format(group_folder, 4))
    time_plt8 = plt.figure(8)
    print("Recall@10")
    for a in range(len(algo)):
        plt.plot(x, select_m(quality_measures, a, 4), label=algo[a][1])
        print(algo[a][1], " ", select_m(quality_measures, a, 4))
    plt.legend(loc="upper right")
    plt.ylabel('y')
    plt.xlabel('l')
    plt.title("Recall@10")
    plt.savefig("{}Accuracy{}.pdf".format(group_folder, 5))
    time_plt9 = plt.figure(9)
    print("F1")
    for a in range(len(algo)):
        plt.plot(x, select_m(quality_measures, a, 5), label=algo[a][1])
        print(algo[a][1], " ", select_m(quality_measures, a, 5))
    plt.legend(loc="upper right")
    plt.ylabel('y')
    plt.xlabel('l')
    plt.title("F1")
    plt.savefig("{}Accuracy{}.pdf".format(group_folder, 6))
    z = 1000
    for a in range(len(algo)):
        time_pltdd = plt.figure(z)
        fig, ab = plt.subplots(3, 1)
        fig.tight_layout(pad=1.5)

        ab[0].plot(x, select_m(time_measures, a, 0))
        ab[0].set_title("Train Time for {}".format(algo[a][1]))
        ab[1].plot(x, select_m(time_measures, a, 1))
        ab[1].set_ylabel('time')
        ab[1].set_title("Predict Time for {}".format(algo[a][1]))
        ab[2].plot(x, select_m(time_measures, a, 2))
        ab[2].set_xlabel('l')
        ab[2].set_title("Total Time for {}".format(algo[a][1]))
        plt.savefig("{}MergedTime{}.pdf".format(group_folder, algo[a][1]))
        z = z + 3
        # time_pltdd = plt.figure(z)
        # plt.plot(x, select_m(time_measures, a, 0), label = algo[a][1])
        # plt.legend(loc="upper right")
        # plt.ylabel('time')
        # plt.xlabel('l')
        # plt.title("Train Time for {}".format(algo[a][1]))
        # plt.savefig("{}Time{}{}.pdf".format(group_folder,1,algo[a][1]))
        # time_pltdd = plt.figure(z+1)
        # plt.plot(x, select_m(time_measures, a, 1), label = algo[a][1])
        # plt.legend(loc="upper right")
        # plt.ylabel('time')
        # plt.xlabel('l')
        # plt.title("Predict Time for {}".format(algo[a][1]))
        # plt.savefig("{}Time{}{}.pdf".format(group_folder,2,algo[a][1]))
        # time_pltdd = plt.figure(z+2)
        # plt.plot(x, select_m(time_measures, a, 2), label = algo[a][1])
        # plt.legend(loc="upper right")
        # plt.ylabel('time')
        # plt.xlabel('l')
        # plt.title("Total Time for {}".format(algo[a][1]))
        # plt.savefig("{}Time{}{}.pdf".format(group_folder,3,algo[a][1]))
        # z=z+3



def pr_plots(algo,thrshld_list,ratings):
    group_folder = "D:/Python Projects/thesis/results\group/"
    test, train = extract.set_divide(ratings, 0.3)
    time_plt1 = plt.figure(11)
    for a, i in algo:
        p =[]
        r = []
        predicted = a(test,train,ratings)[0]
        for t in thrshld_list:
            res = metrics.order_dependent_metrics(test,predicted,t,10)
            p.append(res[0])
            r.append(res[1])
        plt.plot(p, r, label=i)
        c =0
        for n, j in zip(p, r):
            plt.annotate(thrshld_list[c], xy=(n, j))
            c = c+1
    plt.legend(loc="best")
    plt.ylabel('recall@10')
    plt.xlabel('precision@10')
    plt.title("Precision/Recall Curve")
    plt.savefig("{}PR{}.pdf".format(group_folder,1))
    k = 2
    for a, i in algo:
        p =[]
        r = []
        predicted = a(test,train,ratings)[0]
        for t in thrshld_list:
            res = metrics.order_dependent_metrics(test,predicted,t,10)
            p.append(res[0])
            r.append(res[1])
        time_plt1 = plt.figure(k+10)
        plt.plot(p, r, label=i)
        k = k+1
        c =0
        for n, j in zip(p, r):
            plt.annotate(thrshld_list[c], xy=(n, j))
            c = c+1
        plt.legend(loc="best")
        plt.ylabel('recall@10')
        plt.xlabel('precision@10')
        plt.title("Precision/Recall Curve for {}".format(i))
        plt.savefig("{}PR{}.pdf".format(group_folder, i))


def prk_plots(algo, k_list, ratings):
    group_folder = "D:/Python Projects/thesis/results\group/"
    test, train = extract.set_divide(ratings, 0.3)
    time_plt1 = plt.figure(111)
    for a, i in algo:
        p = []
        r = []
        predicted = a(test, train, ratings)[0]
        for k in k_list:
            res = metrics.order_dependent_metrics(test, predicted, 3.5, k)
            p.append(res[0])
            r.append(res[1])
        plt.plot(p, r, label=i)
        c = 0
        for n, j in zip(p, r):
            plt.annotate(k_list[c], xy=(n, j))
            c = c + 1
    plt.legend(loc="best")
    plt.ylabel('recall@k')
    plt.xlabel('precision@k')
    plt.title("Precision/Recall Curve at k")
    plt.savefig("{}PRK{}.pdf".format(group_folder, 1))
    f = 200
    for a, i in algo:
        p = []
        r = []
        predicted = a(test, train, ratings)[0]
        for k in k_list:
            res = metrics.order_dependent_metrics(test, predicted, 3.5, k)
            p.append(res[0])
            r.append(res[1])
        time_plt1 = plt.figure(f)
        plt.plot(p, r, label=i)
        f = f + 1
        c = 0
        for n, j in zip(p, r):
            plt.annotate(k_list[c], xy=(n, j))
            c = c + 1
        plt.legend(loc="best")
        plt.ylabel('recall@k')
        plt.xlabel('precision@k')
        plt.title("Precision/Recall Curve at k for {}".format(i))
        plt.savefig("{}PRK{}.pdf".format(group_folder, i))


