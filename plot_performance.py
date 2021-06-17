import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import pickle
if __name__ == '__main__':
    algorithm_name = "AGEM"
    with open('{}_result.pkl'.format(algorithm_name), 'rb') as f:
        data = pickle.load(f)
    with open("{}_index.pkl".format(algorithm_name), 'rb') as f:
        index = pickle.load(f)
    index = index.transpose()
    data = data.transpose()
    plt.rcParams['figure.figsize'] = (7.0, 4.5)
    sbn.set()
    handles = []
    legend_names = []
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.ylim([-1000, 2000])
    for i in range(data.shape[0]):
        handle, = plt.plot(index[:], data[i,:])
        handles.append(handle)
        legend_names.append("Task{}".format(i+1))
    plt.legend(handles=handles, labels = legend_names, loc="upper left")
    plt.show()
    # plt.savefig('{}_all.pdf'.format(algorithm_name))