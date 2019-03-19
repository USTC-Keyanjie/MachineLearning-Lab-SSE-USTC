from __future__ import print_function
import numpy as np
from sklearn import datasets
from collections import Counter
from sklearn.datasets import make_classification


def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    return X[idx], y[idx]


# 正规化数据集 X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)

# 标准化数据集 X
def standardize(X):
    X_std = np.zeros(X.shape)
    # 每一列求均值
    mean = X.mean(axis=0)
    # 每一列求标准差
    std = X.std(axis=0)
    # print(mean)
    # print(std)

    for col in range(np.shape(X)[1]):
        # 感觉这里写错了，应该是：
        X_std[:, col] = (X[:, col] - mean[col]) / std[col]
        # X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]

    return X_std


def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)

    n_train_samples = int(X.shape[0] * (1 - test_size))
    x_train, x_test = X[:n_train_samples], X[n_train_samples:]
    y_train, y_test = y[:n_train_samples], y[n_train_samples:]

    return x_train, x_test, y_train, y_test


def accuracy(y, y_pred):
    y = y.reshape(y.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)

    return np.sum(y == y_pred) / len(y)


class KNN:
    """
    K 近邻分类算法
    Parameters:
    -----------
    k: int 最近邻个数
    """
    def __init__(self, k=5):
        self.k = k

    def euclidean_distance(self, one_sample, X_train):
        one_sample = one_sample.reshape(1, -1)
        # 不太懂这句话有啥用
        X_train = X_train.reshape(X_train.shape[0], -1)
        # np.tile(one_sample, (X_train.shape[0], 1)): one_sample.shape由(1, 4) 逐行复制为 (134, 4)
        # distance.shape=(134, ) 表示此样例到每个点的距离
        distances = np.power(np.tile(one_sample, (X_train.shape[0], 1)) - X_train, 2).sum(axis=1)

        return distances

    # 获取k个近邻的类别标签
    def get_k_neighbor_labels(self, distances, y_train, k):
        k_neighbor_labels = []
        pre_distance = -1
        # np.sort(distances)[:k] 找出前k小的距离
        # print("np.sort(distances)[:k]=", np.sort(distances)[:k])
        for distance in np.sort(distances)[:k]:
            if distance == pre_distance:
                continue
            else:
                # print("distance=", distance)
                # 这里的a==b就是将a与b中每一个数值进行比对，不等就是False，相等就是True。
                # y[a==b]，就是求a==b中为True的那个值所对应的index，在y中对应的数。
                label = y_train[distances == distance]
                # print("label=", label)
                for l in label:
                    k_neighbor_labels.append(l)
                pre_distance = distance

        # print("k_neighbor_labels", k_neighbor_labels)
        return np.array(k_neighbor_labels).reshape(-1, )

    def vote(self, one_sample, X_train, y_train, k):
        distances = self.euclidean_distance(one_sample, X_train)

        # y_train.shape: (134，) -> (134, 1)
        y_train = y_train.reshape(y_train.shape[0], 1)
        k_neighbor_labels = self.get_k_neighbor_labels(distances, y_train, k)
        # print(k_neighbor_labels.shape)
        find_label, find_count = 0, 0

        # k_neighbor_labels中记录了最靠近样本点的k个点的标签
        # Counter(k_neighbor_labels)返回一个字典序，表示每个label出现的次数
        # find_count记录出现最多的count，find_label表示出现最多的label
        for label, count in Counter(k_neighbor_labels).items():
            if count > find_count:
                find_count = count
                find_label = label

        return find_label

    def predict(self, X_test, X_train, y_train):
        y_pred = []

        for sample in X_test:
            label = self.vote(sample, X_train, y_train, self.k)
            y_pred.append(label)

        return np.array(y_pred)


if __name__ == '__main__':
    # n_samples=200 200条样本
    # n_features=4 4个特征
    # n_informative=2 其中2个特征是有信息的特征
    # n_redundant=2 2个特征是冗余特征
    # n_repeated=0 没有重复特征
    # n_classes=2 二分类问题
    data = make_classification(n_samples=200, n_features=4, n_informative=2,
                               n_redundant=2, n_repeated=0, n_classes=2)

    X, y = data[0], data[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
    clf = KNN(k=5)

    y_pred = clf.predict(X_test, X_train, y_train)

    accu = accuracy(y_test, y_pred)
    print("Accurary:{:.2f}%".format(float(accu*100)))

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)
    best_acc = 0
    best_k = 0
    for i in range(X_train.shape[0]):
        clf = KNN(k=i)
        y_pred = clf.predict(X_test, X_train, y_train)
        accu = accuracy(y_test, y_pred)
        if accu > best_acc:
            best_acc = accu
            best_k = i
        print("Accurary:{:.2f}%".format(float(accu * 100)))

    print("The best accurary is {:.2f}% when k is {}".format(float(best_acc * 100), best_k))
