# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 画图
from sklearn.externals.six import StringIO
import pydotplus


def main():
    # 得到数据集
    iris = datasets.load_iris()
    iris_feature, iris_target = iris.data, iris.target

    # 划分数据集
    feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
                                                                              shuffle=True)

    dt_model = tree.DecisionTreeClassifier(criterion='entropy')  # 所以参数均置为默认状态
    dt_model.fit(feature_train, target_train)  # 使用训练集训练模型

    predict_results = dt_model.predict(feature_test)  # 使用模型对测试集进行预测

    accuracy = accuracy_score(predict_results, target_test)  # 计算预测结果的准确度
    # 在 scikit-learn 中的分类决策树模型就带有 score 方法，只是传入的参数和 accuracy_score() 不太一致
    # scores = dt_model.score(feature_test, target_test)
    print('accuracy = ', accuracy)

    # 决策树可视化
    dot_data = StringIO()
    tree.export_graphviz(dt_model,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # 输出pdf，显示整个决策树的思维过程
    graph.write_pdf("iris_tree.pdf")


if __name__ == '__main__':
    main()
