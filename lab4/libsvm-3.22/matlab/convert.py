import random
from sklearn import datasets

def main():
	# 得到数据集
	iris = datasets.load_iris()
	iris_feature, iris_target = iris.data, iris.target
	# print(iris_target)
	
	fo_train = open("iris_data_train.txt", 'w')
	fo_test = open("iris_data_test.txt", 'w')

	for i in range(len(iris_feature)):
		target = iris_target[i]
		feature = ''
		for j in range( len(iris_feature[0])):
			feature += str(j+1) + ':' + str(iris_feature[i][j]) + ' '
		line = str(target) + ' ' +feature[:-1] + '\n'
		if random.randint(0, 10) < 3:
			fo_test.write(line)
		else:
			fo_train.write(line)

	fo_train.close()
	fo_test.close()

if __name__ == '__main__':
	main()