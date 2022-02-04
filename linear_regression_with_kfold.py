import pandas
from sklearn import linear_model

from sklearn.model_selection import KFold

dataset = pandas.read_csv("dataset.csv")

target = dataset.iloc[:,0].values
data = dataset.iloc[:,3:9].values

kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)

print(kfold_object)

for training_index, test_index in kfold_object.split(data):
	print("Training Index:  ")
	print(training_index)
	print("Test Index:  ")
	print (test_index)
	print ("\n\n")
	data_training = data[training_index]
	target_training = target[training_index]
	data_test = data[test_index]
	machine = linear_model.LinearRegression()
	machine.fit(data_training, target_training)
	new_target = machine.predict(data_test)
	print(metrics.r2_score(target_test, new_target))

