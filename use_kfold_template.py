import kfold_template
import pandas

dataset = pandas.read_csv("dataset.csv")
target = dataset.iloc[:,2].values
data = dataset.iloc[:,3:9].values
kfold_template.run_kfold(data, target, 4, linear_model.LogisticRegression(), 1, 1)

print(r2_scores)

r2_scores, accuracy_scores, confusion_matrices = kfold_template.run_kfold(data, target, 4, linear_model.LogisticRegression())

print(r2_scores)
print(accuracy_scores)

for confusion_matrices in confusion_matrices:
	print(confusion_matrices)