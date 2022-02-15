import kfold_template
import pandas

dataset = pandas.read_csv("dataset.csv")
target = dataset.iloc[:,0].values
data = dataset.iloc[:,3:9].values
kfold_template.run_kfold(data, target, 4, linear_model.LogisticRegression(), 1, 1)