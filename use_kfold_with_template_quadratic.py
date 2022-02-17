import kfold_template
import pandas
from sklearn import linear_model

dataset = pandas.read_csv("dataset.csv")

dataset['x3x4'] = dataset['x3']*dataset['x4']
dataset['x3sq'] = dataset['x3']**2
dataset['x3p4'] = dataset['x3']**4
dataset['x3p5'] = dataset['x3']**5
dataset['x3p6'] = dataset['x3']**6
dataset['x3p7'] = dataset['x3']**7
dataset['x3p8'] = dataset['x3']**8
dataset['x3p9'] = dataset['x3']**9

dataset 
print(dataset)

target = dataset.iloc[:,0].values
data = dataset.iloc[:,3:18].values

r2_scores = kfold_template.run_kfold(data, target, 4, linear_model.LinearRegression())




print(r2_scores)