import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("bank-full.csv")
header_data = list(data)
for i in range(10, 16):
    data = pd.concat([data, pd.get_dummies(data[header_data[i]], prefix= header_data[i])],axis=1)


input = data.drop(header_data[10:16], axis = 1)
output = data["y"]

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=5)

input_test.shape,input_train.shape

output_train.mean(), output_test.mean()

tree=DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=5  )

tree.fit(input_train,output_train)

from sklearn.metrics  import accuracy_score

accuracy_score(output_test,tree.predict(input_test))

tree_params={'max_depth': range(1, 20)}

tree=DecisionTreeClassifier()

tree_search=GridSearchCV(tree,tree_params,cv=10,n_jobs=-1,verbose=1)

tree_search.fit(input_train,output_train)

tree_search.best_params_

tree_search.best_score_

tree_search.cv_results_['mean_test_score']

plt.scatter(tree_params['max_depth'],tree_search.cv_results_['mean_test_score']);
