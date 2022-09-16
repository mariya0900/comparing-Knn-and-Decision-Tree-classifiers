import pandas as pd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score



data1=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data-numeric', sep='\s+', header=None)
data1.columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',  'Class']
print(data1)
Y = data1['Class']
X = data1.drop(['Class'],axis=1)



test_time_kNN = np.zeros(5)
test_Fmeasure_kNN = np.zeros(5)

#Part I (inference efficiency):
#builf KNN classifier for k=5
k=5
for i in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=i)
    clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    clf = clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)

    time_start=time.time()
    Y_predTest = clf.predict(X_test)
    time_end=time.time()

    test_Fmeasure_kNN[i] = f1_score(Y_test, Y_predTest, average='weighted')
    test_time_kNN[i]=time_end-time_start

#table for F-Measure and Test Time
data = {'Test 1': [test_Fmeasure_kNN[0], test_time_kNN[0]], 'Test 2': [test_Fmeasure_kNN[1], test_time_kNN[1]], 'Test 3': [test_Fmeasure_kNN[2], test_time_kNN[2]], 'Test 4': [test_Fmeasure_kNN[3], test_time_kNN[3]], 'Test 5': [test_Fmeasure_kNN[4], test_time_kNN[4]], 'Average': [np.average(test_Fmeasure_kNN), np.average(test_time_kNN)]}   
table=pd.DataFrame(data, index=['F-Measure', 'Test time'])
print("Table for Part I, 2 (kNN): \n, ", table)

bar_graph = pd.DataFrame({'Iteration': [1, 2, 3, 4, 5, 'average'], 'F-Measure for kNN': [test_Fmeasure_kNN[0], test_Fmeasure_kNN[1], test_Fmeasure_kNN[2], test_Fmeasure_kNN[3], test_Fmeasure_kNN[4], np.average(test_Fmeasure_kNN)]})
bar_graph.plot.bar(x='Iteration', y='F-Measure for kNN', rot = 0)
plt.show(block=True)
bar_graph = pd.DataFrame({'Iteration': [1, 2, 3, 4, 5, 'average'], 'Test time for kNN': [test_time_kNN[0], test_time_kNN[1], test_time_kNN[2], test_time_kNN[3], test_time_kNN[4], np.average(test_time_kNN)]})
bar_graph.plot.bar(x='Iteration', y='Test time for kNN', rot = 0)
plt.show(block=True)

#Part I, 3
#building decision tree classifier
test_time_DT = np.zeros(5)
test_Fmeasure_DT = np.zeros(5)
for i in range(5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=i)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)

    time_start=time.time()
    Y_predTest = clf.predict(X_test)
    time_end=time.time()

    test_Fmeasure_DT[i] = f1_score(Y_test, Y_predTest, average='weighted')
    test_time_DT[i]=time_end-time_start

#table for F-Measure and Test Time
data = {'Test 1': [test_Fmeasure_DT[0], test_time_DT[0]], 'Test 2': [test_Fmeasure_DT[1], test_time_DT[1]], 'Test 3': [test_Fmeasure_DT[2], test_time_DT[2]], 'Test 4': [test_Fmeasure_DT[3], test_time_DT[3]], 'Test 5': [test_Fmeasure_DT[4], test_time_DT[4]], 'Average': [np.average(test_Fmeasure_DT), np.average(test_time_DT)]}   
table=pd.DataFrame(data, index=['F-Measure', 'Test time'])
print("Table for Part I, 3 (decision trees): \n, ", table)

bar_graph = pd.DataFrame({"F-Measure for kNN":[test_Fmeasure_kNN[0], test_Fmeasure_kNN[1], test_Fmeasure_kNN[2], test_Fmeasure_kNN[3], test_Fmeasure_kNN[4], np.average(test_Fmeasure_kNN)], "F-measure for DT": [test_Fmeasure_DT[0], test_Fmeasure_DT[1], test_Fmeasure_DT[2], test_Fmeasure_DT[3], test_Fmeasure_DT[4], np.average(test_Fmeasure_DT)]}, index=[1, 2, 3, 4, 5, 'Average'])
bar_graph.plot.bar(rot = 0, title="Part I, 4 for F-Measure")
plt.show(block=True)
bar_graph = pd.DataFrame({"Test time for kNN":[test_time_kNN[0], test_time_kNN[1], test_time_kNN[2], test_time_kNN[3], test_time_kNN[4], np.average(test_time_kNN)], "Test Time for DT": [test_time_DT[0], test_time_DT[1], test_time_DT[2], test_time_DT[3], test_time_DT[4], np.average(test_time_DT)]}, index=[1, 2, 3, 4, 5, 'Average'])
bar_graph.plot.bar(rot = 0, title="Part I, 4 for Test Time")
plt.show(block=True)

#Part II (model selection or Comparing Different K):

validation_Fmeasure = np.zeros(5)
for k in range(1, 6):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    X_train, X_validate, Y_train, Y_validate = train_test_split(X_train, Y_train, test_size=0.1)
    clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', p=2)
    clf = clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predValidate = clf.predict(X_validate)
    validation_Fmeasure[i-1] = f1_score(Y_validate, Y_predValidate, average='weighted')
    

#data = {'Test 1': [test_Fmeasure[0], test_time[0]], 'Test 2': [test_Fmeasure[1], test_time[1]], 'Test 3': [test_Fmeasure[2], test_time[2]], 'Test 4': [test_Fmeasure[3], test_time[3]], 'Test 5': [test_Fmeasure[4], test_time[4]]}   
data=dict()
for i in range(5):
    data[f'k={i+1}'] = [validation_Fmeasure[i]]
table=pd.DataFrame(data, index=['F-measure'])
print("\nTable for Part II, 2 (COMPARING DIFFERENT K): \n, ", table)
best_k=np.argmax(validation_Fmeasure, axis=0)+1
print("Best k is: ", best_k)



#Part II 3 (comparing different maxDepth):
maxdepths = [3,4,5,6,7,8,9,10]
validation_Fmeasure2 = np.zeros(len(maxdepths))
index = 0
for depth in maxdepths:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(X_train, Y_train)
    Y_predTrain = clf.predict(X_train)
    Y_predValidate = clf.predict(X_validate)
    validation_Fmeasure2[i-1] = f1_score(Y_validate, Y_predValidate, average='weighted')
    index += 1

data=dict()
for i in range(len(maxdepths)):
    data[f'maxdepth={i+3}'] = [validation_Fmeasure2[i]]
table=pd.DataFrame(data, index=['F-measure'])
print("\nTable for Part II, 3 (COMPARING DIFFERENT maxdepths): \n, ", table)
best_maxdepth=np.argmax(validation_Fmeasure2, axis=0)+3
print("Best maxdepth: ", best_maxdepth)

#Part II 3b
clf = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=2)
clf = clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
Y_predValidate = clf.predict(X_validate)
validation_Fmeasure_bestkNN = f1_score(Y_validate, Y_predValidate, average='weighted')
print(validation_Fmeasure_bestkNN)

clf = tree.DecisionTreeClassifier(max_depth=best_maxdepth)
clf = clf.fit(X_train, Y_train)
Y_predTrain = clf.predict(X_train)
Y_predValidate = clf.predict(X_validate)
validation_Fmeasure2_bestDepth = f1_score(Y_validate, Y_predValidate, average='weighted')
print(validation_Fmeasure2_bestDepth)
df = pd.DataFrame({'Classification':['kNN', 'Decision tree'], 'comparison':[validation_Fmeasure_bestkNN, validation_Fmeasure2_bestDepth]})
ax = df.plot.bar(x='Classification', y='comparison', rot=0)
plt.show(block=True)