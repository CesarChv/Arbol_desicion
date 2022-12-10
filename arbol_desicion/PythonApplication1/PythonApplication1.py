##load dataset (student Portuguese scores)
#import pandas as pd
#d = pd.read_csv('student-por.csv', sep=';')
#len(d)

##generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum >=30

#d['pass'] = d.apply(lambda row: 1 if (row['G1']+ row['G2']+ row['G3']) >= 35 else 0 , axis=1)
#d = d.drop(['G1','G2','G3'],axis=1)
#d.head


##user one-hot encoding on categorical columns
#d = pd.get_dummies(d,columns = [ 'sex','school','address','famsize','Pstatus','Mjob','Fjob',
#								'reason','guardian','schoolsup','famsup','paid','activities',
#								'nursery','higher','internet','romantic'])

#d.head()
##shuffle rows 

#d= d.sample(frac=1)
## split training and testing data

#d_train = d[:500]
#d_test = d[500:]

#d_train_att = d_train.drop(['pass'],axis=1)
#d_train_pass  = d_train['pass']

#d_test_att = d_test.drop(['pass'],axis=1)
#d_test_pass = d_test['pass']

#d_att = d.drop(['pass'],axis=1)
#d_pass = d['pass']

##number of passing students in whole dataset:
#import numpy as np
#print("Passing: %d out of %d (%.2f%%)" % (np.sum(d_pass),len(d_pass),100*float(np.sum(d_pass))/len(d_pass)))

##fit a decision tree
#from sklearn import tree
#t = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
#t = t.fit(d_train_att,d_train_pass)
#tree.export_graphviz(t,out_file="student-performance.dot", label="all",impurity = False,
#	proportion = True, feature_names = list(d_train_att),class_names=["fail","pass"],
#	filled = True, rounded=True)
#t.score(d_test_att, d_test_pass)

#from sklearn.model_selection import cross_val_score
#scores = cross_val_score(t,d_att,d_pass,cv=5)
##show average score and +/- two standard deviations away (convering 95% of scores)
#print("Accuacy: %0.2f (+/- %0.2f)" % (scores.mean(),scores.std() * 2))

#for max_depth in range(1,30):
#	t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
#	scores = cross_val_score(t,d_att,d_pass, cv=5)
#	print("Max depth: %d, Accuacy: %0.2f (+/- %0.2f)" % (max_depth,scores.mean(),scores.std() * 2))

#	depth_acc = np.empty((19,3), float)
#	i=0
#	for max_depth in range(1,20):
#		t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
#		scores = cross_val_score(t,d_att,d_pass,cv=5)
#		depth_acc[i,0] = max_depth
#		depth_acc[i,1] = scores.mean()
#		depth_acc[i,2] = scores.std() * 2
#		i+=1
#	depth_acc


##obj=[[1,0,0,1,17,1,0,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,1,2,0,0,1,1,0,0,1,0,1,0,1,1,0,1,0,0,1,5,3,3,1,1,3,4]]
##obj_p = t.predict(obj)
##print(obj_p)

##plt.figure('Arbol')
##tree.plot_tree(t, filled=True)
##plt.show()


import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split

'''
*Cesar Chave Zamorano
*Sistemas Expertos
*Programa desarrollado
*ICO: Ingenieria en computacion
'''

db_iris = load_iris()

data_train, data_test, c_train, c_test = train_test_split(db_iris.data, db_iris.target)

#arbol
t = tree.DecisionTreeClassifier()

t = t.fit(data_train, c_train)

score = t.score(data_test, c_test)
print ('Score - test:', score)

score - t.score(data_train, c_train)
print ('Score - entrenamiento:', score)

#tree.export_graphviz(t, out_file="arbol_iris.dot",
#                     feature_names = db_iris.feature_names,
#                     class_names = db_iris.target_names, filles = True)


plt.figure('Arbol')
tree.plot_tree(t, filled = True)
plt.show()

presc = cross_val_score(t, db_iris.data, db_iris.target, cv = 3)
print("Precision: %0.2f (+/- %0.2f)"%(presc.mean(),presc.std()*2))



obj=[[4.6,3.5,1.4,0.3]]
obj_p = t.predict(obj)
print (obj_p, '=', db_iris.target_names[obj_p])

obj=[[7.6,2.5,6.4,2.3]]
obj_p = t.predict(obj)
print (obj_p, '=', db_iris.target_names[obj_p])


obj=[[5.6,2.3,4.4,1.3]]
obj_p = t.predict(obj)
print (obj_p, '=', db_iris.target_names[obj_p])

obj=[[5.6,2.5,3.8,1.1]]
obj_p = t.predict(obj)
print (obj_p, '=', db_iris.target_names[obj_p])


obj=[[4.1,3.9,1.9,0.3]]
obj_p = t.predict(obj)
print (obj_p, '=', db_iris.target_names[obj_p])




#obj=[[1,0,0,1,1,1,0,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,1,2,0,0,1,1,0,0,1,0,1,0,1,1,0,1,0,0,1,5,3,3,1,1,3,4]]
#obj_p = t.predict(obj)
#print(obj_p)