import csv

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

X = []
y = []

with open('data_minging\Absenteeism_at_work.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)

    for line in csv_reader:
        lst = line[9].split(',')
        line[9] = int(lst[0]) * 1000 + int(lst[1])
        absent = int(line[-1])
        line = line[1:-1]
        if absent > 10:
            absent = 1
        else:
            absent = 0
        X.append(line)
        y.append(absent)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf1 = DecisionTreeClassifier()
clf1 = clf1.fit(X_train,y_train)
y_pred1 = clf1.predict(X_test)
print("Decision tree Accuracy:",metrics.accuracy_score(y_test, y_pred1))


from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=50)
clf2 = clf2.fit(X_train,y_train)
y_pred2 = clf2.predict(X_test)
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred2))

clf3 = RandomForestClassifier(n_estimators=100)
clf3 = clf3.fit(X_train,y_train)
y_pred3 = clf3.predict(X_test)
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred3))
# from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO
# from IPython.display import Image
# import pydotplus
#
# features = 'Reason for absence,Month of absence,Day of the week,Seasons,Transportation expense,Distance from Residence to Work,Service time,Age,Work load Average/day,Hit target,Disciplinary failure,Education,Son,Social drinker,Social smoker,Pet,Weight,Height,Body mass index'
# feature_cols = features.split(',')
# dot_data = StringIO()
# export_graphviz(clf, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = feature_cols,class_names=['0','1'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('diabetes.png')
# Image(graph.create_png())
