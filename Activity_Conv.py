
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import pdb as pdb

act=pd.read_csv("masked_asthma_data.csv",low_memory=False)
act['Encounter_Type'].replace([1,3,5,7,8,9,12,13,14],0,inplace=True)
act['Encounter_Type'].replace([2,4,6,41,42],1,inplace=True)

act_unique=act.groupby(['PatientID'])['StartDate'].nunique().reset_index()

act_unique['No_visit']=act_unique['StartDate']
act_unique.drop(['StartDate'], axis=1,inplace=True)
act_unique.to_csv('act_unique.csv')
final_act=pd.merge(act, act_unique, on='PatientID', how='inner')
#---convert start date into date formate--
final_act.to_csv('final_asthma_file.csv')
#pdb.set_trace()
final_act['StartDate'] = pd.to_datetime(final_act['StartDate'])


#--define winter seasion
start_date = '10-01-2018'
end_date = '12-31-2018'

final_act.loc[(final_act['StartDate'] >= start_date) & (final_act['StartDate'] <= end_date), 'Winter_Flag'] = 1

final_act.loc[final_act['StartDate'] < start_date, 'Winter_Flag'] = 0

#final_file= final_act.to_csv('final_asthma_file.csv')

final_variable= pd.DataFrame(final_act,columns=['Activity_Type','Encounter_Type','PatientShare','No_visit','Winter_Flag'])
#pdb.set_trace()
final_variable.drop_duplicates()

#----------modeling---
feature_cols=['Activity_Type','PatientShare','No_visit','Winter_Flag']
X = final_variable[feature_cols]

y = final_variable.Encounter_Type # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)



#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#------------------end--------------------------------------
#--------------------graph------------------------------------------

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('fullmodel.png')
Image(graph.create_png())

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('simplemodel.png')
Image(graph.create_png())

