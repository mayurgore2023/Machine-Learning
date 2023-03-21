
# pay predictor using Logestic regression:

# load the data set:
import pandas as pd

Dataset =pd.read_csv('PlayPredictor.csv',index_col=0)
print("Dataset:")
print(Dataset)
print("-----------------------------------------------------------------------")

# Find out unique labels in dara type:
print('Unique labels:')
print(Dataset['Play'].unique())    # there are two unique labels 'Yes' and 'No'
print("-----------------------------------------------------------------------")

# check there is any missing value in dataset or not:
print("Null values:")
print(Dataset.isnull().sum())

print("-----------------------------------------------------------------------")

# Data in Dataset  is not in numaric format so we have to encode it:
from sklearn.preprocessing  import LabelEncoder

encoder = LabelEncoder()

Dataset['Whether']= encoder.fit_transform(Dataset['Whether'])
Dataset['Temperature']= encoder.fit_transform(Dataset['Temperature'])
Dataset['Play']= encoder.fit_transform(Dataset['Play'])

print("Dataset after encoding:",'\n',Dataset)
print("-----------------------------------------------------------------------")


# separate out dependant and independant variables:

X_ind = Dataset.iloc[:,0:2].values
print(X_ind)


Y_dep = Dataset.iloc[:,2].values
print(Y_dep)

print("-----------------------------------------------------------------------")


# splitting the data for training_set  and testing_set:
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X_ind,Y_dep,test_size = 0.2)

print("-----------------------------------------------------------------------")

# select the classifier for model:
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
print("-----------------------------------------------------------------------")

# if we have to tune the parameters:
from sklearn.model_selection import GridSearchCV
parameters ={'penalty':['l1','l2','elasticnet'],'C':[1.0,2,3,4,5,6,10,20,30,40,50],'max_iter':[100,200,300]} # parameters to tuned

classifier_regression = GridSearchCV(classifier,param_grid =parameters,scoring='accuracy',cv=5)
 # estimator = model we want to select , param_grid = parameters we want to tuned, scoring = training accuracy , cv = no of times it intaranally divdes training dataset

# train the model by tuning different parameters:
training = classifier_regression.fit(X_train,Y_train)
print(training)
print("-----------------------------------------------------------------------")

# check which parameter combination gives best  result:

print(classifier_regression.best_params_)

print("-----------------------------------------------------------------------")

# check how much accuracy iss given by selected best parameters:
print(classifier_regression.best_score_)

print("-----------------------------------------------------------------------")


# testing of model:
predict= classifier_regression.predict(X_test)
print(predict)
print("-----------------------------------------------------------------------")

# determine final mmodel accuracy:
from sklearn.metrics import accuracy_score,classification_report

accuracy = accuracy_score(predict,Y_test)
print(accuracy)
print("-----------------------------------------------------------------------")

# To get precision ,recall and F1 score
report = classification_report(predict,Y_test)
print(report)

print("-----------------------------------------------------------------------")

print(' Confusion Matrix:')

Matrix = confusion_matrix(Y_test,predict)
print(Matrix)
