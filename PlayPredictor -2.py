import pandas as pd


#load the csv file

Dataset = pd.read_csv('PlayPredictor.csv')
print(Dataset)

# Encoding of data:
from sklearn.preprocessing  import LabelEncoder

# create object of Encoder:
encoder = LabelEncoder()

Dataset['Whether']= encoder.fit_transform(Dataset['Whether'])
Dataset['Temperature']= encoder.fit_transform(Dataset['Temperature'])
Dataset['Play']= encoder.fit_transform(Dataset['Play'])
print(Dataset)


New_Dataset = Dataset

# separting depeendent and independant variables:

X = New_Dataset[['Whether','Temperature']]
print(X)

Y = New_Dataset['Play']
print(Y)

# splitting of data into training and testing dataset:
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

# Deciding classifier:
from sklearn.tree import DecisionTreeClassifier

# credatie object of classifier:
classifier = DecisionTreeClassifier()

# train the data:
classifier.fit(X_train,Y_train)

# test the data
predict = classifier.predict(X_test)
print()

# calculate acuuracy of model
from  sklearn.metrics import accuracy_score

Accuracy = accuracy_score(Y_test,predict)
print(Accuracy*100)


from sklearn import tree
plot = tree.plot_tree(classifier)
print(plot)