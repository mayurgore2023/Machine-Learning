# Import Libraries
import pandas as pd
import numpy as np

#('Import Data :

print('Import Data :')
Dataset = pd.read_csv("Data.csv")
print(Dataset)
print("_---------------------------------------------------------------------_")

Unique = Dataset['Name'].unique()
print(Unique)
print("_---------------------------------------------------------------------_")


Count = Dataset['Insurance'].value_counts()
print(Count)
print("_---------------------------------------------------------------------_")

from sklearn.preprocessing import LabelEncoder

Encoder = LabelEncoder()

# Encoding of Independant_Variable
Dataset['Name'] = Encoder.fit_transform(Dataset['Name'])

# Encoding of Dependant_Variable
Dataset['Insurance'] = Encoder.fit_transform(Dataset['Insurance'])


New_Dataset  = Dataset

#separating of Dependant And independant Variables:

X = New_Dataset[['Name','Age','Salary']].values
print(X)

Y = New_Dataset['Insurance'].values
print(Y)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

Classifier = RandomForestClassifier()

train = Classifier.fit(X,Y)

Prediction = Classifier.predict([[6,42,36000]])
print(Prediction)






