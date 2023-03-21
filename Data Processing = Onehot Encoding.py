import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# loading of csv file:
Data = pd.read_csv('OnehotEncoding.csv')
print(Data)
print('--------------------------------------------------------------')

print(Data.dtypes)
print('--------------------------------------------------------------')

Country_Name = Data['Country'].unique()
print(Country_Name)
print()

Country_Count = Data['Country'].value_counts()
print(Country_Count)
print('--------------------------------------------------------------')


Colour_Name = Data['Colour'].unique()
print(Colour_Name)
print()

Colour_Count = Data['Colour'].value_counts()
print(Colour_Count)

print('--------------------------------------------------------------')


# Using Onehot encoder:

Encoder = OneHotEncoder()

print('making araay of fetures with encoding')
Features_array = Encoder.fit_transform(Data[['Country','Colour']]).toarray() # .toarray() make array
print(Features_array)
print()

Feature_categories =  Encoder.categories_
print(Feature_categories)

print('--------------------------------------------------------------')
# categories saved in differant are we save it in singal array by using numpy

Feature_labels = np.array(Feature_categories).ravel() # .ravel() makes single array
print(Feature_labels)
print('---------------------------------------------------------------')

# makinng of Datafram of all encoded features

Dataframe = pd.DataFrame(Features_array,columns= Feature_labels)
print(Dataframe)
print("----------------------------------------------------------------")

# combine the Dataframe with original Data :

Datframe_And_Data = pd.concat([Data,Dataframe],axis=1)  # .concat = Join to data Frames \ axis = to concat along.
print(Datframe_And_Data)





