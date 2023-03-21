# Wine predictor case study using ensemble machine learning technique:

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans


def KNN_Classifier(A,B,C,D):
     Data_Train = A
     Data_test = B
     Target_train = C
     Target_test = D

     Classifier = KNeighborsClassifier()

     Training_Data_Set = Classifier.fit(Data_Train,Target_train)


     Testing_Data_Set = Classifier.predict(Data_test)

     Accuracy = accuracy_score(Target_test,Testing_Data_Set)
     return Accuracy




def DT_Classifier(A,B,C,D):

    Classifier = DecisionTreeClassifier()

    Training_Data_Set = Classifier.fit(A,C)

    Testing_Data_Set = Classifier.predict(B)

    Accuracy = accuracy_score(D, Testing_Data_Set)
    return Accuracy


def RF_Classifier(A,B,C,D):

    Classifier = RandomForestClassifier()

    Training_Data_Set = Classifier.fit(A,C)

    Testing_Data_Set = Classifier.predict(B)

    Accuracy = accuracy_score(D, Testing_Data_Set)
    return Accuracy


def KMeans_Classifier(A,B,C,D):

    Classifier = KMeans(n_clusters=5,n_init=24)

    Training_Data_Set = Classifier.fit(A,C)

    Testing_Data_Set = Classifier.predict(B)

    Accuracy = accuracy_score(D, Testing_Data_Set)
    return Accuracy


def main():
    print("________________________Application for Accuracy comparision__________________________")
    print()

    #load Data:
    Data = pd.read_csv("WinePredictor.csv")
    #print(Data)

    #separiting of Dependant & independant Variables:

    Wine_Class = Data.iloc[:,0].values
    #print(Wine_Class)

    Wine_contents = Data.iloc[:,1:14].values
    #print(Wine_contents)

    #Spliting of Data:
    Data_train,Data_test,Target_train,Target_test = train_test_split(Wine_contents,Wine_Class,test_size = 0.3 )

    Accuracy_by_KNN = KNN_Classifier(Data_train,Data_test,Target_train,Target_test)
    KNN = Accuracy_by_KNN * 100
    print("Accuracy using KNN is ",KNN)

    Accuracy_by_DTC = DT_Classifier(Data_train,Data_test,Target_train,Target_test)
    DTC = Accuracy_by_DTC * 100
    print("Accuracy using DTC is",DTC)


    Accuracy_by_RFC  = RF_Classifier(Data_train,Data_test,Target_train,Target_test)
    RFC = Accuracy_by_RFC * 100
    print("Accuracy using RFC is",RFC)


    Accuracy_by_Kmeans = KMeans_Classifier(Data_train,Data_test,Target_train,Target_test)
    KMeans = Accuracy_by_Kmeans * 100
    print("Accuracy using Kmeans is",KMeans)



    Accuracy_list =[]
    Accuracy_list.append(KNN )
    Accuracy_list.append(DTC)
    Accuracy_list.append(RFC)
    Accuracy_list.append(KMeans)
    print(Accuracy_list)

    X = np.array(['KNN','DTC','RFC','KMeans'])
    Y = np.array([Accuracy_list])
    Colour = ['r','y','g','b']

    plt.scatter(X,Y, c=Colour)
    plt.title("Accuracy of Model using Different Algorithms")
    plt.xlabel("<--- Name of Algorithms --->")
    plt.ylabel("<--- Accuracy (in %) --->")
    plt.show()




if __name__=="__main__":
    main()












