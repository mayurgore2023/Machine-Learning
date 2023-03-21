# play predictor case study using KNN classifier:

import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier


def MarvellousPrepredictor():

    print("Load Dataset:")

    Data = pd.read_csv("PlayPredictor.csv")
    print(Data)
    print("----------------------------------------------------------------")

    print("Clean, Prepare and Mnipulate the data :")

    Feature_names = ['Whether', 'Temperature']
    print(Feature_names)
    Whether = Data.Whether
    Temperature = Data.Temperature
    Play = Data.Play

    print("----------------------------------------------------------------")


    print("Encoding the Dataset:")

    le = preprocessing.LabelEncoder()

    print("Wheather Encoding Dataset :")

    Weather_encoded = le.fit_transform(Whether)
    print(Weather_encoded)
    print()

    print("Temperature Encoding Dataset :")


    Temperature_encoded = le.fit_transform(Temperature)
    print(Temperature_encoded)
    print()

    print("Label Encoding Dataset :")

    Label_encoded = le.fit_transform(Play)
    print(Label_encoded)
    print()

    print("Combining  of Encoded Features Dataset:")

    Features = list(zip(Weather_encoded,Temperature_encoded))
    print(Features)
    print()
    print("----------------------------------------------------------------")



    Model = KNeighborsClassifier(n_neighbors=3)

    Training = Model.fit(Features,Label_encoded)

    Testing_Result = Model.predict([[0,1]])
    print("Prediction result:")
    if Testing_Result == 0:
        print("Not play the game")
    else:
        if Testing_Result == 1 :
            print("Play the game")






def main():
    print("---------------------------------------------------")
    print("ML operation on PlayPredictor Dataset case study")
    print("---------------------------------------------------")
    print()

    MarvellousPrepredictor()



if __name__=="__main__":
    main()
