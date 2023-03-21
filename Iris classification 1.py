# import libraries:

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#  Load the data
def MarvellousKNeighborsClassifier():
    Dataset = load_iris()       

 #  separate out dependent and independent functions:
    Data = Dataset.data
    Target = Dataset.target

 # splitting of dataset:
    Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size = 0.5)

# choose classifier:
    Classifier = KNeighborsClassifier()

# training of dataset:
    Classifier.fit(Data_train, Target_train)

# testing of data:
    Predictions = Classifier.predict(Data_test)

# finding out accuracy of model:
    Accuracy = accuracy_score(Target_test, Predictions)

    return Accuracy

def main():
    Ret = MarvellousKNeighborsClassifier()

    print("Acuracy of Iris dataset with KNN is ",Ret * 100)

if __name__ == "__main__":
    main()
