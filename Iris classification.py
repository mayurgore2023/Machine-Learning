# iris classification using Decission tree clasifier:


# import libraries:
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def MarvellousDecisionTreeClassifier():
     # 1 Load the data
    Dataset = load_iris()     

     # separatout depenant and independant variables:
    Data = Dataset.data
    print(Data)
    Target = Dataset.target
    
    # splitting of dataset:
    Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size = 0.5)
    
    # classifier:
    Classifier = DecisionTreeClassifier()
    
    # training:
    Classifier.fit(Data_train, Target_train)
    
    # testing:
    Predictions = Classifier.predict(Data_test)
    
    # accuracy:
    Accuracy = accuracy_score(Target_test, Predictions)

    return Accuracy

def main():
    Ret = MarvellousDecisionTreeClassifier()

    print("Accuracy of Iris dataset with DecisionTreeClassifier is ",Ret * 100)

if __name__ == "__main__":
    main()
