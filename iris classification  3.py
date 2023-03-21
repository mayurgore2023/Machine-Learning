
# import libraries :
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier



def classifier():
    # load the dataset:
    Dataset = load_iris()


    # separate out dependent  and independent variable:
    data = Dataset.data
    print("Independent variables:",'\n',data)
    print("-------------------------------------------------------------------------------------")

    target = Dataset.target
    print('Dependant variables:','\n',target)
    print("-------------------------------------------------------------------------------------")


    # splitting of dataset  for  training and testing:
    data_train,data_test,target_train,target_test = train_test_split(data,target,test_size=0.3)

    # splitted data for training:
    print('Data_train:','\n',data_train)
    # splitted target for training:
    print('Target_train:','\n',target_train)


    # selectioon of classifier:
    model = DecisionTreeClassifier()

    # train the data:
    train = model.fit(data_train,target_train)

    # test the data:
    predict= model.predict(data_test)

    train_accuracy = accuracy_score(target_train,train.predict(data_train))
    print(train_accuracy)





def main():
    classifier()




if __name__=="__main__":
    main()

