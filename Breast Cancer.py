from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics



def SVM():
    # Load the dataset
    Dataset = datasets.load_breast_cancer()

    # Print name of the features
    print('Features of cancer data set:',Dataset.feature_names)
    print('--------------------------------------------------------')

    # print the label/type of Cancer
    print('Labels of cancer dataset : ',Dataset.target_names)
    print('----------------------------------------------------------')

    # first five records
    print('First 5 records are')
    print(Dataset.data[0:5])

    # split dataset into training set and test set
    x_train,x_test,y_train,y_test =train_test_split(Dataset.data,Dataset.target,test_size = 0.3,random_state=109)


    # create SVM classifier:
    clf = svm.SVC(kernel ='linear')


    # train the model using training sets:
    clf.fit(x_train,y_train)

    # predict the response for test dataset
    y_predict = clf.predict(x_test)

    # Model accuracy : how  often model is correct
    print('Accuracy of model is :',metrics.accuracy_score(y_test,y_predict)*100)














def main():
    print("------------Application of Support Vector Machine------------------")

    SVM()

if __name__=="__main__":
    main()