import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def Head_brain():

    # load the dataset:
    dataset = pd.read_csv('MarvellousHeadBrain.csv')
    print(dataset)
    print('------------------------------------------------------------------------------------')

    print('shape of dataset is',dataset.shape)
    print('------------------------------------------------------------------------------------')

    # separate dependant and independent variables
    x =  dataset['Brain Weight(grams)'].values
    print(x)
    print('-------------------------------------------------------------------')

    X = x.reshape(-1,1) # we need  reshape when single feature
    print(X)


    print('------------------------------------------------------------------------------------')

    Y= dataset['Brain Weight(grams)'].values
    print(Y)
    print('---------------------------------------------------------------------------')
    reg = LinearRegression()

    training = reg.fit(X,Y)
    Y_predict = reg.predict(X)

    print('-----------------------------------------------------------')

    r2 = reg.score(X,Y)
    print(r2 *100)









def main():

    print("---------Application of HeadBrain case study----------------")

    Head_brain()

if __name__=='__main__':
    main()