#definitions to be used

def ReRun():
    while True:
        input("Press any key to continue...")
        break

def Model_Results():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt 

    #read in boston data
    boston_data = load_boston()

    #set to dataframe and pull from target the medv price
    boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
    boston['MEDV'] = boston_data.target

    #setting up model
    X = pd.DataFrame(np.c_[boston["CRIM"], boston["ZN"], boston["NOX"], boston['RM'],boston["AGE"],boston["DIS"],boston["TAX"],boston["PTRATIO"],boston['LSTAT'],], columns = ["Crime","Residential","NOX","Rooms","AGE","Distance","TAX","Teachers","Population"])
    Y = boston['MEDV']

    #test train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=5)

    #model
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    results = lin_model.coef_
    titles = ["CRIM","ZN","NOX","RM","AGE","DIS","TAX","PTRATIO","LSTAT"]

    #dict
    res = {titles[i]: results[i] for i in range(len(results))}

    #abs max in dict
    maxkey = max(res, key=lambda y: abs(res[y]))

    print("------------------------")
    print("Coefficients:", res)
    print("The most impactful varible is ", maxkey)
    print("------------------------")


def KMean_Results():
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn import metrics
    from sklearn import datasets
    import matplotlib.pyplot as plt

    #read in IRIS data
    iris_data = datasets.load_iris()
    #make to df
    iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    #add in target
    group = iris_data.target

    rand_index = []
    Mutual = []
    elbow =[]
    for i in range(1,6):
        kmean = KMeans(n_clusters= i).fit_predict(iris)
        rand_index.append(metrics.rand_score(group, kmean))
        Mutual.append(metrics.adjusted_mutual_info_score(group,kmean))
        kmeans = KMeans(n_clusters= i).fit(iris)
        elbow.append(kmeans.fit(iris).inertia_)
    li = [rand_index, Mutual, elbow]

    #Model Evaluation
    #Rand index is a function that measures the similarity of the two assignments
    #Mutual Information is a function that measures the agreement of the two assignments
    #both Random Index and Mutual Information are methods to evaluate your K-Mean Model
    #the Elbow method can be used to identify the groupings

    print("------------------------")
    print("The Random Index and Mutual Information are methods to evaluate your model, 0 is bad while 1 is good. The elbow method helps identify how many cluster you should use by looking just before the values level off.")
    
    results = pd.DataFrame(data = li, columns= ["c1", "c2", "c3", "c4", "c5"])
    results.insert(0,"names",["rand_index","Mutual", "elbow"])
    print(results)

    print("Here we can see the different cluster tested. c3 is the max for both rand_index and Mutual. c3 is also just before the elbow method levels off.")
    print("------------------------")
