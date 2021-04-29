#SKLearn
#Jake Vick
from SKLearnFunctions import Model_Results, ReRun, KMean_Results

#main loop
while True:
    user_input = input("Press 1 to run the Linear Regression\nPress 2 to run the K-Mean\nPress 0 to exit\nSelection: ")
    # Loop until users opts to go again or quit

    try:
        user_input = int(user_input)  # non-numeric input from user could otherwise crash at this point
        if user_input == 0:
            break # break out of this while loop
        elif user_input == 1:
            print("Linear Regression Results:\nBelow are the coefficients in the model. The sign of each coefficient indicates the direction of the relationship between a predictor variable and the response variable.")
            Model_Results()
            ReRun()
        elif user_input == 2:
            print("K-Mean Clustering Results:")
            KMean_Results()
            ReRun()
        else:
            print("Not in number range please try again.")
    except:
        print("Non number entered")