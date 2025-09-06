from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

data = load_breast_cancer()
X = pd.DataFrame(data.data , columns = data.feature_names) 
y = pd.Series(data.target , name="target")

#splitting the data 
X_train , X_test , y_train , y_test = train_test_split(X , y)

#Creating the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

#Defining the parameters range
params_grid = {
    "n_estimators": [10, 50 , 100],
    "max_depth": [None , 3 , 8 ,15]
}

# Applying GridSearchCv
grid_search = GridSearchCV(estimator = rf , param_grid = params_grid , cv=5 , n_jobs=-1 , verbose=2)


################# Without mlflow use this code starting from here  #################################
#traing the model 
# grid_search.fit(X_train , y_train)

# #Displaying the best parameters

# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print(f"Best parameters for rf achieved after : {best_params}\nBest score achieved on these parameters is : {best_score}")
# ################# Without mlflow use this code ending here  ###########################################

mlflow.set_experiment("breast-cancer-rf-hp")
with mlflow.start_run():
    grid_search.fit(X_train , y_train)
    
    
    #Display best parameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    #log params
    mlflow.log_params(best_params)
    
    #log accuracy
    mlflow.log_metric("Accuracy" , best_score)
    
    #Log training Data 
    train_df = X_train.copy()
    train_df["target"] = y_train
    
    #in order to log the training data we need to convert it into an mlflow object first
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df , "training")
    
    #log test data
    test_df = X_test.copy()
    test_df["target"] = y_test
    
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df , "testing")
    
    
    #log the source file
    mlflow.log_artifact(__file__)
    
    #log the model
    mlflow.sklearn.log_model(grid_search.best_estimator_ ,"RandomForest")
    
    #set logs
    
    mlflow.set_tag("Author", "Arosha")
    
    print(best_params)
    print(best_score)