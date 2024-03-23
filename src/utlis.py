#Utlis.py file is created so that the module that are used again and again, such as save path, we dont have to create function afgain and again, we can define functions which has tp be used again and again in utlis.py 

import os
import sys
import pandas
import numpy
import dill
import pickle
from sklearn.metrics import r2_score
from src.exception import CustomException
#defining a funciton called save_object to save the preprocessor as pickle fath at speicifed path
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(file_path,"wb"  ) as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report =  {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred =  model.predict(X_test)
            #calculating the r2_score oor the train and test data for  the model
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
                #appending in the report
            report[list(models.keys())[i]]= test_model_score
            return report 
    except Exception as e:
        raise CustomException(e,sys)



    