#Utlis.py file is created so that the module that are used again and again, such as save path, we dont have to create function afgain and again, we can define functions which has tp be used again and again in utlis.py 

import os
import sys
import pandas
import numpy
import dill
import pickle
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