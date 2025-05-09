import os
import sys
from dataclasses import dataclass


from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor



from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
           
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                
            }
            params = {
               
                "Random Forest": {
                   'n_estimators': [50, 100],  # Avoid going too high early on
                   'max_depth': [5, 10, 15],   # Force trees to be shallower
                   'min_samples_split': [5, 10],  # Prevent deep splitting
                   'min_samples_leaf': [2, 4, 6],  # Force more generalization at leaves
                   'max_features': ['sqrt', 0.5],  # Try fewer features per split
                   'bootstrap': [True]
                     },
                
            
                
                
        }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

           
            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            print("Best model found is : ",best_model_name)

            r2_square = r2_score(y_test, predicted)
            mae=mean_absolute_error(y_test,predicted)
            mse=mean_squared_error(y_test,predicted)
            print("R2 score of best model is : ",r2_square)
            print("Mean Absolute Error of best model is : ",mae)
            print("Mean Squared Error of best model is : ",mse)
          
           
        
            return r2_square,mae,mse
            



            
        except Exception as e:
            raise e