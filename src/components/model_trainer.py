import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_path, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
                "KNeighbors": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
            }
            
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    #'splitter': ['best', 'random'],
                    #'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    #'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    #'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    #'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    #'criterion': ['friedman_mse', 'squared_error'],
                    #'max_feature': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "KNeighbors": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    #'weights': ['uniform', 'distance'],
                    #'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                },
                "XGBoost": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                },
                "CatBoost": {
                    'depth': [6, 8, 10],
                    'learning_rate': [.1, .01, .05, .001],
                    'iterations': [30, 50, 100],
                },
                "AdaBoost": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [.1, .01, .05, .001],
                    #'loss': ['linear', 'square', 'exponential'],
                },
            }
            
            model_report: dict=evaluate_models(
                X_train=X_train,
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models,
                params=params,
            )
            
            # Get best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            # Get best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best model found!!!")
            
            save_path(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
        
            
        except Exception as e:
            raise CustomException(e, sys)