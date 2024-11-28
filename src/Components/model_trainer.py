from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.utils import evaluate_model, save_object
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train, test):
        try:
            logging.info("Splitting input and target features from train and test arrays")
            X_train, X_test, y_train, y_test = (
                train.drop(columns = 'Exited'),
                test.drop(columns = 'Exited'),
                train['Exited'],
                test['Exited']
                )
            models = {
                "Logistic Regression":LogisticRegression(),
                "Random Forest":RandomForestClassifier(),
                "Support Vector Machines":SVC(),
                "XGBoost Classifier": XGBClassifier()
                }
            params={
                "logistic_regression":{},
                "Random Forest":{
                    'n_estimators': [100, 200, 300],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [10, 20, 30]
                    },
                "svc":{
                    'C': [0.1, 1, 10],
                    'kernel': ['poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto']

                    },
                "xgboost":{
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.3]
                    }
                }
            logging.info("Model Evaluation Started")
            model_report:dict = evaluate_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models = models,
                params = params
                )
            logging.info("Model Evaluation Completed")
            logging.info("Best Model Selected")

             #To get best model score from dict
            best_model_score = max(model_report.values())
        
            #To get best name from dict
            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = best_model
            )

            print(f"best_model_score:{best_model_score}")
            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)

