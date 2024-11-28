import os
import sys
import pandas as pd
import numpy as np
import scipy.stats as scis
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object, evaluate_model
from src.Components.model_trainer import ModelTrainer

from src.Components.data_ingestion import DataIngestion 

from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline, FunctionTransformer
from feature_engine.transformation import BoxCoxTransformer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
            self.preprocessor_path = DataTransformationConfig()
    
    def TransformerObject(self):
            try:
                def balance_transform(X):
                    """Convert balance to categories."""
                    balance_category = X['Balance'].apply(lambda x: 'Bal_Zero' if x == 0 else ('Bal_low_0_50k' if 0 < x <= 50000 else 'Bal_high_50k+')).rename('Balance_Category')
                    return pd.DataFrame(data = balance_category, columns=['Balance_Category'])

                def satisfaction_score_transform(X):
                    """ Convert Satisfaction Score into object column"""
                    return pd.DataFrame(data = X['Satisfaction Score'].astype('object'))

                def drop_columns(X):
                    """Drops specified columns from a DataFrame."""
                    columns_to_drop = ['CustomerId', 'Complain', 'Surname']  # Specify the columns to drop
                    return X.drop(columns=columns_to_drop)

                age_pipeline = Pipeline(steps=[
                                ('boxcox', BoxCoxTransformer()),
                                ('standsca', StandardScaler())
                                ])
                balance_pipeline = Pipeline(steps=[
                                ('bal_category', FunctionTransformer(balance_transform, validate=False)),
                                ('onehot', OneHotEncoder(sparse_output=False))
                                ])
                satisfaction_pipeline = Pipeline(steps=[
                                ('satisfaction_transform', FunctionTransformer(satisfaction_score_transform, validate=False)),
                                ('onehot', OneHotEncoder(sparse_output=False))
                                ])
                cat_pipeline = Pipeline(steps=[
                                ('onehot', OneHotEncoder(sparse_output=False))
                                ])

                num_norm_pipeline = Pipeline(steps=[
                                ('stan_scaler', StandardScaler())
                                ])

                num_uni_pipeline = Pipeline(steps=[
                                ('minmax_scaler', MinMaxScaler())
                                ])

                # Step 3: Combine transformations using ColumnTransformer
                preprocessor = ColumnTransformer(
                                transformers=[
                                    ('age', age_pipeline, ['Age']),
                                    ('balance', balance_pipeline, ['Balance']),
                                    ('satisfaction', satisfaction_pipeline, ['Satisfaction Score']),
                                    ('categorical', cat_pipeline, ['Geography', 'Gender', 'Card Type']),
                                    ('numerical_uni', num_uni_pipeline, ['Tenure', 'EstimatedSalary', 'Point Earned']),
                                    ('numerical_norm', num_norm_pipeline, ['CreditScore', 'NumOfProducts'])
                                ], verbose_feature_names_out=False, remainder= "passthrough")
                preprocessor.set_output(transform='pandas')

                # Step 4: Create the full pipeline including column dropping
                full_pipeline = Pipeline(steps=[
                                    ('preprocessing', preprocessor),            # Apply transformations
                                    ('drop_columns', FunctionTransformer(drop_columns, validate=False))  # Drop original columns
                                ])
                
                return full_pipeline
            except Exception as e:
                  raise CustomException(e, sys)
    
    def InitiateDataTransformation(self, train_path, test_path):
            try:
                  train_df = pd.read_csv(train_path)
                  test_df = pd.read_csv(test_path)
                  
                  target_column = 'Exited'

                  logging.info("Read train and test data completed")
                  logging.info("instantiating preprocessing object")

                  PreprocessorObj = self.TransformerObject()

                  input_feature_train = train_df.drop(columns=target_column, axis=1)
                  target_feature_train = train_df[target_column]

                  input_feature_test = test_df.drop(columns=target_column, axis=1)
                  target_feature_test = test_df[target_column]
                  
                  logging.info("Applying Preprocessing object on train and test data frames Start")
                  # 1. Apply preprocessing object to the train and test data sets
                  input_feature_preprocess_train = PreprocessorObj.fit_transform(input_feature_train)
                  input_feature_preprocess_test = PreprocessorObj.transform(input_feature_test)
                  logging.info("Applying Preprocessing object on train and test data frames End")
                  
                  logging.info("SMOTE on train data initiation")
                  # 2. Applying SMOTE on training data
                  smote = SMOTE(random_state=42)
                  X_train_smote, y_train_smote = smote.fit_resample(input_feature_preprocess_train, target_feature_train)
                  logging.info("SMOTE on train data End")
                  
                  # 3. Combine the preprocessed train and test data sets with the target column
                  train_preprocess = pd.concat([X_train_smote, pd.DataFrame(y_train_smote)], axis=1)
                  test_preprocess = pd.concat([input_feature_preprocess_test, pd.DataFrame(target_feature_test)], axis=1)
                  
                  
                  logging.info("Saving preprocessing object")
                  save_object(
                       file_path=self.preprocessor_path.preprocessor_obj_path,
                       obj= PreprocessorObj
                  )
                  logging.info("Saved preprocessing object")
                  return (
                    train_preprocess,
                    test_preprocess,
                    self.preprocessor_path.preprocessor_obj_path
                  )

                  
            except Exception as e:
                raise CustomException(e, sys)
            

if __name__ == '__main__':
    prep =  DataTransformation()
    di = DataIngestion()
    train_data_path, test_data_path = di.initiate_data_ingestion()
    train_preprocess, test_preprocess, preprocessor_path = prep.InitiateDataTransformation(train_data_path, test_data_path)
    # print(train_preprocess.shape, test_preprocess.shape)
    # print(train_preprocess.columns, test_preprocess.columns)
    # print(train_preprocess.describe(), test_preprocess.describe())
    model = ModelTrainer()
    model.initiate_model_training(train_preprocess, test_preprocess)