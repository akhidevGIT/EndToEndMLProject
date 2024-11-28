import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        
        custid: int,
        sname: str,
        creditscore: int,
        geography: str,
        gender:str,
        age: int,
        tenure: int,
        balance: int,
        products:int,
        card:int,
        active:int,
        salary:int,
        complain: int,
        satscore:int,
        cardtype:str,
        points:int
        ):
        
        self.custid = custid
        self.sname = sname
        self.creditscore = creditscore
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.products = products
        self.card = card
        self.active = active
        self.salary = salary
        self.complain = complain
        self.satscore = satscore
        self.cardtype = cardtype
        self.points = points
        
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                
                'CustomerId': [self.custid],
                'Surname': [self.sname],
                'CreditScore': [self.creditscore],
                'Geography': [self.geography],
                'Gender': [self.gender],
                'Age': [self.age],
                'Tenure': [self.tenure],
                'Balance': [self.balance],
                'NumOfProducts': [self.products],
                'HasCrCard': [self.card],
                'IsActiveMember': [self.active],
                'EstimatedSalary': [self.salary],
                'Complain': [self.complain],
                'Satisfaction Score': [self.satscore],
                'Card Type': [self.cardtype],
                'Point Earned': [self.points]



                
                
                
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        


# if __name__ == '__main__':
#     pred = PredictPipeline()
#     cust_data = CustomData(
#         rownum= 1,
#         custid= '12345',
#         sname= 'Smith',
#         creditscore= 650,
#         geography= 'France',
#         gender= 'Male',
#         age= 45,
#         tenure= 3,
#         balance= 60000,
#         products= 2,
#         card= 1,
#         active= 1,
#         salary= 50000,
#         complain= 0,
#         satscore= 5,
#         cardtype= 'GOLD',
#         points= 323
#         )
    
#     df = cust_data.get_data_as_data_frame()
#     print(pred.predict(df))
# else:
#     print('This script is not meant to be run directly. Use the PredictPipeline class instead.')