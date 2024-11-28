from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.Pipelines.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            custid=request.form.get('custid'),
            sname=request.form.get('sname'),
            creditscore=request.form.get('creditscore'),
            geography=request.form.get('geography'),
            gender=request.form.get('gender'),
            age=request.form.get('age'),
            tenure=request.form.get('tenure'),
            balance=request.form.get('balance'),
            products=request.form.get('products'),
            card=request.form.get('card'),
            active=request.form.get('active'),
            salary=request.form.get('salary'),
            complain=request.form.get('complain'),
            satscore=request.form.get('satscore'),
            cardtype=request.form.get('cardtype'),
            points=request.form.get('points')
            )
        
        pred_df=data.get_data_as_data_frame()

        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        if results[0] == 0:
            churn =  "The customer will not leave the bank"
        else:
            churn =  "The customer will leave the bank"
        print(results)
        print("after Prediction")
        return render_template('home.html',churn)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)        