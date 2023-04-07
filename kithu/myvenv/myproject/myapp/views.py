from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Create your views here.

def index(request):

    return render(request,'index.html')


 
def result(request):

    data=pd.read_csv(r"C:\Users\Sangeeth\Downloads\my project data.csv")
    x=data.drop(["IT_organization"],axis=1)
    le_data=LabelEncoder()
    x["course_title"]=le_data.fit_transform(x["course_title"])
    x["syllabus_range"]=le_data.fit_transform(x["syllabus_range"])
    x["certifications"]=le_data.fit_transform(x["certifications"])

    y=data["IT_organization"]

    scaler=MinMaxScaler(feature_range=(0,1))
    x=scaler.fit_transform(x)

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.15)

    model=RandomForestClassifier()
    model.fit(xtrain,ytrain)
    model.score(xtest,ytest)

    joblib.dump(model,'joblib_model')
    new_model=joblib.load('joblib_model')
    # v1=0
    # v2=4.8
    # v3=3
    # v4=40000
    # v5=5
    # v6=0
    v1 = request.GET['Course_title']
    v2 = float(request.GET['online_rating'])
    v3 = request.GET['Syllabus_range']
    v4 = float(request.GET['Course_fee'])
    v5 = float(request.GET['Course_duration'])
    v6 = request.GET['Certifications']

    print("v1", v1)
    print("v2", v2)
    print("v3", v3)
    print("v4", v4)
    print("v5", v5)
    print()

    x_values1=[v1,v3,v6] 
    x_values1=le_data.fit_transform(x_values1)
    x_values2=np.insert(x_values1,1,v2)
    x_values3=np.insert(x_values2,3,v4)
    x_values4=np.insert(x_values3,4,v5)
    print("x_val", x_values4)
    pred = model.predict([x_values4])
    result =pred[0]
    return render(request,'index.html',{'result':result})