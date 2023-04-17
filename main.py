import pickle
from flask import Flask,render_template,request
import numpy as np
#create an obect of the class flask
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    input_data=(4,110,92,0,0,37.6,0.191,30)
    input_data_as_numpy_array=np.asarray(request.form.getlist('diabetes[]'))
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=model.predict(input_data_reshaped)
    if(prediction[0]==1):
        output="Diabetic"
    else:
        output="Non Diabetic"
    return render_template('index.html', prediction_text=f'Prediction :-{output}')
if __name__=='__main__':
    app.run(debug=True)
