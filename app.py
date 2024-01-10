from flask import Flask, render_template, request
import numpy as np
import os
from keras.models import load_model
from sklearn.preprocessing import  StandardScaler, LabelEncoder
import pickle

app = Flask(__name__, static_folder='static')

app.logger.info(app.root_path)

# app.static_folder = 'static'

label_encoder_path = os.path.join(app.root_path, 'models', 'label_encoder.pkl')
label_encoder_target_path = os.path.join(app.root_path, 'models', 'label_encoder_target.pkl')
final_model_path = os.path.join(app.root_path, 'models', 'nn_acc_06749.pkl')

with open(label_encoder_path, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

with open(label_encoder_target_path, 'rb') as let_file:
    label_encoder_target = pickle.load(let_file)

with open(final_model_path, 'rb') as predict_file:
    final_model  = pickle.load(predict_file)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/evaluateTriage', methods=['POST', 'GET'])
def evaluateTriage():

    scaler = StandardScaler()
    # Categorical Features
    genderVar = request.form.get('gender')
    arrivalmodeVar = request.form.get('arrivalMode')
    painlevelVar = request.form.get('pain')
    
    symptomsVar = request.form.getlist('symptoms')
    symptom1 = symptomsVar[1] if len(symptomsVar) > 0 else None
    symptom2 = symptomsVar[0] if len(symptomsVar) > 1 else None

    # Numerical features
    temperatureVar = float(request.form.get('temperature'))
    heartrateVar = float(request.form.get('heartrate'))
    resperateVar = float(request.form.get('resperate'))
    o2saturationVar = float(request.form.get('o2saturation'))
    sbpVar = float(request.form.get('sbp'))
    dbpVar = float(request.form.get('dbp'))

    categorical_features1 = (
        genderVar,
        arrivalmodeVar
       
    )

    categorical_features2 = (
        painlevelVar,
        symptom1,
        symptom2
    )
    
    # 1st model - encoding
    flat_categorical_features1 = np.array(categorical_features1).flatten()
    encoded_categorical_features1 = label_encoder.transform(flat_categorical_features1)

    flat_categorical_features2 = np.array(categorical_features2).flatten()
    encoded_categorical_features2 = label_encoder.transform(flat_categorical_features2)

    reshape_numerical_features = np.array([temperatureVar, heartrateVar, resperateVar, o2saturationVar, sbpVar, dbpVar]).reshape(-1, 1)
    # print('reshaped numerical', reshape_numerical_features)
    # print("shape", reshape_numerical_features.shape)

    scaled_numerical_features = scaler.fit(reshape_numerical_features)
    # print('scaled numerical after fit', scaled_numerical_features)

    scaled_numerical_features = scaler.transform(reshape_numerical_features)
    # print('scaled numerical before flatten', scaled_numerical_features)

    scaled_numerical_features = scaled_numerical_features.flatten()
    # print('scaled numerical after flatten', scaled_numerical_features)

    features = np.concatenate([encoded_categorical_features1, scaled_numerical_features, encoded_categorical_features2]).reshape(1,-1)
    # print('all features', features)

    # Use the loaded models for prediction
    encoded_target = final_model.predict(features)

    # Find the predicted class index
    predicted_class = np.argmax(encoded_target)

    # Map the index to the actual class label
    predicted_class_label = label_encoder_target.inverse_transform([predicted_class])[0]

    return render_template("index1.html", predicted_acuity = predicted_class_label)
    # return render_template("index1.html", data=predicted_class_label, data1 = predicted_class, genderVar = genderVar, 
    #                        arrivalmodeVar = arrivalmodeVar, painlevelVar = painlevelVar, symptom1 = symptom1, 
    #                        symptom2 = symptom2, temperatureVar = temperatureVar, heartrateVar = heartrateVar, 
    #                        resperateVar = resperateVar, o2saturationVar = o2saturationVar, sbpVar = sbpVar, dbpVar = dbpVar, 
    #                        reshape_numerical_features = reshape_numerical_features, scaled_numerical_features = scaled_numerical_features  )

if __name__ == '__main__':
    app.run(debug=True)