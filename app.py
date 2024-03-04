import numpy as np
from flask import Flask,request,jsonify
from flask_cors import CORS
import pickle
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity,verify_jwt_in_request
from keras.utils import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import base64


uri = "mongodb+srv://mdyusuf2521:ndKgi0pnBnB32bkX@cluster0.sfnvtv8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

app=Flask(__name__)
cors = CORS(app)


#model=pickle.load(open('model.pkl','rb'))

###############################################################################################################################################

model0 = load_model("arthritis.h5")
model1 = load_model("brain_tumor.h5")
model2 = load_model("coronary_artery_disease.h5")
model3 = load_model("kidney_failure.h5")
model4_1 = load_model("breast_cancer.h5")
model4_2 = load_model("covid.h5")
model4_3 = load_model("lungs_cancer.h5")
model4_4 = load_model("tuberculosis.h5")

###############################################################################################################################################

print("Loaded model from disk")


# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['MLProject']

app.config['JWT_SECRET_KEY'] = 'iuasoirgKGSDIOUgad+O)*Y0iwerwqer'  
jwt = JWTManager(app)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


#############################################################################################################################################
    
class_label0 = ["Arthritis Detected", "Normal"]
class_label1 = ["Benign Cancer Detected", "Malign Cancer Detected", "Normal"]
class_label2 = ["Coronary Artery Disease Suspected", "Normal"]
class_label3 = ["Kidney Failure Detected", "Normal"]
class_label4_1 = ["Breast Cancer Detected", "Normal"]
class_label4_2 = ["Corona Virus Detected", "Normal"]
class_label4_3 = ["Lungs Cancer Detected", "Normal"]
class_label4_4 = ["Tuberculosis Detected", "Normal"]

#############################################################################################################################################


#controllers function
def find_data(collection,email):
    try:
        data = db[collection].find_one({'email': email})
        return data
    except Exception as e:
        print(f"Error finding data: {e}")
        return None


def load_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor,axis=0)
    img_tensor = img_tensor/255
    return img_tensor

#requests
@app.route('/')
def hello():
    return "hello world"

@app.route('/predict',methods=['POST'])
def predict():
    # file=request.files['file']
    # category=request.form['category']
    # subcategory=request.form['subcategory']

    image_data_uri=request.json['image']
    image_data = base64.b64decode(image_data_uri.split(',')[1])
    category=request.json['category']
    subcategory=request.json['subCategory']

    print(category)
    print(subcategory)  

    
###############################################################################################################################################
    
    if category == "Bones":
        if subcategory == "arthritis":
            save_path0 = 'uploads/bones'

            if not os.path.exists(save_path0):
                os.makedirs(save_path0)

            file_path = os.path.join(save_path0, 'uploaded_image0.jpg')

            with open(file_path, 'wb') as f:
                f.write(image_data)

    if category == "Brain":
        if subcategory == "brain_tumor":
            save_path1 = 'uploads/brain'

            if not os.path.exists(save_path1):
                os.makedirs(save_path1)

            file_path = os.path.join(save_path1, 'uploaded_image1.jpg')

            with open(file_path, 'wb') as f:
                f.write(image_data)

    if category == "Heart":
        if subcategory == "coronary_artery_disease":
            save_path2 = 'uploads/heart'

            if not os.path.exists(save_path2):
                os.makedirs(save_path2)

            file_path = os.path.join(save_path2, 'uploaded_image2.jpg')

            with open(file_path, 'wb') as f:
                f.write(image_data)

    if category == "Kidney":
        if subcategory == "kidney_failure":
            save_path3 = 'uploads/kidney'

            if not os.path.exists(save_path3):
                os.makedirs(save_path3)

            file_path = os.path.join(save_path3, 'uploaded_image3.jpg')

            with open(file_path, 'wb') as f:
                f.write(image_data)

    if category == "Lungs":
        if subcategory == "breast_cancer":
            save_path4_1 = 'uploads/lungs'

            if not os.path.exists(save_path4_1):
                os.makedirs(save_path4_1)

            file_path = os.path.join(save_path4_1, 'uploaded_image4_1.jpg')

            with open(file_path, 'wb') as f:
                f.write(image_data)
        
        elif subcategory == "covid":
            save_path4_2 = 'uploads/lungs'

            if not os.path.exists(save_path4_2):
                os.makedirs(save_path4_2)

            file_path = os.path.join(save_path4_2, 'uploaded_image4_2.jpg')

            with open(file_path, 'wb') as f:
                f.write(image_data)
    
        elif subcategory == "lungs_cancer":
            save_path4_3 = 'uploads/lungs'

            if not os.path.exists(save_path4_3):
                os.makedirs(save_path4_3)

            file_path = os.path.join(save_path4_3, 'uploaded_image4_3.jpg')

            with open(file_path, 'wb') as f:
                f.write(image_data)
        
        elif subcategory == "tuberculosis":
            save_path4_4 = 'uploads/lungs'

            if not os.path.exists(save_path4_4):
                os.makedirs(save_path4_4)

            file_path = os.path.join(save_path4_4, 'uploaded_image4_4.jpg')

            with open(file_path, 'wb') as f:
                f.write(image_data)    
    
################################################################################################################################################
    if category == "Bones":
        if subcategory == "arthritis":
            path0 = 'D:/projects/MachineLearning/HackJMI/uploads/bones/uploaded_image0.jpg'
            loaded_image = load_image(path0)
            prediction = model0.predict(loaded_image)
            class_id = np.argmax(prediction, axis=1) #max value
            output=class_label0[int(class_id)]#id wrt dataset
            output=str(output)
            return jsonify({'disease':output}),200
        
    if category == "Brain":
        if subcategory == "brain_tumor":
            path1 = 'D:/projects/MachineLearning/HackJMI/uploads/brain/uploaded_image1.jpg'
            loaded_image = load_image(path1)
            prediction = model1.predict(loaded_image)
            class_id = np.argmax(prediction, axis=1) #max value
            output=class_label1[int(class_id)]#id wrt dataset
            output=str(output)
            return jsonify({'disease':output}),200
        
    if category == "Heart":
        if subcategory == "coronary_artery_disease":
            path2 = 'D:/projects/MachineLearning/HackJMI/uploads/heart/uploaded_image2.jpg'
            loaded_image = load_image(path2)
            prediction = model2.predict(loaded_image)
            class_id = np.argmax(prediction, axis=1) #max value
            output=class_label2[int(class_id)]#id wrt dataset
            output=str(output)
            return jsonify({'disease':output}),200
        
    if category == "Kidney":
        if subcategory == "kidney_failure":
            path3 = 'D:/projects/MachineLearning/HackJMI/uploads/kidney/uploaded_image3.jpg'
            loaded_image = load_image(path3)
            prediction = model3.predict(loaded_image)
            class_id = np.argmax(prediction, axis=1) #max value
            output=class_label3[int(class_id)]#id wrt dataset
            output=str(output)
            return jsonify({'disease':output}),200
    
    if category == "Lungs":
        if subcategory == "breast_cancer":
            path4_1 = 'D:/projects/MachineLearning/HackJMI/uploads/lungs/uploaded_image4_1.jpg'
            loaded_image = load_image(path4_1)
            prediction = model4_1.predict(loaded_image)
            class_id = np.argmax(prediction, axis=1) #max value
            output=class_label4_1[int(class_id)]#id wrt dataset
            output=str(output)
            return jsonify({'disease':output}),200
        
        elif subcategory == "covid":
            path4_2 = 'D:/projects/MachineLearning/HackJMI/uploads/lungs/uploaded_image4_2.jpg'
            loaded_image = load_image(path4_2)
            prediction = model4_2.predict(loaded_image)
            class_id = np.argmax(prediction, axis=1) #max value
            output=class_label4_2[int(class_id)]#id wrt dataset
            output=str(output)
            return jsonify({'disease':output}),200
        
        elif subcategory == "lungs_cancer":
            path4_3 = 'D:/projects/MachineLearning/HackJMI/uploads/lungs/uploaded_image4_3.jpg'
            loaded_image = load_image(path4_3)
            prediction = model4_3.predict(loaded_image)
            class_id = np.argmax(prediction, axis=1) #max value
            output=class_label4_3[int(class_id)]#id wrt dataset
            output=str(output)
            return jsonify({'disease':output}),200
        
        elif subcategory == "tuberculosis":
            path4_4 = 'D:/projects/MachineLearning/HackJMI/uploads/lungs/uploaded_image4_4.jpg'
            loaded_image = load_image(path4_4)
            prediction = model4_4.predict(loaded_image)
            class_id = np.argmax(prediction, axis=1) #max value
            output=class_label4_4[int(class_id)]#id wrt dataset
            output=str(output)
            return jsonify({'disease':output}),200
    
##############################################################################################################################################





@app.route('/login',methods=['POST'])
def login():
    body=request.get_json()
 
    dbVal=find_data(body['person'],body['email'])
    if dbVal==None:
        res=jsonify({"status":"login failed"})
        res.status_code=404
        return res

    #check for the password
    if dbVal['password']!=body['password']:
        res=jsonify({"status":"wrong email or password"})
        res.status_code=403
        return res
    
    #create access token
    user_info = {'email': body['email'], 'role': body['person']}
    access_token = create_access_token(identity=user_info)
    redirect=''
    if body['person']=='patient':
        redirect='/profile/patient'
    elif body['person']=='doctor':
        redirect='/profile/doctor'
    
    res=jsonify({"accessToken":access_token,"redirect":redirect})
    res.status_code=200
    return res


@app.route('/register',methods=['POST'])
def register():
    body=request.get_json()

    dbVal=find_data(body['person'],body['email'])
    if dbVal!=None:
        res=jsonify({"message":"user with that email already exits"})
        res.status_code=404
        return res
    
    #save the user to the database
    db[body['person']].insert_one(body)

    #create access token
    user_info = {'email': body['email'], 'role': body['person']}
    access_token = create_access_token(identity=user_info)
    redirect=''
    if body['person']=='patient':
        redirect='/profile/patient'
    elif body['person']=='doctor':
        redirect='/profile/doctor'
    res=jsonify({"accessToken":access_token,"redirect":redirect})
    res.status_code=200
    return res
    


@app.route('/details/patient')
def fetch_patient_details():
    try:
        verify_jwt_in_request()
    except Exception as e:
        return jsonify({'message': 'Unauthorized'}), 401
    user = get_jwt_identity()
    patient_details=find_data('patient',user['email'])
    del patient_details['_id']
    del patient_details['password']
    res=jsonify({"data":patient_details})
    res.status_code=200
    return res


@app.route('/bookappointment',methods=['POST'])
def book():
    body=request.get_json()
    try:
        verify_jwt_in_request() 
    except Exception as e:
        return jsonify({'message': 'Unauthorized'}), 401
    user = get_jwt_identity()
    #check for the availability of
    all_doctor_details=db['doctor'].find({"specialization":body['doctor']})
    all_doctor_details=list(all_doctor_details)
    doctor_details=all_doctor_details[0]
    del doctor_details['password']
    del doctor_details['_id']

    patient_details=db['patient'].find_one({"email":user['email']})
    del patient_details['_id']
    del patient_details['password']

    storePayload={
        'doctor':doctor_details,
        'patientDetails':patient_details,
        'symptoms':body['symptoms'],
        'date':body['date'],
        'status':'pending',
    }



    db['temp_appointment'].insert_one(storePayload)
    res=jsonify({"success":True})
    res.status_code=200
    return res


@app.route('/appointmentRequest')
def temp_appointment():
    try:
        verify_jwt_in_request() 
    except Exception as e:
        return jsonify({'message': 'Unauthorized'}), 401
    user = get_jwt_identity()
    if user:
        temp_appointments=db['temp_appointment'].find({"doctor.email":user['email']})
        temp_appointments=list(temp_appointments)

        for e in temp_appointments:
            e.pop('_id', None)

        res=jsonify({"data":temp_appointments})
        res.status_code=200
        return res
    else:
         return jsonify({'message': 'Unauthorized'}), 404


@app.route('/appointmentRequest/accept',methods=['POST'])
def acceptAppointment():
    body=request.get_json()
    try:
        verify_jwt_in_request() 
    except Exception as e:
        return jsonify({'message': 'Unauthorized'}), 401
    user = get_jwt_identity()
    if user:
        appointment=db['temp_appointment'].find_one({"$and":[{"patientDetails.email":body['patientEmail']},{"doctor.email":user['email']}]})
        del appointment['_id']
        db['appointment'].insert_one(appointment)

        db['temp_appointment'].delete_one({"$and":[{"patientDetails.email":body['patientEmail']},{"doctor.email":user['email']}]})

        res=jsonify({"success ":True})
        res.status_code=200
        return res
    else:
         return jsonify({'message': 'Unauthorized'}), 404


# @app.route('/appointmentRequest/reject',methods=['POST'])
# def rejectAppointment():



@app.route('/details/doctor')
def doctor_details():
    try:
        verify_jwt_in_request()
    except Exception as e:
        return jsonify({'message': 'Unauthorized'}), 401

    user = get_jwt_identity()
    doctor_details=find_data('doctor',user['email'])
    del doctor_details['password']
    del doctor_details['_id']
    
    res=jsonify({"data":doctor_details})
    res.status_code=200
    return res



if __name__=='__main__':
    app.run(debug=True)


#################################################################################################################################################


