import numpy as np
import cv2
import pickle5 as pickle
import urllib
import json
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from fastapi import APIRouter, File, UploadFile
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from enum import Enum
from random import choice
from typing import List


class MemberStorage(str, Enum):
    student_storage = "student_storage"
    teaching_staff_storage  = "teaching_staff_storage"
    non_teaching_staff_storage = "non_teaching_staff_storage"
    visitor_storage = "visitor_storage"

from server.database import (
    retrieve_members_switcher,
    retrieve_member_switcher,
    update_member_switcher,
    collection_switcher,

    Member,
    

)

from server.models.student import (
    ErrorResponseModel,
    ResponseModel,
)

router = APIRouter()


# load the models
embeddding_model = load_model('/home/brian/app/media/ml_models/facenet_keras.h5') #Facenet model
classifier_model = pickle.load(open('/home/brian/app/media/ml_models/Knn_model.sav', 'rb')) #KNN model

# Creating face embeddings
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	embeddings = model.predict(samples)
	return embeddings[0]

#funtion to convert image downloaded from url to image arrays
async def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_array = np.asarray(image)
    
    # return the image array
    return face_array

#function to crop face from image using mtcnn
def face_cropper(image, required_size=(160, 160), detector = MTCNN()):
    if detector == None:
        return image
    image = Image.fromarray(image)       #open the image
    if image:
        image = image.convert('RGB')    #convert the image to RGB format 
        face_pixels = np.asarray(image)      #convert the image to numpy array
        face_pixels = cv2.cvtColor(face_pixels, cv2.COLOR_BGR2RGB)  #Converting the image from BGR to RGB
        f = detector.detect_faces(face_pixels)
        if f:
            #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
            x1,y1,w,h = f[0]['box']             
            x1, y1 = abs(x1), abs(y1)
            x2 = abs(x1+w)
            y2 = abs(y1+h)

            #locate the co-ordinates of face in the image
            store_face_cor = face_pixels[y1:y2, x1:x2]
            face = Image.fromarray(store_face_cor,'RGB')  #convert the numpy array to object
            face = face.resize(required_size)             #resize the image
            face_array = np.asarray(face)                 #image to array
            return face_array
        return None
        
    return None
    
#function to encove x variables and y labels
def encoder(x, y):
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    x = in_encoder.transform(x)
    
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(y)
    y = out_encoder.transform(y)

    return x, y, out_encoder

#Facial Recognition biometrics system model train function
def model_trainer(Model, members_data):
    pic_data = list()
    pic_label = list()

    #Fetching members data
    for member in members_data:
        if len(member["embeddings"])==0:
            continue
        pic_data.extend(np.array(member["embeddings"]))
        all_lb = [member["fullname"] for _ in range(len(member["embeddings"]))]
        pic_label.extend(all_lb)

    pic_data = np.array(pic_data)
    pic_label = np.array(pic_label)
    
    #Splitting data into train and test labels (test=0.25)
    tr_x, te_x, tr_y, te_y = train_test_split(pic_data, pic_label, test_size=0.20, random_state=42)
    
    #encoding all the data
    train = encoder(tr_x,tr_y)
    test = encoder(te_x,te_y)
    
    #encoded data
    trainX, testX, trainy, testy = train[0], test[0], train[1], test[1] 

    # fit model
    model = Model
    model.fit(trainX, trainy)

    #predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)

    # tr_ts score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)

    #ftd_encoder = train[2] #fitted (with trainy data) encodder.
    return score_train*100, score_test*100, tr_x, tr_y

#Function to enable model to make predictions
def predictor(classifier_model, data, face):
    trained_model = model_trainer(classifier_model, data)
    train_score =trained_model[0]
    test_score = trained_model[1]

    #Reading image from pic_Path, converting it to rgb, then finally resizing to 160*160
    #Precropped image
    """pic = path
    img = cv2.imread(pic)

    if type(img) == None:
        return None
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_np=np.asarray(img)
    img = Image.fromarray(img_np)
    rs_img=img.resize((160,160))
    val=np.asarray(rs_img)"""

    #Realtime Cropping
    val = face_cropper(face)
    if val is not None:
        #Encodding the data
        encoded_data = encoder(trained_model[2], trained_model[3])

        #fit model
        model = classifier_model
        trainX, trainy = encoded_data[0], encoded_data[1]
        model.fit(trainX, trainy)

        #Making Predictions on new data
        val_emb = get_embedding(embeddding_model,val)
        val = np.expand_dims(val_emb, axis = 0)
        yhat_class = model.predict(val)
        yhat_prob = model.predict_proba(val)

        #get Name
        out_encoder = encoded_data[2]      #Fitted encoder
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_name = out_encoder.inverse_transform(yhat_class)
        
        return  {
            "Predicted Name": "".join(str(x) for x in predict_name),
            "Prediction Score": class_probability,
            "Train_Score": train_score,
            "Test_Score": test_score
        }

    return None
    

# Function to fetch data after member is recognised by the model
async def face_data_retriever(member, data, face):
    #Running KNN machine learning algorithm to the train,test and validation data to get scores(probabilities)
    scores = predictor(classifier_model, data, face)
    if scores:
        if scores["Prediction Score"] >= 50:
            name = scores["Predicted Name"]
            data = collection_switcher(member).find_one({"fullname" : name})

            if data:
                return ResponseModel(
                    dict({ 
                        "scores": scores,
                        "results": {
                            "fullname": data['fullname'],
                            "reg_no": data['reg_no'],
                            "course": data['course']
                            }
                        }),
                    f"Member of name {name} is part of the institution",
                )
                
            return ErrorResponseModel(
                "An Error Occured",
                400,
                " No Data Found"
            )
            
        else:
            return ResponseModel(
                    f"Probability score({scores['Prediction Score']}) < 50%",
                    "Member Not Found",
                )

    return ErrorResponseModel(
                "An Error Occured",
                400,
                "Invalid Path"
            )

@router.put("/{id}",response_description="Update Member's Face Embeddings")
async def update_member_face_embeddings(Member: Member, id: str):
    #Retrieve a single member widh id
    member = await retrieve_member_switcher(Member.value, id)

    if member:
        face_urls = [list(i.values())[0] for i in member["pics"]]
        faces = [await url_to_image(url) for url in face_urls]
        face_embeddings = []
        for face_pixels in faces:
            face_pixels = np.array(face_pixels) # convert face pixels to arrays
            face_embeddings.append(get_embedding(embeddding_model,face_pixels).tolist()) # convert face embeddings to list for storage into the db

        await update_member_switcher(Member.value, id, {"embeddings": face_embeddings}) #update member embeddings
        return ResponseModel(
            face_embeddings,
            f"Embeddings update was successfull.",
        )

    return ErrorResponseModel(
        "An error occured updating the member's picture embeddings data",
        404,
        "Member does not exist!!"
    )

@router.post("/predict",response_description="Retrieving Member'data from prediction")
async def retrieve_facial_recognition_data(Member: Member, pic: UploadFile = File(...)):
    file_byts = pic.file.read()
    #converting file bytes to Array of bytes
    face = np.asarray(bytearray(file_byts), dtype="uint8")
    face = cv2.imdecode(face, cv2.IMREAD_COLOR)   #decoding bytesarray to Image
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  #Converting the image from BGR to RGB
    #Retrieve all Members data for train and test data
    members = await retrieve_members_switcher(Member.value)
    if members:
        data = await face_data_retriever(Member.value, members, face)
        return data

    return ResponseModel(
        "No Picture Found!",
        "Insufficient data"
    )