import numpy as np
import cv2
import pickle5 as pickle
from keras.models import load_model
from fastapi import APIRouter, File, UploadFile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
from pytz import timezone
from enum import Enum

from server.database import (
    retrieve_members_switcher,
    collection_switcher,
    retrieve_all_fr_model_data,
    retrieve_fr_model_data,
    add_fr_model_data,
    delete_fr_model_data,
    
    Member,
)

from server.utils import(
    Facenet,
    get_embedding,
    face_cropper,
    encoder,
    data_fetcher,
)

from server.models.fr_model import (
    ErrorResponseModel,
    ResponseModel,
)

router = APIRouter()

#Predefining Weights
class Weights(str, Enum):
    distance = "distance"
    uniform = "uniform"

#Facial Recognition biometrics system model train function
async def model_trainer(Member: str, Neighbours: int, weight: str = "distance"):
    FILEPATH=f'/home/brian/Documents/Projects/School Project/frbs_api/media/ml_models/Knn_{Member}_model.sav'    #Storage Path

    #Retrieve all Members data for train and test data
    members = await retrieve_members_switcher(Member, False)
    X, y = data_fetcher(members, Member)

    #Splitting data into train and test labels (test=0.25)
    tr_x, te_x, tr_y, te_y = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #encoding all the data
    train = encoder(tr_x,tr_y)
    test = encoder(te_x,te_y)
    
    #encoded data
    trainX, testX, trainy, testy = train[0], test[0], train[1], test[1] 

    # fit model
    model = KNeighborsClassifier(n_neighbors = Neighbours, weights = weight)
    model.fit(trainX, trainy)

    #predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)

    # tr_ts score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)

    #Time settings
    fmt = '%Y-%m-%d %H:%M:%S %Z%z'
    current_time = datetime.now(timezone('Africa/Nairobi'))
    time = current_time.strftime(fmt)

    #Saving the Model Localy
    pickle.dump(model, open(FILEPATH, 'wb'))

    data = dict({   "Train_Score": score_train*100, 
                    "Test_Score": score_test*100,
                    "Train_Time": time,
                    "Data_Size":  {"X":len(X),"y":len(y)},
                    "Neighbours": Neighbours,
                    })

    #return score_train*100, score_test*100, tr_x, tr_y
    return data

#Function to enable model to make predictions
async def predictor(member: str, face):
    FILEPATH = f'/home/brian/Documents/Projects/School Project/frbs_api/media/ml_models/Knn_{member}_model.sav'
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

    #Realtime Cropping !MTCNN
    val = face_cropper(face)
    if val is not None:
        members = await retrieve_members_switcher(member, False)
        X, y = data_fetcher(members, member)            #fetching all data
        X_e, y_e, out_encoder = encoder(X, y)   #encoding all data

        #fit model
        classifier_model = pickle.load(open(FILEPATH, 'rb')) #KNN model

        model = classifier_model
        model.fit(X_e, y_e)

        #Making Predictions on new data
        val_emb = get_embedding(Facenet, val)
        val = np.expand_dims(val_emb, axis = 0)
        yhat_class = model.predict(val)
        yhat_prob = model.predict_proba(val)

        #get Name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_name = out_encoder.inverse_transform(yhat_class)
        
        return  {
            "Predicted Reg": "".join(str(x) for x in predict_name),
            "Prediction Score": class_probability
        }

    return None
    

# Function to fetch data after member is recognised by the model
async def face_data_retriever(member: str, face):
    #Running KNN machine learning algorithm to the train,test and validation data to get scores(probabilities)
    scores = await predictor(member, face)
    if scores:
        if scores["Prediction Score"] >= 90:
            reg = scores["Predicted Reg"]
            pred_data = collection_switcher(member).find_one({"reg_no" : reg}) if member == "student" else collection_switcher(member).find_one({"work_id" : reg})
            results = { "fullname":pred_data['fullname'],
                        "reg_no":  pred_data['reg_no'],
                        "course":  pred_data['course']
                        } if member == "student" else { "fullname":  pred_data['fullname'],
                                                        "work_id":   pred_data['work_id'],
                                                        "occupation":pred_data['occupation']}
            if pred_data:
                return ResponseModel(
                    dict({ 
                        "scores": scores,
                        "results": results
                        }),
                    f"Member of name {pred_data['fullname']} is part of the institution",
                )
                
            return ErrorResponseModel(
                "An Error Occured",
                400,
                " No Data Found"
            )
            
        else:
            return ResponseModel(
                    f"Probability score({scores['Prediction Score']}) < 90%",
                    "Member Not Found",
                )

    return ErrorResponseModel(
                "An Error Occured",
                400,
                "Invalid Path or Picture"
            )

@router.get("/", response_description="All fr_model data retrieved")
async def get_all_fr_model_data():
    all_fr_model = await retrieve_all_fr_model_data()
    
    if all_fr_model:
        return ResponseModel(all_fr_model, "All fr_model data retrieved successfully")
    
    return ResponseModel(all_fr_model, "Empty list returned")

@router.get("/{id}", response_description="fr_model data retrieved")
async def get_fr_model_data(id: str):
    fr_model = await retrieve_fr_model_data(id)
    
    if fr_model:
        return ResponseModel(fr_model, "fr_model data retrieved successfully")
    
    return ErrorResponseModel("An error occurred", 404, "fr_model doesn't exist.")

@router.put("/predict",response_description="Retrieved fr_model data from prediction")
async def predict(Member: Member, pic: UploadFile = File(...)):
    try:
        file_byts = pic.file.read()   #Converting upload image to bytes
        #converting file bytes to Array of bytes
        face = np.asarray(bytearray(file_byts), dtype="uint8")
        face = cv2.imdecode(face, cv2.IMREAD_COLOR)   #decoding bytesarray to Image Array
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  #Converting the image from BGR to RGB

        data = await face_data_retriever(Member.value, face)
    except (TimeoutError, MemoryError, ResourceWarning, RuntimeError) as err:
            return ErrorResponseModel( 
                "An error occured while Making Prediction", 
                404, 
                str(err))
    except Exception as err:
        return ErrorResponseModel( 
                "An error occured while Making Prediction", 
                404, 
                f"Invalid Image: Exception => {str(err)}")
    else:
        return data

@router.post("/train",response_description="Trained the Machine Learning model and saved it on server")
async def train(Member: Member, weight: Weights, Neighbours: int):
    try:
        data = await model_trainer(Member.value, Neighbours, weight.value)
        if data:
            post_data = {
                "Train_Score":data["Train_Score"],
                "Test_Score":data["Test_Score"],
                "Train_Time":data["Train_Time"],
                "Data_Size":data["Data_Size"],
                "Neighbours":data["Neighbours"],
            }
            await add_fr_model_data(post_data)
            return ResponseModel(data,"data saved in the db")

        return ErrorResponseModel(
            "An error occurred", 
            404, 
            "The Data corrupted or None"
        )
    except (ValueError, TimeoutError, MemoryError, ResourceWarning, RuntimeError) as err:
        return ErrorResponseModel( 
            "An error occured while Training Model", 
            404, 
            str(err))

@router.delete("/{id}", response_description="deleted fr_model data from the database")
async def delete_fr_model_data(id: str):
    try:
        deleted_fr_model_data = await delete_fr_model_data(id)
        if deleted_fr_model_data:
            return ResponseModel(
                "fr_model_data with ID: {} removed".format(id), 
                "fr_model_data deleted successfully"
            )
        
        return ErrorResponseModel(
            "An error occurred", 
            404, 
            "fr_model_data with id {0} doesn't exist".format(id)
        )
    except ( TimeoutError, MemoryError, ResourceWarning, RuntimeError) as err:
        return ErrorResponseModel(
            " An error occured while deleting Trained Model instance",
            404, 
            str(err))