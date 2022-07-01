import numpy as np
import cv2
import urllib
import copy

from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from keras.models import load_model

#Load Facenet Embedder
Facenet = load_model('/home/brian/Documents/Projects/School Project/frbs_api/media/ml_models/facenet_keras.h5') #Facenet model


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

# Creating face Augmentations
def get_augmentations(face_pixels):
    if face_pixels is not None:
        #flipping face_pixels in horizontally
        pixels = copy.deepcopy(face_pixels)
        
        aug_list = list()
        face_pixels_hf = copy.deepcopy(pixels)
        face_pixels_hf = np.flip(face_pixels_hf, axis=1)
        aug_list.append(face_pixels_hf)

        #LIghtening the face_pixels 
        face_pixels_lt = copy.deepcopy(pixels)
        rand_val_lt = round(np.random.uniform(1.5, 2.0), 3)
        for i in range(len(face_pixels_lt)):
            face_pixels_lt[i] = np.clip(rand_val_lt*face_pixels_lt[i], 0.0, 255.0)
        aug_list.append(face_pixels_lt)
        
        #darkening the face_pixels
        face_pixels_dk = copy.deepcopy(pixels)
        rand_val_dk = round(np.random.uniform(0.2, 0.5), 3)
        for i in range(len(face_pixels_dk)):
            face_pixels_dk[i] = np.clip(rand_val_dk*face_pixels_dk[i], 0.0, 255.0)
        aug_list.append(face_pixels_dk)

        return aug_list

    return None
    
    

#funtion to convert image downloaded from url to image arrays
async def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    if url != None or url != "":
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_array = np.asarray(image)
        
        # return the image array
        return face_array
    return None

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

#Function to fetch all embedings and their respective labels
def data_fetcher(members, member_t):
    pic_data = list()
    pic_label= list()
    emb_data = list()

    #Fetching members data
    for member in members:
        if len(member["embeddings"])==0:
            continue
        emb_data = [list(i.values())[0] for i in member["embeddings"]]
        pic_data.extend(np.array(emb_data))
        all_lb = [member["reg_no"] for _ in range(len(member["embeddings"]))] if member_t=="student" else [member["work_id"] for _ in range(len(member["embeddings"]))]
        pic_label.extend(all_lb)

    pic_data = np.array(pic_data)   #X variables (embeddings).
    pic_label = np.array(pic_label) #y labels

    return pic_data, pic_label
