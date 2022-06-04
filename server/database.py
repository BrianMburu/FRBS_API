import firebase_admin
import os
import cv2
import numpy as np
import io
#import gridfs
#import pyrebase
#import motor.motor_asyncio
from mtcnn.mtcnn import MTCNN
from environs import Env
from pymongo import MongoClient
from bson.objectid import ObjectId
from firebase_admin import credentials
from firebase_admin import storage
from enum import Enum
from PIL import Image

env = Env()
env.read_env()

#Mongo Database config
MONGO_DETAILS = env("MONGO_DETAILS")

#client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)     #Motor
client = MongoClient(MONGO_DETAILS)     #PyMongo

student_database = client.students
teaching_staff_database = client.teaching
non_teaching_staff_database = client.non_teaching
visitor_database = client.visitors

student_collection = student_database.student_collection
teaching_staff_collection = teaching_staff_database.teaching_collection
non_teaching_staff_collection = non_teaching_staff_database.non_teaching_collection
visitor_collection = visitor_database.get_collection("visitors_collection")


#firestore Storage Config
credentials = credentials.Certificate("/home/brian/app/ServiceAccountKey.json")
firebase_admin.initialize_app(credentials, {
    "storageBucket": env("STORAGE_BUCKET")
    })

bucket = storage.bucket()   #Storage bucket

"""
Switchers
"""
#Swtchers to simplify using different api functions.
def collection_switcher(arg):
    switcher = {
        "student": student_collection,
        "teaching_staff":   teaching_staff_collection,
        "non_teaching_staff": non_teaching_staff_collection,
    }
    return switcher.get(arg, None)

async def retrieve_members_switcher(arg):
    switcher = {
        "student": await retrieve_students(),
        "teaching_staff":   await retrieve_teaching_staffs(),
        "non_teaching_staff": await retrieve_non_teaching_staffs(),
    }
    return switcher.get(arg, None)

async def retrieve_member_switcher(arg: str, id: str,):
    switcher = {
        "student": await retrieve_student(id),
        "teaching_staff":   await retrieve_teaching_staff(id),
        "non_teaching_staff": await retrieve_non_teaching_staff(id),
    }
    return switcher.get(arg, None)

async def update_member_switcher(arg: str, id: str, data: dict):
    switcher = {
        "student": await update_student(id, data),
        "teaching_staff":   await update_teaching_staff(id, data),
        "non_teaching_staff": await update_non_teaching_staff(id, data),
    }
    return switcher.get(arg, None)

#Predefining all members
class Member(str, Enum):
    student = "student"
    teaching_staff = "teaching_staff"
    non_teaching_staff= "non_teaching_staff"

"""
Student
"""
def student_helper(student) -> dict:
    return {
        "id": str(student["_id"]),
        "fullname": student["fullname"],
        "reg_no": student["reg_no"],
        "course": student["course"],
        "pics": student["pics"],
        "embeddings": student["embeddings"]
    }
# Retrieve all students present in the database
async def retrieve_students():
    students=[]
    for student in student_collection.find():
        students.append(student_helper(student))
    return students

# Add a new student into to the database
async def add_student(student_data: dict) -> dict:
    student =  student_collection.insert_one(student_data)
    new_student =  student_collection.find_one({"_id": student.inserted_id})
    return student_helper(new_student)

# Retrieve a student with a matching ID
async def retrieve_student(id: str)-> dict:
    student =  student_collection.find_one({"_id": ObjectId(id)})
    if student:
        return student_helper(student)

# Update a student with a matching ID
async def update_student(id: str,data: dict):
    #Return false if an empty request body is sent.
    if len(data) <1:
        return False
    student =  student_collection.find_one({"_id": ObjectId(id)})
    if student:
        updated_student =  student_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_student:
            return True
        return False

# Delete a student from the database
async def delete_student(id: str):
    student =  student_collection.find_one({"_id": ObjectId(id)})
    if student:
        student_collection.delete_one({"_id": ObjectId(id)})
        return True


"""
STAFF
"""
# staff helpers
def staff_helper(staff) -> dict:
    return {
        "id": str(staff["_id"]),
        "fullname": staff["fullname"],
        "work_id": staff["work_id"],
        "department": staff["department"],
        "occupation": staff["occupation"],
        "pics": staff["pics"],
        "embeddings": staff["embeddings"]
    }

"""
Teaching
"""
    
# Retrieve all teaching staff present in the database
async def retrieve_teaching_staffs():
    teaching_staffs=[]
    for teaching_staff in teaching_staff_collection.find():
        teaching_staffs.append(staff_helper(teaching_staff))
    return teaching_staffs

# Add a new teaching staff into to the database
async def add_teaching_staff(teaching_staff_data: dict) -> dict:
    teaching_staff =  teaching_staff_collection.insert_one(teaching_staff_data)
    new_teaching_staff =  teaching_staff_collection.find_one({"_id": teaching_staff.inserted_id})
    return staff_helper(new_teaching_staff)

# Retrieve a teaching staff with a matching ID
async def retrieve_teaching_staff(id: str)-> dict:
    teaching_staff =  teaching_staff_collection.find_one({"_id": ObjectId(id)})
    if teaching_staff:
        return staff_helper(teaching_staff)

# Update a teaching staff with a matching ID
async def update_teaching_staff(id: str,data: dict):
    #Return false if an empty request body is sent.
    if len(data) <1:
        return False
    teaching_staff =  teaching_staff_collection.find_one({"_id": ObjectId(id)})
    if teaching_staff:
        updated_teaching_staff =  teaching_staff_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_teaching_staff:
            return True
        return False

# Delete a teaching staff from the database
async def delete_teaching_staff(id: str):
    teaching_staff =  teaching_staff_collection.find_one({"_id": ObjectId(id)})
    if teaching_staff:
        teaching_staff_collection.delete_one({"_id": ObjectId(id)})
        return True
    
"""
Non Teaching
"""
    
# Retrieve all Non teaching staff present in the database
async def retrieve_non_teaching_staffs():
    non_teaching_staffs=[]
    for non_teaching_staff in non_teaching_staff_collection.find():
        non_teaching_staffs.append(staff_helper(non_teaching_staff))
    return non_teaching_staffs

# Add a new Non teaching staff into to the database
async def add_non_teaching_staff(non_teaching_staff_data: dict) -> dict:
    non_teaching_staff =  non_teaching_staff_collection.insert_one(non_teaching_staff_data)
    new_non_teaching_staff =  non_teaching_staff_collection.find_one({"_id": non_teaching_staff.inserted_id})
    return staff_helper(new_non_teaching_staff)

# Retrieve a Non teaching staff with a matching ID
async def retrieve_non_teaching_staff(id: str)-> dict:
    non_teaching_staff =  non_teaching_staff_collection.find_one({"_id": ObjectId(id)})
    if non_teaching_staff:
        return staff_helper(non_teaching_staff)

# Update a Non teaching staff with a matching ID
async def update_non_teaching_staff(id: str,data: dict):
    #Return false if an empty request body is sent.
    if len(data) <1:
        return False
    non_teaching_staff =  non_teaching_staff_collection.find_one({"_id": ObjectId(id)})
    if non_teaching_staff:
        updated_non_teaching_staff =  non_teaching_staff_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_non_teaching_staff:
            return True
        return False

# Delete a Non teaching staff from the database
async def delete_non_teaching_staff(id: str):
    non_teaching_staff =  non_teaching_staff_collection.find_one({"_id": ObjectId(id)})
    if non_teaching_staff:
        non_teaching_staff_collection.delete_one({"_id": ObjectId(id)})
        return True

"""
Visitor
"""

# Visitor Helper
def visitor_helper(visitor) -> dict:
    return {
        "id": str(visitor["_id"]),
        "fullname": visitor["fullname"],
        "index": visitor["index"],
        "Appointment": visitor["Appointment"],
        "Approved": visitor["Approved"],
    }

# Retrieve all visitors present in the database
async def retrieve_visitors():
    visitors=[]
    for visitor in visitor_collection.find():
        visitors.append(visitor_helper(visitor))
    return visitors

# Add a new visitor into to the database
async def add_visitor(visitor_data: dict) -> dict:
    visitor = await visitor_collection.insert_one(visitor_data)
    new_visitor = await visitor_collection.find_one({"_id": visitor.inserted_id})
    return visitor_helper(new_visitor)

# Retrieve a visitor with a matching ID
async def retrieve_visitor(id: str)-> dict:
    visitor = await visitor_collection.find_one({"_id": ObjectId(id)})
    if visitor:
        return visitor_helper(visitor)

# Update a visitor with a matching ID
async def update_visitor(id: str,data: dict):
    #Return false if an empty request body is sent.
    if len(data) < 1:
        return False
    visitor = await visitor_collection.find_one({"_id": ObjectId(id)})
    if visitor:
        updated_visitor = await visitor_collection.update_one(
            {"_id": ObjectId(id)}, {"$set": data}
        )
        if updated_visitor:
            return True
        return False

# Delete a visitor from the database
async def delete_visitor(id: str):
    visitor = await visitor_collection.find_one({"_id": ObjectId(id)})
    if visitor:
        await visitor_collection.delete_one({"_id": ObjectId(id)})
        return True


"""
File
"""
from server.routes.fr_model import(
    face_cropper
)

#Save a file from file data
async def upload_file(file, filename: str, folder: str = "media/images", content_type: str = "image/jpeg", required_size=(160, 160), detector = MTCNN()):
    if file:
        #converting file bytes to Array of bytes
        image = np.asarray(bytearray(file), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)   #decoding bytesarray to Image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  #Converting the image from BGR to RGB
        face = face_cropper(image, required_size, detector)  #Cropping the face using MTCNN and converting to an array
        
        if face is not None: 
            is_success, im_buf_arr = cv2.imencode(".jpg", face)
            byte_im = im_buf_arr.tobytes()
            blob = bucket.blob(os.path.join(folder, filename))      
            blob.upload_from_string(byte_im, content_type)
            blob.make_public()
            return is_success
        return False

    return False

#Retrieve a File
async def retrieve_file(filename: str, folder: str = "media/images"):
    blob = bucket.blob(os.path.join(folder, filename))
    if blob:
        url = blob.public_url
        if url:
            return url
        return None
    return None

#delete a file
async def delete_file(filename: str, folder: str = "media/images"):
    blob = bucket.blob(os.path.join(folder, filename))
    if blob:
        blob.delete()
        return True
    return False