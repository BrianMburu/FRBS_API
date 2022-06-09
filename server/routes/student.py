from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder

from server.database import (
    add_student,
    delete_student,
    retrieve_student,
    retrieve_students,
    update_student,
)

from server.models.student import (
    ErrorResponseModel,
    ResponseModel,
    StudentSchema,
    UpdateStudentModel,
)

router = APIRouter()

@router.post("/", response_description="student data added into the database")
async def add_student_data(student: StudentSchema = Body(...)):
    student = jsonable_encoder(student)
    new_student = await add_student(student)
    
    return ResponseModel(new_student,"student added successfully")

@router.get("/", response_description="students retrieved")
async def get_students(short: bool = True):
    students = await retrieve_students(short)
    
    if students:
        return ResponseModel(students, "Students data retrieved successfully")
    
    return ResponseModel(students, "Empty list returned")

@router.get("/{id}", response_description="Student data retrieved")
async def get_student_data(id: str, short: bool = True):
    student = await retrieve_student(id, short)
    
    if student:
        return ResponseModel(student, "Student data retrieved successfully")
    
    return ErrorResponseModel("An error occurred", 404, "Student doesn't exist.")

@router.put("/{id}")
async def update_student_data(id: str, req: UpdateStudentModel=Body(...)):
    req = {k: v for k,v in req.dict().items() if v is not None}
    updated_student = await update_student(id,req)
    
    if updated_student:
        return ResponseModel(
            "Student with ID: {} name update is successful".format(id),
            "Student name updated successfully",
        )
                 
    return ErrorResponseModel(
        "An error occured",
        404,
        "There was an error updating the student data."
    )

@router.delete("/{id}", response_description="Student data deleted from the database")
async def delete_student_data(id: str):
    deleted_student = await delete_student(id)
    
    if deleted_student:
        return ResponseModel(
            "Student with ID: {} removed".format(id), 
            "Student deleted successfully"
        )
    
    return ErrorResponseModel(
        "An error occurred", 
        404, 
        "Student with id {0} doesn't exist".format(id)
    )

"""
Student Picture Routes
"""


"""@router.post("/pictures", response_description="Student adding pictures to Gridfs Storage Bucket")
async def add_pictures(id: str, filenames: list = [],  files: list = []):
    if files != None and len(files) > 0:
        pics_idx = 0
        student = await retrieve_student(id)
        if student:
            pics = student['pics']
            for i in range(len(files)): 
                if files[i] != None or files[i] != "" and len(filenames)==len(files):
                    await upload_file(files[i], filenames[i])
                    pic_url = await retrieve_file(filenames[i])
                    pics.append({filenames[i]: pic_url})
                    pics_idx += 1
                continue
            await update_student(id,{"pics": pics})

            if pics_idx < 1:
                return ResponseModel(
                    "No Pics uploaded",
                    "Broken paths or files"
                )
            return ResponseModel(
                    "Operation was successful",
                    f"{pics_idx} pictures were uploaded"
                )
        return ErrorResponseModel(
            "An error occurred", 
            404, 
            "Mtudent with id {0} doesn't exist".format(id)
        )
    else: 
        return ErrorResponseModel(
        "An error occurred", 
        404, 
        "No File specified"
    )

@router.get("/pictures", response_description="Student retrieving all pictures from Gridfs Storage Bucket")
async def retrieve_pictures():
    students = await retrieve_students()
    if students:
        pics = list()
        for student in students: 
            for pic in student['pics']:
                pic_url = await retrieve_file(pic)
                pics.append(pic_url)
                
        return ResponseModel(pics, "Pics data retrieved successfully")

    return ResponseModel(students, "Empty list returned")
    

@router.get("/{id}/pictures/{filename}", response_description="Student retrieving a picture from Gridfs Storage Bucket")
async def retrieve_picture(id: str,filename: str):
    student = retrieve_student(id)
    if student:
        if filename:
            pic_url = await retrieve_file(filename)
            return pic_url

        return ErrorResponseModel(
            "Retrive operation failed",
            404,
            "File does not exist!",
        )
    return ErrorResponseModel("An error occurred", 404, "Student doesn't exist.")

@router.delete("/{id}/pictures/{filename}", response_description="Student deleting a picture from Gridfs Storage Bucket")
async def delete_picture(id: str, filename: str):
    student = await retrieve_student(id)
    if student:
        pics = list()
        keys = list()
        for pic in student['pics']:
            for key in pic.keys():
                keys.append(key)
            pics.append(pic)
        keys = set(keys)
        if filename in keys:
            await delete_file(filename)
            pics = list(filter(lambda i : i.keys()[0] != filename, pics))
            await update_student(id, {"pics": pics})
            return ResponseModel(
                "Operation was successful",
                f"{filename} deleted!"
            )
        return ErrorResponseModel(
            f"Delete operation failed{keys}",
            404,
            "File does not exist!",
        )
    return ErrorResponseModel("An error occurred", 404, "Student doesn't exist.")"""