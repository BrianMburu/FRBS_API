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
    students = await retrieve_students()    #TIMEOUT EXCEPTION
    regs = [student["reg_no"].lower() for student in students]

    if student['reg_no'].lower() in regs:
        return ResponseModel("Error adding new student","Student already Exist!")

    new_student = await add_student(student)
    
    return ResponseModel(new_student,f"student added successfully")

@router.get("/", response_description="students retrieved")
async def get_students(short: bool = True):
    students = await retrieve_students(short)
    
    if students:
        return ResponseModel(students, "Students data retrieved successfully")
    
    return ResponseModel(students, "Empty list returned")

@router.get("/{id}", response_description="Student data retrieved")
async def get_student_data(id: str, short: bool = True):
    try:
        student = await retrieve_student(id, short) #TIMEOUT EXCEPTION
        
        if student:
            return ResponseModel(student, "Student data retrieved successfully")
        
        return ErrorResponseModel("An error occurred", 404, "Student doesn't exist.")
    except (Exception, RuntimeError, TimeoutError) as err:
        return ErrorResponseModel( 
            "An error occured while retrieving the student data",
            404, 
            str(err))

@router.put("/{id}")
async def update_student_data(id: str, req: UpdateStudentModel = Body(...)):
    try:
        req = {k: v for k,v in req.dict().items() if v is not None}
        updated_student = await update_student(id,req)
        
        if updated_student:
            return ResponseModel(
                "Student with ID: {} data updated!".format(id),
                "Student data updated successfully!!",
            )
                    
        return ErrorResponseModel(
            "An error occured",
            404,
            "There was an error updating the student data."
        )
    except (Exception, RuntimeError, TimeoutError) as err:
        return ErrorResponseModel( 
            "An error occured while updating the Student's data",
            404, 
            str(err))

@router.delete("/{id}", response_description="Student data deleted from the database")
async def delete_student_data(id: str):
    try:
        deleted_student = await delete_student(id)  #TIMEOUT EXCEPTION
        
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
    except (Exception, RuntimeError, TimeoutError) as err:
        return ErrorResponseModel( 
            "An error occured while deleting student's data",
            404, 
            str(err))