from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder
from pydantic import Field

from server.database import (
    add_teaching_staff,
    delete_teaching_staff,
    retrieve_teaching_staff,
    retrieve_teaching_staffs,
    update_teaching_staff,
)

from server.models.teaching_staff import (
    ErrorResponseModel,
    ResponseModel,
    TeachingStaffSchema,
    UpdateTeachingStaffModel,
)

router = APIRouter()

@router.post("/", response_description="teaching_staff data added into the database")
async def add_teaching_staff_data(teaching_staff: TeachingStaffSchema = Body(...)):
    teaching_staff = jsonable_encoder(teaching_staff)
    teaching_staffs = await retrieve_teaching_staffs()
    work_ids = [teaching_staff["work_id"].lower() for teaching_staff in teaching_staffs]

    if teaching_staff['work_id'].lower() in work_ids:
        return ResponseModel("Error adding new teaching_staff","Teaching_staff already Exist!")

    new_teaching_staff = await add_teaching_staff(teaching_staff)
    
    return ResponseModel(new_teaching_staff,"teaching_staff added successfully")

@router.get("/", response_description="teaching_staffs retrieved")
async def get_teaching_staffs(short: bool = True):
    teaching_staffs = await retrieve_teaching_staffs(short)
    
    if teaching_staffs:
        return ResponseModel(teaching_staffs, "teaching_staffs data retrieved successfully")
    
    return ResponseModel(teaching_staffs, "Empty list returned")

@router.get("/{id}", response_description="teaching_staff data retrieved")
async def get_teaching_staff_data(id: str, short: bool = True):
    teaching_staff = await retrieve_teaching_staff(id, short)
    
    if teaching_staff:
        return ResponseModel(teaching_staff, "teaching_staff data retrieved successfully")
    
    return ErrorResponseModel("An error occurred", 404, "teaching_staff doesn't exist.")

@router.put("/{id}")
async def update_teaching_staff_data(id: str, req: UpdateTeachingStaffModel=Body(...)):
    req = {k: v for k,v in req.dict().items() if v is not None}
    updated_teaching_staff = await update_teaching_staff(id,req)
    
    if updated_teaching_staff:
        return ResponseModel(
            "teaching_staff with ID: {} name update is successful".format(id),
            "teaching_staff name updated successfully",
        )
                 
    return ErrorResponseModel(
        "An error occured",
        404,
        "There was an error updating the teaching_staff data."
    )

@router.delete("/{id}", response_description="teaching_staff data deleted from the database")
async def delete_teaching_staff_data(id: str):
    deleted_teaching_staff = await delete_teaching_staff(id)
    
    if deleted_teaching_staff:
        return ResponseModel(
            "teaching_staff with ID: {} removed".format(id), 
            "teaching_staff deleted successfully"
        )
    
    return ErrorResponseModel(
        "An error occurred", 
        404, 
        "teaching_staff with id {0} doesn't exist".format(id)
    )