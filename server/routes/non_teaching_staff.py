from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder
from pydantic import Field

from server.database import (
    add_non_teaching_staff,
    delete_non_teaching_staff,
    retrieve_non_teaching_staff,
    retrieve_non_teaching_staffs,
    update_non_teaching_staff,
)

from server.models.non_teaching_staff import (
    ErrorResponseModel,
    ResponseModel,
    NonTeachingStaffSchema,
    UpdateNonTeachingStaffModel,
)

router = APIRouter()

@router.post("/", response_description="nonteaching_staff data added into the database")
async def add_non_teaching_staff_data(non_teaching_staff: NonTeachingStaffSchema = Body(...)):
    non_teaching_staff = jsonable_encoder(non_teaching_staff)
    non_teaching_staffs = await retrieve_non_teaching_staffs()
    work_ids = [non_teaching_staff["work_id"].lower() for non_teaching_staff in non_teaching_staffs]

    if non_teaching_staff['work_id'].lower() in work_ids:
        return ResponseModel("Error adding new non_teaching_staff","Non_teaching_staff already Exist!")

    new_non_teaching_staff = await add_non_teaching_staff(non_teaching_staff)
    
    return ResponseModel(new_non_teaching_staff,"non_teaching_staff added successfully")

@router.get("/", response_description="non_teaching_staffs retrieved")
async def get_non_teaching_staffs(short: bool = True):
    non_teaching_staffs = await retrieve_non_teaching_staffs(short)
    
    if non_teaching_staffs:
        return ResponseModel(non_teaching_staffs, "non_teaching_staffs data retrieved successfully")
    
    return ResponseModel(non_teaching_staffs, "Empty list returned")

@router.get("/{id}", response_description="non_teaching_staff data retrieved")
async def get_non_teaching_staff_data(id: str, short: bool = True):
    non_teaching_staff = await retrieve_non_teaching_staff(id, short)
    
    if non_teaching_staff:
        return ResponseModel(non_teaching_staff, "non_teaching_staff data retrieved successfully")
    
    return ErrorResponseModel("An error occurred", 404, "non_teaching_staff doesn't exist.")

@router.put("/{id}")
async def update_non_teaching_staff_data(id: str, req: UpdateNonTeachingStaffModel=Body(...)):
    req = {k: v for k,v in req.dict().items() if v is not None}
    updated_non_teaching_staff = await update_non_teaching_staff(id,req)
    
    if updated_non_teaching_staff:
        return ResponseModel(
            "non_teaching_staff with ID: {} name update is successful".format(id),
            "non_teaching_staff name updated successfully",
        )
                 
    return ErrorResponseModel(
        "An error occured",
        404,
        "There was an error updating the non_teaching_staff data."
    )

@router.delete("/{id}", response_description="non_teaching_staff data deleted from the database")
async def delete_non_teaching_staff_data(id: str):
    deleted_non_teaching_staff = await delete_non_teaching_staff(id)
    
    if deleted_non_teaching_staff:
        return ResponseModel(
            "non_teaching_staff with ID: {} removed".format(id), 
            "non_teaching_staff deleted successfully"
        )
    
    return ErrorResponseModel(
        "An error occurred", 
        404, 
        "non_teaching_staff with id {0} doesn't exist".format(id)
    )