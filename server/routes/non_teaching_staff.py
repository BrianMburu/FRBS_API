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

"""
non_teaching_staff Picture Routes
"""

"""@router.post("/pictures", response_description="non_teaching_staff adding pictures to Gridfs Storage Bucket")
async def add_pictures(id: str, filenames: list = [],  files: list = []):
    if files != None and len(files) > 0:
        pics_idx = 0
        non_teaching_staff = await retrieve_non_teaching_staff(id)
        if non_teaching_staff:
            pics = non_teaching_staff['pics']
            for i in range(len(files)): 
                if files[i] != None or files[i] != "" and len(filenames)==len(files):
                    await upload_file(files[i], filenames[i])
                    pic_url = await retrieve_file(filenames[i])
                    pics.append({filenames[i]: pic_url})
                    pics_idx += 1
                continue
            await update_non_teaching_staff(id,{"pics": pics})

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
            "non_teaching_staff with id {0} doesn't exist".format(id)
        )
    else: 
        return ErrorResponseModel(
        "An error occurred", 
        404, 
        "No File specified"
    )

@router.get("/pictures", response_description="non_teaching_staff retrieving all pictures from Gridfs Storage Bucket")
async def retrieve_pictures():
    non_teaching_staffs = await retrieve_non_teaching_staffs()
    if non_teaching_staffs:
        pics = list()
        for non_teaching_staff in non_teaching_staffs: 
            for pic in non_teaching_staff['pics']:
                pic_url = await retrieve_file(pic)
                pics.append(pic_url)
                
        return ResponseModel(pics, "Pics data retrieved successfully")

    return ResponseModel(non_teaching_staffs, "Empty list returned")
    

@router.get("/{id}/pictures/{filename}", response_description="non_teaching_staff retrieving a picture from Gridfs Storage Bucket")
async def retrieve_picture(id: str,filename: str):
    non_teaching_staff = retrieve_non_teaching_staff(id)
    if non_teaching_staff:
        if filename:
            pic_url = await retrieve_file(filename)
            return pic_url

        return ErrorResponseModel(
            "Retrive operation failed",
            404,
            "File does not exist!",
        )
    return ErrorResponseModel("An error occurred", 404, "non_teaching_staff doesn't exist.")

@router.delete("/{id}/pictures/{filename}", response_description="non_teaching_staff deleting a picture from Gridfs Storage Bucket")
async def delete_picture(id: str,filename: str):
    non_teaching_staff = await retrieve_non_teaching_staff(id)
    if non_teaching_staff:
        pics = list()
        keys = list()
        for pic in non_teaching_staff['pics']:
            for key in pic.keys():
                keys.append(key)
            pics.append(pic)
        keys = set(keys)
        if filename in keys:
            await delete_file(filename)
            pics = list(filter(lambda i : i != filename, pics))
            await update_non_teaching_staff(id, {"pics": pics})
            return ResponseModel(
                "Operation was successful",
                f"{filename} deleted!"
            )
        return ErrorResponseModel(
            f"Delete operation failed{keys}",
            404,
            "File does not exist!",
        )
    return ErrorResponseModel("An error occurred", 404, "non_teaching_staff doesn't exist.")"""