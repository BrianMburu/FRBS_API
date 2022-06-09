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

"""
teaching_staff Picture Routes
"""


"""@router.post("/pictures", response_description="teaching_staff adding pictures to Gridfs Storage Bucket")
async def add_pictures(id: str, filenames: list = [],  files: list = []):
    if files != None and len(files) > 0:
        pics_idx = 0
        teaching_staff = await retrieve_teaching_staff(id)
        if teaching_staff:
            pics = teaching_staff['pics']
            for i in range(len(files)): 
                if files[i] != None or files[i] != "" and len(filenames)==len(files):
                    await upload_file(files[i], filenames[i])
                    pic_url = await retrieve_file(filenames[i])
                    pics.append({filenames[i]: pic_url})
                    pics_idx += 1
                continue
            await update_teaching_staff(id,{"pics": pics})

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
            "teaching_staff with id {0} doesn't exist".format(id)
        )
    else: 
        return ErrorResponseModel(
        "An error occurred", 
        404, 
        "No File specified"
    )

@router.get("/pictures", response_description="teaching_staff retrieving all pictures from Gridfs Storage Bucket")
async def retrieve_pictures():
    teaching_staffs = await retrieve_teaching_staffs()
    if teaching_staffs:
        pics = list()
        for teaching_staff in teaching_staffs: 
            for pic in teaching_staff['pics']:
                pic_url = await retrieve_file(pic)
                pics.append(pic_url)
                
        return ResponseModel(pics, "Pics data retrieved successfully")

    return ResponseModel(teaching_staffs, "Empty list returned")
    

@router.get("/{id}/pictures/{filename}", response_description="teaching_staff retrieving a picture from Gridfs Storage Bucket")
async def retrieve_picture(id: str,filename: str):
    teaching_staff = retrieve_teaching_staff(id)
    if teaching_staff:
        if filename:
            pic_url = await retrieve_file(filename)
            return pic_url

        return ErrorResponseModel(
            "Retrive operation failed",
            404,
            "File does not exist!",
        )
    return ErrorResponseModel("An error occurred", 404, "teaching_staff doesn't exist.")

@router.delete("/{id}/pictures/{filename}", response_description="teaching_staff deleting a picture from Gridfs Storage Bucket")
async def delete_picture(id: str,filename: str):
    teaching_staff = await retrieve_teaching_staff(id)
    if teaching_staff:
        pics = list()
        keys = list()
        for pic in teaching_staff['pics']:
            for key in pic.keys():
                keys.append(key)
            pics.append(pic)
        keys = set(keys)
        if filename in keys:
            await delete_file(filename)
            pics = list(filter(lambda i : i != filename, pics))
            await update_teaching_staff(id, {"pics": pics})
            return ResponseModel(
                "Operation was successful",
                f"{filename} deleted!"
            )
        return ErrorResponseModel(
            f"Delete operation failed{keys}",
            404,
            "File does not exist!",
        )
    return ErrorResponseModel("An error occurred", 404, "Member doesn't exist.")"""