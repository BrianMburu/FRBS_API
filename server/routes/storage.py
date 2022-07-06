from typing import List
from fastapi import APIRouter, File, UploadFile ,Body
from fastapi.encoders import jsonable_encoder

from server.database import (
    retrieve_members_switcher,
    retrieve_member_switcher,
    update_member_switcher,

    upload_file,
    retrieve_file,
    delete_file,

    Member,
)
from server.models.student import (
    ErrorResponseModel,
    ResponseModel
)


router = APIRouter()


@router.post("/pictures", response_description="Member adding pictures to Firestore Storage Bucket")
async def add_pictures(Member: Member, id: str,  files: List[UploadFile] = File(...), folder: str = "media/images", content_type: str = "image/jpeg"):
    if files != None and len(files) > 0:
        try:
            pics_idx = 0
            member = await retrieve_member_switcher(Member.value, id)   #TIMEOUT EXCEPTION
            filenames = ["" for _ in range(len(files))]
            if member:
                pics = member['pics']
                face_pics_or = [list(i.keys())[0] for i in pics]
                for i in range(len(files)): 
                    if files[i] != None or files[i] != "":
                        #Swithing between realtime cropping and precropped pics
                        filenames[i] = files[i].filename
                        if filenames[i] in face_pics_or:
                            continue
                        file_byts = files[i].file.read()
                        upld = await upload_file(file_byts, filenames[i], folder, content_type)     #TIMEOUT EXCEPTION
                        if upld != False:
                            pic_url = await retrieve_file(filenames[i])
                            pics.append({filenames[i]: pic_url})
                            pics_idx += 1
                        continue
                    continue
                await update_member_switcher(Member.value, id, {"pics": pics})

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
                "Student with id {0} doesn't exist".format(id)
            )
        except (Exception, RuntimeError, TimeoutError) as err:
            return ErrorResponseModel( 
                "An error occured updating the member's pictures",
                404, 
                str(err))
    else: 
        return ErrorResponseModel(
        "An error occurred", 
        404, 
        "No File specified"
    )
@router.get("/pictures", response_description="Member retrieving all pictures from Firestore Storage Bucket")
async def retrieve_pictures(Member: Member, folder: str = "media/images"):
    try:
        members = await retrieve_members_switcher(Member.value)     #TIMEOUT EXCEPTION
        if members:
                pics = list()
                pic_names = list()
                for member in members:
                    pic_names = [list(i.keys())[0] for i in member['pics']]
                    for pic in pic_names:
                        pic_url = await retrieve_file(pic, folder)
                        pics.append(pic_url)
                        
                return ResponseModel(pics, "Pics data retrieved successfully")
        return ResponseModel(members, "Empty list returned")
    except (Exception, RuntimeError, TimeoutError) as err:
            return ErrorResponseModel( 
                "An error occured while retrieving all the members' pictures",
                404, 
                str(err))
    

@router.get("/{id}/pictures/{filename}", response_description="Member retrieving a picture from Firestore Storage Bucket")
async def retrieve_picture(Member: Member, id: str, filename: str, folder: str = "media/images"):
    try:
        member = await retrieve_member_switcher(Member.value, id, True)
        if member:
                if filename:
                    pic_url = await retrieve_file(filename, folder)     #TIMEOUT EXCEPTION
                    return pic_url

                return ErrorResponseModel(
                    "Retrive operation failed",
                    404,
                    "File does not exist!",
                )
        return ErrorResponseModel("An error occurred", 404, "Member doesn't exist.")

    except (Exception, RuntimeError, TimeoutError) as err:
            return ErrorResponseModel( 
                "An error occured while retrieving the picture",
                404, 
                str(err))

@router.delete("/{id}/pictures/{filename}", response_description="Member deleting a picture from Firestore Storage Bucket")
async def delete_picture(Member: Member, id: str, filename: str, folder: str = "media/images"):
    try:
        member = await retrieve_member_switcher(Member.value, id)   #TIMEOUT EXCEPTION
        if member:
                pics = member['pics']
                pic_names = [list(i.keys)(0) for i in pics] if len(pics)>0 else list()
                if filename in pic_names:
                    await delete_file(filename, folder)     #TIMEOUT EXCEPTION
                    pics = list(filter(lambda i : list(i.keys())[0] != filename, pics)) #Deleting file dictionary from database
                    await update_member_switcher(Member.value, id, {"pics": pics})
                    return ResponseModel(
                        "Operation was successful",
                        f"{filename} deleted!"
                    )
                return ErrorResponseModel(
                    f"Delete operation failed!!",
                    404,
                    "File does not exist!",
                )
        return ErrorResponseModel("An error occurred", 404, "Member doesn't exist.")

    except (Exception, RuntimeError, TimeoutError) as err:
            return ErrorResponseModel( 
                "An error occured deleting the picture",
                404, 
                str(err))