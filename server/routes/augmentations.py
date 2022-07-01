import cv2
from fastapi import APIRouter

from server.database import (
    retrieve_member_switcher,
    update_member_switcher,

    upload_file,
    retrieve_file,
    delete_file,
    
    Member,
)

from server.utils import(
    get_augmentations,
    url_to_image,
)

from server.models.student import (
    ErrorResponseModel,
    ResponseModel,
)
router = APIRouter()

@router.put("/{id}",response_description="Update Member's Face pixels' Augmentations and Upload them to Firestore storage")
async def update_member_face_augmentations(Member: Member, id: str, pic_name: str, folder="media/augmentations"):
    #Retrieve a single member widh id
    member = await retrieve_member_switcher(Member.value, id, False)

    if member:
        augmentations = member["augmentations"]
        pics = [list(i.keys())[0] for i in member["pics"]]
        
        face_pics_ag = list()
        if len(augmentations)>0:
            face_pics_ag = [list(i.keys())[0] for i in augmentations]
        if pic_name in pics:  
            pic_url = await retrieve_file(pic_name)
            #aug_urls = list()
            #aug_names = list()
            augs_idx = 0
            if pic_url:
                pic_pixels = await url_to_image(pic_url)
                pic_augs = get_augmentations(pic_pixels)
                for i in range(len(pic_augs)):
                    im_buf_arr = cv2.imencode(".jpg", pic_augs[i])[1]
                    byte_im = im_buf_arr.tobytes()
                    img_aug_name = f"{pic_name}-aug{i}"
                    if img_aug_name in face_pics_ag:
                        continue
                    else:
                        #aug_names.append(img_aug_name)
                        uploaded = await upload_file(byte_im, img_aug_name, folder, 'image/jpeg', None)
                        if uploaded:
                            aug_url = await retrieve_file(img_aug_name, folder)
                            augmentations = augmentations + [{img_aug_name: aug_url}]
                            await update_member_switcher(Member.value, id, {"augmentations": augmentations})
                            #aug_urls.append(aug_url)
                            augs_idx += 1 
                        else:
                            continue

                if augs_idx < 3:
                    return ResponseModel(
                        f"only {augs_idx} out of 3 augmentations uploaded",
                        f"{3 - augs_idx} are broken or aleady exist!!"
                    )
                return  ResponseModel(
                    f"{augs_idx} Augmentations for {pic_name} updated successfully",
                    "Augmentations update was successfull.",
                )
                
            return ErrorResponseModel(
                "An error occured updating the member's picture Augmented data",
                404,
                "Picture url is Broken!!"
            )
        return ErrorResponseModel(
                "An error occured updating the member's picture Augmented data",
                404,
                "Picture does not exist!! in user's pictures"
            )

    return ErrorResponseModel(
        "An error occured updating the member's picture Augmented data",
        404,
        "Member does not exist!!"
    )

@router.delete("/{id}/augmentations/{filename}", response_description="Member deleting all augmentations of a picture from Firestore Storage Bucket")
async def delete_Picture_augments(Member: Member, id: str, filename: str, folder: str = "media/augmentations"):
    member = await retrieve_member_switcher(Member.value, id, False)
    if member:
        augmentations = member["augmentations"]
        pictures = member["pics"]

        aug_pic_dicts = [i for i in augmentations] if len(augmentations)>0 else list()
        or_pics = [list(i.keys())[0] for i in pictures] if len(pictures)>0 else list()

        if filename in or_pics:
            filenames = [f"{filename}-aug{i}" for i in range(3)]

            for fn in filenames:
                await delete_file(fn, folder)       #Deleting all the augmentations for the file in storageBucket
            aug_pic_dicts = list(filter(lambda i : list(i.keys())[0] not in filenames, aug_pic_dicts)) #Deleting file dictionary from database
            await update_member_switcher(Member.value, id, {"augmentations": aug_pic_dicts}) 
            
            return ResponseModel(
                "Operation was successful",
                f"{filenames} Augmentations for {filename} deleted!!"
            )
        return ErrorResponseModel(
            f"Delete operation failed!!",
            404,
            "File does not exist!",
        )
    return ErrorResponseModel("An error occurred", 404, "Member doesn't exist.")