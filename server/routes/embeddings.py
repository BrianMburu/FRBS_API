import numpy as np
from fastapi import APIRouter

from server.database import (
    retrieve_member_switcher,
    update_member_switcher,
    
    Member,
)

from server.utils import(
    Facenet,
    get_embedding,
    url_to_image,
)

from server.models.student import (
    ErrorResponseModel,
    ResponseModel,
)

router = APIRouter()


@router.put("/{id}",response_description="Update Member's Face Embeddings")
async def update_member_face_embeddings(Member: Member, id: str):
    #Retrieve a single member widh id
    member = await retrieve_member_switcher(Member.value, id, False)

    if member:
        pics = member["pics"]
        augmentations = member["augmentations"]
        embeddings = member["embeddings"]
        
        #Face Embeddings
        face_embd_ks = [list(i.keys())[0] for i in embeddings] if len(embeddings)>0 else list()
        
        #Original Pictures
        face_urls_or = [list(i.values())[0] for i in pics] if len(pics)>0 else list()
        face_pics_or = [list(i.keys())[0] for i in pics] if len(pics)>0 else list()
        faces_or = [await url_to_image(url) for url in face_urls_or] if len(pics)>0 else list() 
        
        #Augmented Pictures
        face_urls_ag = [list(i.values())[0] for i in augmentations] if len(augmentations)>0 else list()
        face_pics_ag = [list(i.keys())[0] for i in augmentations] if len(augmentations)>0 else list()
        faces_ag = [await url_to_image(url) for url in face_urls_ag] if len(augmentations)>0 else list()
            
        #combined original and Augmented data
        face_pics = face_pics_or + face_pics_ag
        faces = faces_or + faces_ag
        
        emb_idx = 0

        if len(faces)>0:
            assert len(faces) == len(face_pics) 

            for i in range(len(faces)):
                face_pixels = faces[i]
                pic_n = face_pics[i]

                if pic_n in face_embd_ks:
                    continue

                else:
                    face_pixels = np.array(face_pixels) # convert face pixels to arrays
                    emb = get_embedding(Facenet(), face_pixels).tolist() #fetching embeddings for given face pixels

                    embeddings = embeddings+[{pic_n: emb}]
                    await update_member_switcher(Member.value, id, {"embeddings": embeddings}) #update member embeddings
                    emb_idx += 1
                    
            return ResponseModel(
                f"{emb_idx} Embeddings added",
                "Embeddings update was successfull.",
            )

        return ResponseModel(
                "0 Embeddings added",
                "Embeddings update was successfull.",
            )

    return ErrorResponseModel(
        "An error occured updating the member's picture embeddings data",
        404,
        "Member does not exist!!"
    )

@router.delete("/{id}",response_description="Format all Member's Face Embeddings")
async def delete_embeddings(Member: Member, id: str, filename: str = None , all: bool = False):
    member = await retrieve_member_switcher(Member.value, id, False)
    if member:
        if all:
            member_name = str(member["fullname"])
            await update_member_switcher(Member.value, id,{"embeddings": []})
            return ResponseModel(
                f"All Face Embeddings for {member_name} Deleted",
                "Embeddings update was successfull.",
            )
        if filename != None:
            embeddings = member['embeddings']
            face_embd_dts = [i for i in embeddings] if len(embeddings)>0 else list()
            embeddings =  list(filter(lambda i : list(i.keys())[0] != filename, face_embd_dts))

            await update_member_switcher(Member.value, id, {"embeddings": embeddings})
            
            return ResponseModel(
                "Operation was successful",
                f"Embeddings for {filename} deleted!"
            )

    return ErrorResponseModel(
        "An error occured updating the member's picture embeddings data",
        404,
        "Member does not exist!!"
    )