from fastapi import APIRouter, Body
from fastapi.encoders import jsonable_encoder

from server.database import (
    add_visitor,
    delete_visitor,
    retrieve_visitor,
    retrieve_visitors,
    update_visitor,
)

from server.models.visitor import (
    ErrorResponseModel,
    ResponseModel,
    VisitorSchema,
    UpdateVisitorModel,
)

router = APIRouter()

@router.post("/", response_description="Visitor data added into the database")
async def add_visitor_data(visitor: VisitorSchema = Body(...)):
    visitor = jsonable_encoder(visitor)
    new_visitor = await add_visitor(visitor)
    
    return ResponseModel(new_visitor,"Visitor added successfully")

@router.get("/", response_description="Visitors retrieved")
async def get_visitors():
    visitors = await retrieve_visitors()
    
    if visitors:
        return ResponseModel(visitors, "Visitors data retrieved successfully")
    
    return ResponseModel(visitors, "Empty list returned")

@router.get("/{id}", response_description="Visitor data retrieved")
async def get_visitor_data(id):
    visitor = await retrieve_visitor(id)
    
    if visitor:
        return ResponseModel(visitor, "Visitor data retrieved successfully")
    
    return ErrorResponseModel("An error occurred", 404, "Visitor doesn't exist.")

@router.put("/{id}")
async def update_visitor_data(id: str,req:UpdateVisitorModel = Body(...)):
    req = {k: v for k,v in req.dict().items() if v is not None}
    updated_visitor = await update_visitor(id, req)
    
    if updated_visitor:
        return ResponseModel(
            "Visitor with ID: {} name update is successful".format(id),
            "Visitor name updated successfully",
        )
        
    return ErrorResponseModel(
        "An error occured",
        404,
        "There was an error updating the visitor data."
    )
    
@router.delete("/{id}",response_description="Visitor data deleted from the database")
async def delete_visitor_data(id: str):
    deleted_visitor = await delete_visitor(id)
    if deleted_visitor:
        return ResponseModel(
            "Visitor with ID: {} removed".format(id), 
            "Visitor deleted successfully"
        )
    
    return ErrorResponseModel(
        "An error occurred", 
        404, 
        "Visitor with id {0} doesn't exist".format(id)
    )
    