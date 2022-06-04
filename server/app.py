from fastapi import FastAPI

from server.routes.student import router as StudentRouter
from server.routes.teaching_staff import router as TeachingStaffRouter
from server.routes.non_teaching_staff import router as NonTeachingStaffRouter
from server.routes.visitor import router as VisitorRouter
from server.routes.storage import router as StorageRouter
from server.routes.fr_model import router as Fr_ModelRouter

app = FastAPI()

app.include_router(StudentRouter, tags=['student'], prefix='/student')
app.include_router(TeachingStaffRouter, tags=['teaching_staff'], prefix='/teaching_staff')
app.include_router(NonTeachingStaffRouter, tags=['non_teaching_staff'], prefix='/non_teaching_staff')
app.include_router(VisitorRouter, tags=['visitor'],prefix='/visitor')

app.include_router(StorageRouter, tags=['storage'], prefix='/storage')
app.include_router(Fr_ModelRouter, tags=['fr_model'], prefix='/fr_model')


@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to this fantastic APP!😃"}