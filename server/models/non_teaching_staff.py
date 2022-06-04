from typing import Optional

from pydantic import BaseModel, Field

class NonTeachingStaffSchema(BaseModel):
    fullname: str = Field(...)
    work_id: str = Field(...)
    department: Optional[str] = Field(None, title = "Staff Department")
    occupation: str = Field(...)
    pics: list = Field([])
    embeddings: list = Field([])

    class config:
        schema_extra={
            "example": {
                "fullname": "Brian Sentinel",
                "work_id": "scm211-0001/2018",
                "department": "Bsc Maths and Computer Science",
                "occupation": "Student",
                "pics": [[123,456,789]],
                "embeddings": []
            }
        }


class UpdateNonTeachingStaffModel(BaseModel):
    fullname: Optional[str]
    work_id: Optional[str]
    department: Optional[str]
    occupation: Optional[str]
    pics: Optional[list]
    embeddings: Optional[list]

    class config:
        schema_extra={
            "example": {
                "fullname": "Brian Sentinel",
                "work_id": "scm211-0001/2018",
                "department": "Bsc Maths and Computer Science",
                "occupation": "Student",
                "pics": [[123,456,977]],
                "embeddings": []
            }
        }


def ResponseModel(data, message):
    return {
        "data": [data],
        "code": 200,
        "message": message
    }

def ErrorResponseModel(error, code, message):
    return {
        "error": error,
        "code": code,
        "message": message
    }