from typing import Optional

from pydantic import BaseModel, Field

class TeachingStaffSchema(BaseModel):
    fullname: str = Field(...)
    work_id: str = Field(...)
    department: Optional[str] = Field(None, title = "Staff Department")
    occupation: str = Field(...)
    pics: list = Field([])
    embeddings: list = Field([])
    augmentations: list = Field([])

    class config:
        schema_extra={
            "example": {
                "fullname": "Brian Sentinel",
                "work_id": "scm211-0001/2018",
                "department": "Bsc Maths and Computer Science",
                "occupation": "Lecturer",
                "pics": [],
                "embeddings": [],
                "augmentations": []
            }
        }


class UpdateTeachingStaffModel(BaseModel):
    fullname: Optional[str]
    work_id: Optional[str]
    department: Optional[str]
    occupation: Optional[str]
    pics: Optional[list]
    embeddings: Optional[list]
    augmentations: Optional[list]

    class config:
        schema_extra={
            "example": {
                "fullname": "Brian Sentinel",
                "work_id": "scm211-0001/2018",
                "department": "Bsc Maths and Computer Science",
                "occupation": "Lecturer",
                "pics": [],
                "embeddings": [],
                "augmentations": []
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