from typing import Optional

from pydantic import BaseModel, Field

class StudentSchema(BaseModel):
    fullname: str = Field(...)
    reg_no: str = Field(...)
    course: Optional[str] = Field(None, title = "Student Course")
    pics: list = Field([])
    embeddings: list = Field([])
    augmentations: list = Field([])

    class config:
        schema_extra={
            "example": {
                "fullname": "Brian Sentinel",
                "reg_no": "scm211-0001/2018",
                "course": "Bsc Maths and Computer Science",
                "pics": ["example.py.jpg"],
                "embeddings": [],
                "augmentations": []
            }
        }


class UpdateStudentModel(BaseModel):
    fullname: Optional[str]
    reg_no: Optional[str]
    course: Optional[str]
    pics: Optional[list]
    embeddings: Optional[list]
    augmentations: Optional[list]

    class config:
        schema_extra={
            "example": {
                "fullname": "Brian Sentinel",
                "reg_no": "scm211-0001/2018",
                "course": "Bsc Maths and Computer Science",
                "pics": ["example.jpg"],
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