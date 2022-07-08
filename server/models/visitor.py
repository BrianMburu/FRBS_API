from typing import Optional

from pydantic import BaseModel, Field


class VisitorSchema(BaseModel):
    fullname: str = Field(...)
    index: str = Field(...)
    Appointment: str = Field(...)
    Approved: bool = Field(False,title="Appointment Status")

    class config:
        schema_extra={
            "example": {
                "fullname": "Brian Sentinel",
                "index": "hsmax2301",
                "Appointment": "Vice Chancelor Visit",
                "Approved": True
            }
        }
class UpdateVisitorModel(BaseModel):
    fullname: Optional[str]
    index: Optional[str]
    Appointment: Optional[str]
    Approved: Optional[bool]

    class config:
        schema_extra={
            "example": {
                "fullname": "Brian Sentinel",
                "index": "hsmax2302",
                "Appointment": "Vice Chancelor Visit",
                "Approved": True
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