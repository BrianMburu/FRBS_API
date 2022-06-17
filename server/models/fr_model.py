from typing import Optional

from pydantic import BaseModel, Field

class FrModelSchema(BaseModel):
    Train_Score: str = Field(...)
    Test_Score: str = Field(...)
    Train_Time: str = Field(...)
    Data_Size: dict = Field(...)
    Neighbours: str = Field(...)

    class config:
        schema_extra={
            "example": {
                "Train_Score": "98",
                "Train_Time": "12-6-2022 8:30 Pm",
                "Data_Size":  {"X":'500',"y":'500'},
                "Neighbours": "20",
            }
        }


class UpdateFrModelSchema(BaseModel):
    Train_Score: Optional[str]
    Test_Score: Optional[str]
    Train_Time: Optional[str]
    Data_Size: Optional[dict]
    Neighbours: Optional[list]

    class config:
        schema_extra={
            "example": {
                "Train_Score": "98",
                "Train_Time": "12-6-2022 8:30 Pm",
                "Data_Size":  {"X":'500',"y":'500'},
                "Neighbours": "20",
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