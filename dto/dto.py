from pydantic import BaseModel

# Defining input / output classes of api requests
class UserRequestIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    output_text: str
