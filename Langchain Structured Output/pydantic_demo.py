from pydantic import BaseModel,Field
from typing import Optional

class Student(BaseModel):
    #name:str
    #name : str = 'nikhil' -------->you can set default val
    age:Optional[int] = None
    cgpa:float = Field(gt=0,lt=10,description="CGPA must be between 0 and 10")
    #you can set limit w the help of Field class

new_student =  {'age':21,'cgpa':9}

student = Student(**new_student)

print(student)