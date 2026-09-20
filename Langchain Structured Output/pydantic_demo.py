from pydantic import BaseModel

class Student(BaseModel):
    name:str
    #name : str = 'nikhil' -------->you can set default val

new_student =  {'name':'nikhil'}

student = Student(**new_student)

print(student)