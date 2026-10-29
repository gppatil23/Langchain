from langchain_text_splitters import RecursiveCharacterTextSplitter,Language
text = '''
class Student:
    def _init_(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0

# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())
'''
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap = 0

)
chunks = splitter.split_text(text)
print(chunks[0])
print(chunks[1])