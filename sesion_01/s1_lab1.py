from dotenv import load_dotenv
import os

load_dotenv()

name = os.getenv("NAME","Unknown")

path = os.getenv("PATH","Unknown")

print("NAME:", name)
print("PATH:", path)