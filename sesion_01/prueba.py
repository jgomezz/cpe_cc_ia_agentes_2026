from dotenv import load_dotenv
load_dotenv()
import os

name = os.getenv("NAME","Unknown")

print("Hello, World! :", name)