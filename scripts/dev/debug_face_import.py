import sys, traceback
sys.path.insert(0, "src")

print("--- Testing models.face ---")
try:
    from models.face import Face
    print("Face OK")
except Exception as e:
    print("Face ERROR:")
    traceback.print_exc()

print("--- Testing models.face_person ---")
try:
    from models.face_person import FacePerson
    print("FacePerson OK")
except Exception as e:
    print("FacePerson ERROR:")
    traceback.print_exc()
