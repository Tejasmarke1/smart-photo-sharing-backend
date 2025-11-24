import sys, traceback
sys.path.insert(0, "src")

print("---- Testing import: models.face ----")
try:
    from models.face import Face
    print("SUCCESS: Face imported cleanly")
except Exception:
    print("ERROR importing Face:")
    traceback.print_exc()
