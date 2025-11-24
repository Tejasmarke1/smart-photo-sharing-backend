import os, sys
sys.path.insert(0, "src")  # ensure local src is first on path
import models
print("Models imported from:", os.path.abspath(models.__file__))
print("Current working dir:", os.path.abspath("."))
