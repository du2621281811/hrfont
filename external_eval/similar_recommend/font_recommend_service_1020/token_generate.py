import os
import hashlib

if __name__ == "__main__":
    hashlib.sha1(os.urandom(20)).hexdigest()
