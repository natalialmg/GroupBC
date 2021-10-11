
from pathlib import Path
import pickle

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def unpickle(file):
    with open(file, 'rb') as fo:
        mydict = pickle.load(fo, encoding='bytes')
    return mydict