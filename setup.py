import os
from setuptools import setup

if __name__ == "__main__":
    setup()
    os.system("jupytext --to ipynb notebooks/*.py")
