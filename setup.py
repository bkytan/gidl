import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
requirementPath = here + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(name="gidl", version='1.0', install_requires=install_requires, packages=find_packages())