import codecs
import os
from setuptools import setup, find_packages

# these things are needed for the README.md show on pypi (if you dont need delete it)
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


setup(
    name="pyfoam2sdf",
    version="1.1",
    description="A Python package for converting foamMesh data to Signed Distance Fields (SDF).",
    author="ZHANG Rushan",
    author_email="rzhangbq@gmail.com",
    url="https://github.com/srl-ethz/Foam-mesh-reading",
    long_description=long_description,
    packages=find_packages(),
    license='MIT',
    install_requires=[
        "numpy"
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
