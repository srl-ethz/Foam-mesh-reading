from setuptools import setup, find_packages

setup(
    name="pyfoam2sdf",
    version="0.1",
    description="A Python package for converting foamMesh data to Signed Distance Fields (SDF).",
    author="ZHANG Rushan",
    author_email="rzhangbq@gmail.com",
    url="https://github.com/srl-ethz/Foam-mesh-reading",
    packages=find_packages(),
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
