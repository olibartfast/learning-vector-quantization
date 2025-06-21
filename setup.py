#!/usr/bin/env python3
"""
Setup script for LVQ Python bindings
"""

import os
import sys
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        "lvq_python",
        sources=[
            "python_bindings/lvq_python.cpp",
            "src/lvq_network.cpp",
            "src/lvq_trainer.cpp", 
            "src/lvq_tester.cpp",
            "src/data_loader.cpp",
            "src/utils.cpp"
        ],
        include_dirs=[
            "include",
            get_pybind_include(),
            get_pybind_include(user=True),
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-Wall', '-Wextra', '-O2'],
    ),
]

setup(
    name="lvq-python",
    version="1.0.0",
    author="LVQ Network Team",
    author_email="",
    description="Python bindings for Learning Vector Quantization (LVQ) Network",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    keywords="machine learning, neural networks, lvq, vector quantization, classification",
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
) 