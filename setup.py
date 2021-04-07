# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

VERSION = '1.1.0' 
DESCRIPTION = 'Tensorflow/Keras Model Profiler: Tell you model memory requirement, no. of parameters, flops etc.'

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

INSTALL_REQUIRES = [
                    'numpy',
                    'tabulate'
                    ]
# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="model_profiler", 
        version=VERSION,
        author="Talha Ilyas",
        LICENSE = 'MIT License',
        author_email="mr.talhailyas@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=INSTALL_REQUIRES, 
        
        url = 'https://github.com/Mr-TalhaIlyas/Tensorflow-Keras-Model-Profiler',
        
        keywords=['python', 'model_profile', 'gpu memory usage', 
                  'model flops', 'model parameters', 'gpu availability'
                  'mdoel memory requirement','weights memory requirement'],
        classifiers= [
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ]
)