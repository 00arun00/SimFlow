# coding=utf-8

from setuptools import find_packages
from setuptools import setup

with open("README.md", 'r') as fh:
    long_description = fh.read()


setup(
    name='simflow',
    version='0.0.1',
    description='simflow',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
    ],
    url='https://github.com/00arun00/SimFlow',
    author="Arun Joseph",
    author_email="arunjoseph.eng@gmail.com"
)
