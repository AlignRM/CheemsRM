import setuptools
from setuptools import setup

with open("./README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='cheems',
    version='0.0.1',
    packages=setuptools.find_packages(),
    license='Apache-2.0',
    author='Wenxueru',
    author_email='wenxueru2022@iscas.ac.cn',
    description='Code for \"Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch\".',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)
