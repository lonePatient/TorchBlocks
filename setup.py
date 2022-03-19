import os.path as osp
from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        content = f.read()
    return content

def find_version():
    version_file = 'torchblocks/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

def get_requirements(filename='requirements.txt'):
    here = osp.dirname(osp.realpath(__file__))
    with open(osp.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

setup(
    name="torchblocks",
    version=find_version(),
    author="lonePatient",
    author_email="liuweitangmath@163.com",
    description="A PyTorch-based toolkit for natural language processing",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=["NLP", "Deep Learning", "Transformers", "PyTorch"],
    license="MIT",
    url="https://github.com/lonePatient/TorchBlocks",
    packages=find_packages("torchblocks"),
    install_requires=get_requirements(),
    python_requires=">=3.7.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
