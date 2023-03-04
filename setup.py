import os.path as osp
from setuptools import find_packages, setup

def readme():
    with open('README.md') as f:
        content = f.read()
    return content

def find_version():
    version_file = 'src/torchblocks/version.py'
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
    keywords=["NLP", "Deep Learning", "Transformers", "PyTorch",'Natural Language Processing'],
    license="MIT License",
    platforms='Linux',
    url="https://github.com/lonePatient/TorchBlocks",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=get_requirements(),
    python_requires=">=3.7.0",
)
