from setuptools import find_packages, setup
setup(
    name="torchblocks",
    version="1.0.0",
    author="lonePatient",
    author_email="liuweitangmath@163.com",
    description="A PyTorch-based toolkit for natural language processing",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch BERT",
    license="MIT",
    url="https://github.com/lonePatient/TorchBlocks",
    package_dir={"": "torchblocks"},
    packages=find_packages("torchblocks"),
    install_requires=[
        "scikit-learn",
        "tokenizers >= 0.7.0",
        "torch >= 1.6.0",
        "transformers >= 4.1.1",
        "sentencepiece",
        "sacremoses",
    ],
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)


