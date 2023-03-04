## TorchBlocks

(Work in Progress!!)

A PyTorch-based toolkit for natural language processing

![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch_1.10+-ee4c2c?logo=pytorch&logoColor=white)
![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)
![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

### Requirements

- torch>=1.10.0
- tokenizers >= 0.7.0
- transformers>=4.10.0
- torchmetrics>=0.11.3


TorchBlocks requires Python 3.7+. We recommend installing TorchBlocks in a Linux or OSX environment.

### Installation

Recommended (because of active development):

```shell
git clone https://github.com/lonePatient/TorchBlocks.git
cd TorchBlocks
python setup.py install
```
⚠️**Note:** This project is still in the development stage and some of the interfaces are subject to change.

### Tutorials

* Tutorial 1 (text classification): [task_text_classification_cola.py](https://github.com/lonePatient/TorchBlocks/blob/master/examples/task_text_classification_cola.py)
* Tutorial 2 (siamese similarity): [task_siamese_similarity_afqmc.py](https://github.com/lonePatient/TorchBlocks/blob/master/examples/task_siamese_similarity_afqmc.py)
* Tutorial 3 (sequence labeling): [task_sequence_labeling_ner_crf.py](https://github.com/lonePatient/TorchBlocks/blob/master/examples/task_sequence_labeling_ner_crf.py)
* Tutorial 4 (sentence similarity): [task_sentence_similarity_lcqmc.py](https://github.com/lonePatient/TorchBlocks/blob/master/examples/task_sentence_similarity_lcqmc.py)
* Tutorial 5 (triple similarity): [task_triple_similarity_epidemic.py](https://github.com/lonePatient/TorchBlocks/blob/master/examples/task_triple_similarity_epidemic.py)
* Tutorial 6 (sequence labeling): [task_sequence_labeling_resume_beam_search_softmax.py](https://github.com/lonePatient/TorchBlocks/blob/master/examples/task_sequence_labeling_resume_beam_search_softmax.py)
* Tutorual 7 (sequence labeling): [task_sequence_labeling_resume_global_pointer.py](https://github.com/lonePatient/TorchBlocks/blob/master/examples/task_sequence_labeling_resume_global_pointer.py)
* Example scripts for each task: [TorchBlocks/examples/](https://github.com/lonePatient/TorchBlocks/tree/master/examples)

