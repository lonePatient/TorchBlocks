from torchblocks.utils import json_to_text
from torchblocks.tasks import get_spans_from_bio_tags
from torchblocks.data.splits import split_ner_stratified_kfold

'''
采用多标签方式进行划分数据
'''

train_file = '../dataset/cner/train.char.bmes'
dev_file = '../dataset/cner/dev.char.bmes'
folds = 5
sentences = []
lines = []
for input_file in [train_file, dev_file]:
    with open(input_file, 'r') as f:
        words, labels = [], []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    lines.append([words, labels])
                    words, labels = [], []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    label = splits[-1].replace("\n", "")
                    if 'M-' in label:
                        label = label.replace('M-', 'I-')
                    elif 'E-' in label:
                        label = label.replace('E-', 'I-')
                    elif 'S-' in label:  # 去除S标签，主要方便后面做实验
                        label = "O"
                    labels.append(label)
                else:
                    labels.append("O")
        if words:
            lines.append([words, labels])

for i, (words, labels) in enumerate(lines):
    spans = get_spans_from_bio_tags(labels, id2label=None)
    new_spans = []
    for span in spans:
        tag, start, end = span
        new_spans.append([tag, start, end + 1, "".join(words[start:(end + 1)])])
    sentence = {'id': i, 'text': words, 'entities': new_spans, 'bio_seq': labels}
    sentences.append(sentence)

entities_list = [x['entities'] for x in sentences]
all_indices = split_ner_stratified_kfold(entities_list, num_folds=5)
for fold, (train_indices, val_indices) in enumerate(all_indices):
    print("The number of train examples: ",len(train_indices))
    print("The number of dev examples: ", len(val_indices))
    train_data = [sentences[i] for i in train_indices]
    dev_data = [sentences[i] for i in val_indices]
    json_to_text(f'../dataset/cner/cner_train_fold{fold}.json', train_data)
    json_to_text(f'../dataset/cner/cner_dev_fold{fold}.json', dev_data)

