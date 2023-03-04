from torchblocks.utils import json_to_text
from sklearn.model_selection import StratifiedKFold


def get_data(data_path, datatype):
    data = []
    if datatype == 'train':
        with open(data_path) as f:
            for i in f:
                dict_txt = eval(i)
                if dict_txt['query'] == '':
                    continue
                for j in dict_txt['candidate']:
                    if j['text'] == '':
                        continue
                    data.append({'query': dict_txt['query'], 'candidate': j['text'], 'label': j['label']})
    else:
        with open(data_path) as f:
            for i in f:
                dict_txt = eval(i)
                for j in dict_txt['candidate']:
                    data.append({'text_id': dict_txt['text_id'], 'query': dict_txt['query'], 'candidate': j['text']})
    return data


def generate_data(train_data, random_state=42):
    X = range(len(train_data))
    y = [x['label'] for x in train_data]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    for fold, (train_index, dev_index) in enumerate(skf.split(X, y)):
        tmp_train_df = [train_data[index] for index in train_index]
        tmp_dev_df = [train_data[index] for index in dev_index]
        json_to_text(f'../dataset/ccks2021/ccks2021_train_seed{random_state}_fold{fold}.json', tmp_train_df)
        json_to_text(f'../dataset/ccks2021/ccks2021_dev_seed{random_state}_fold{fold}.json', tmp_dev_df)


if __name__ == '__main__':
    seed = 42
    train_path1 = '../dataset/ccks2021/round1_train.txt'
    train_path2 = '../dataset/ccks2021/round2_train.txt'
    train_data1 = get_data(train_path1, 'train')
    train_data2 = get_data(train_path2, 'train')
    train_data = train_data1
    train_data.extend(train_data2)
    generate_data(train_data, 42)
    generate_data(train_data, 24)
    generate_data(train_data, 33)
    print('...............kf finish...........')
