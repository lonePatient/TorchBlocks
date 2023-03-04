from collections import Counter
from .ml_stratifiers import MultilabelStratifiedKFold


def split_ner_stratified_kfold(entities_list, num_folds, shuffle=True, random_state=42):
    """Split to the NER entity-level stratified k-folds.
    Args:
        entities_list: The list of list of entities.
        num_folds: The number of folds to split.
        fold_index: The index of current fold.
    Returns:
        A tuple of indices for train and validation.
    """
    # Collect the entity types and sort them for deterministics.
    entity_types = sorted({y for x in entities_list for y, *_ in x})
    # Count the entity appearances and transform to vectors for the multilabel k-fold.
    entity_counts = [Counter(y for y, *_ in x) for x in entities_list]
    entity_labels = [[cnt[x] for x in entity_types] for cnt in entity_counts]
    kfold = MultilabelStratifiedKFold(num_folds, shuffle=shuffle, random_state=random_state)
    return list(kfold.split(entities_list, entity_labels))


if __name__ == "__main__":
    sentences = [{'id': '0',
                  'text': '大于三十岁的与临时居住地是松陵村东区473号楼的初婚东乡族同网吧的学历为高中的外来务工人员',
                  'entities': [['AGE', 0, 5, '大于三十岁'],
                               ['EDU', 36, 38, '高中'],
                               ['TAG', 39, 45, '外来务工人员'],
                               ['PER', 13, 23, '松陵村东区473号楼']],
                  'intent': 'KBQA'
                  },
                 {'id': '1',
                  'text': '大于三十岁的与临时居住地是松陵村东区473号楼的初婚东乡族同网吧的学历为高中的外来务工人员',
                  'entities': [['AGE', 0, 5, '大于三十岁'],
                               ['EDU', 36, 38, '高中'],
                               ['TAG', 39, 45, '外来务工人员'],
                               ['PER', 13, 23, '松陵村东区473号楼']],
                  'intent': 'KBQA'
                  },
                 {'id': '2',
                  'text': '大于三十岁的与临时居住地是松陵村东区473号楼的初婚东乡族同网吧的学历为高中的外来务工人员',
                  'entities': [['AGE', 0, 5, '大于三十岁'],
                               ['EDU', 36, 38, '高中'],
                               ['TAG', 39, 45, '外来务工人员'],
                               ['PER', 13, 23, '松陵村东区473号楼']],
                  'intent': 'KBQA'
                  },
                 {'id': '3',
                  'text': '大于三十岁的与临时居住地是松陵村东区473号楼的初婚东乡族同网吧的学历为高中的外来务工人员',
                  'entities': [['AGE', 0, 5, '大于三十岁'],
                               ['EDU', 36, 38, '高中'],
                               ['TAG', 39, 45, '外来务工人员'],
                               ['PER', 13, 23, '松陵村东区473号楼']],
                  'intent': 'KBQA'
                  },
                 {'id': '4',
                  'text': '大于三十岁的与临时居住地是松陵村东区473号楼的初婚东乡族同网吧的学历为高中的外来务工人员',
                  'entities': [['AGE', 0, 5, '大于三十岁'],
                               ['EDU', 36, 38, '高中'],
                               ['TAG', 39, 45, '外来务工人员'],
                               ['PER', 13, 23, '松陵村东区473号楼']],
                  'intent': 'KBQA'
                  }
                 ]
    entities_list = [x['entities'] for x in sentences]
    data_indices = split_ner_stratified_kfold(entities_list, num_folds=5)
    '''
    output:
    [(array([0, 1, 2, 3]), array([4])),
     (array([0, 1, 2, 4]), array([3])),
     (array([1, 2, 3, 4]), array([0])),
     (array([0, 1, 3, 4]), array([2])),
     (array([0, 2, 3, 4]), array([1]))]
    '''
