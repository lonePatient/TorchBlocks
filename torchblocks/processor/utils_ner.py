def remove_bio_from_label_name(label_name):
    if label_name[:2] in ['B-', 'I-', 'E-', 'S-']:
        new_label_name = label_name[2:]
    else:
        assert (label_name == 'O')
        new_label_name = label_name
    return new_label_name

def end_current_entity(previous_label_without_bio, current_entity_length, new_labels, i):
    '''
    Helper function for bio_to_bioes
    '''
    if current_entity_length == 0:
        return
    if current_entity_length == 1:
        new_labels[i - 1] = 'S-' + previous_label_without_bio
    else:  # elif current_entity_length > 1
        new_labels[i - 1] = 'E-' + previous_label_without_bio

def bio_to_bioes(labels):
    previous_label_without_bio = 'O'
    current_entity_length = 0
    new_labels = labels.copy()
    for i, label in enumerate(labels):
        label_without_bio = remove_bio_from_label_name(label)
        # end the entity
        if current_entity_length > 0 and (
                label[:2] in ['B-', 'O'] or label[:2] == 'I-' and previous_label_without_bio != label_without_bio):
            end_current_entity(previous_label_without_bio, current_entity_length, new_labels, i)
            current_entity_length = 0
        if label[:2] == 'B-':
            current_entity_length = 1
        elif label[:2] == 'I-':
            if current_entity_length == 0:
                new_labels[i] = 'B-' + label_without_bio
            current_entity_length += 1
        previous_label_without_bio = label_without_bio
    end_current_entity(previous_label_without_bio, current_entity_length, new_labels, i + 1)
    return new_labels

def bioes_to_bio(labels):
    previous_label_without_bio = 'O'
    new_labels = labels.copy()
    for i, label in enumerate(labels):
        label_without_bio = remove_bio_from_label_name(label)
        if label[:2] in ['I-', 'E-']:
            if previous_label_without_bio == label_without_bio:
                new_labels[i] = 'I-' + label_without_bio
            else:
                new_labels[i] = 'B-' + label_without_bio
        elif label[:2] in ['S-']:
            new_labels[i] = 'B-' + label_without_bio
        previous_label_without_bio = label_without_bio
    return new_labels

