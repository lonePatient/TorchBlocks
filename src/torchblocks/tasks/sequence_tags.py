import torch
import numpy as np


def get_spans_from_bios_tags(tags, id2label=None):
    """Gets entities from sequence.
    note: BIOS
    Args:
        tags (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> tags = ['B-PER', 'I-PER', 'O', 'S-LOC']
        >>> get_spans_from_bios_tags(tags)
        # output: [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(tags):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_spans_from_biob_tags(seq, id2label=None):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_spans_from_biob_tags(seq)
        #output
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_spans_from_bio_tags(tags, id2label=None):
    """Gets entities from sequence.
    Args:
        tags (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> tags = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_spans_from_bio_tags(tags)
        # output [['PER', 0,1]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(tags):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def generate_bio_tags_from_spans(entities, offset_mapping):
    """Generate NER-tags (with BIO naming) for subword tokens from the entities.
    Args:
        entities: The list of entities which consist of an entity name with its offset
            mappings.
        offset_mapping: The list of offsets which are positions of the tokens.
    Returns:
        A list of NER-tags encoded from the given entity informations.
    """
    ner_tags = ["O" for _ in offset_mapping]  # [):左闭右开
    for entity_tag, entity_start, entity_end, *_ in sorted(entities, key=lambda x: x[1]):
        current_ner_tag = f"B-{entity_tag}"
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if min(entity_end, token_end) - max(entity_start, token_start) > 0:
                ner_tags[i] = current_ner_tag
                current_ner_tag = f"I-{entity_tag}"
    return ner_tags


def build_ner_bio_conditional_masks(id2label):
    """Build a NER-conditional mask matrix which implies the relations between
    before-tag and after-tag.

    According to the rule of BIO-naming system, it is impossible that `I-Dog` cannot be
    appeard after `B-Dog` or `I-Dog` tags. This function creates the calculable
    relation-based conditional matrix to prevent from generating wrong tags.
    Args:
        id2label: A dictionary which maps class indices to their label names.
    Returns:
        A conditional mask tensor.
    """
    conditional_masks = torch.zeros(len(id2label), len(id2label))
    for i, before in id2label.items():
        for j, after in id2label.items():
            if after == "O" or after.startswith("B-") or after == f"I-{before[2:]}":
                conditional_masks[i, j] = 1.0
    return conditional_masks


def ner_beam_search_decode(log_probs, id2label, beam_size=2):
    """Decode NER-tags from the predicted log-probabilities using beam-search.

    This function decodes the predictions using beam-search algorithm. Because all tags
    are predicted simultaneously while the tags have dependencies of their previous
    tags, the greedy algorithm cannot decode the tags properly. With beam-search, it is
    possible to prevent the below situation:

        >>> sorted = probs[t].sort(dim=-1)
        >>> print("\t".join([f"{id2label[i]} {p}" for p, i in zip()]))
        I-Dog 0.54  B-Cat 0.44  ...
        >>> sorted = probs[t + 1].sort(dim=-1)
        >>> print("\t".join([f"{id2label[i]} {p}" for p, i in zip()]))
        I-Cat 0.99  I-Dog 0.01  ...

    The above shows that if the locally-highest tags are selected, then `I-Dog, I-Dog`
    will be generated even the confidence of the second tag `I-Dog` is significantly
    lower than `I-Cat`. It is more natural that `B-Cat, I-Cat` is generated rather than
    `I-Dog, I-Dog`. The beam-search for NER-tagging task can solve this problem.
    Args:
        log_probs: The log-probabilities of the token predictions.
        id2label: A dictionary which maps class indices to their label names.
        beam_size: The number of candidates for each search step. Default is `2`.

    Returns:
        A tuple of beam-searched indices and their probability tensors.
    """
    # Create the log-probability mask for the invalid predictions.
    log_prob_masks = -10000.0 * (1 - build_ner_bio_conditional_masks(id2label))
    log_prob_masks = log_prob_masks.to(log_probs.device)
    beam_search_shape = (log_probs.size(0), beam_size, log_probs.size(1))
    searched_tokens = log_probs.new_zeros(beam_search_shape, dtype=torch.long)
    searched_log_probs = log_probs.new_zeros(beam_search_shape)
    searched_scores = log_probs.new_zeros(log_probs.size(0), beam_size)
    searched_scores[:, 1:] = -10000.0

    for i in range(log_probs.size(1)):
        # Calculate the accumulated score (log-probabilities) with excluding invalid
        # next-tag predictions.
        scores = searched_scores.unsqueeze(2)
        scores = scores + log_probs[:, i, :].unsqueeze(1)
        scores = scores + (log_prob_masks[searched_tokens[:, :, i - 1]] if i > 0 else 0)
        # Select the top-k (beam-search size) predictions.
        best_scores, best_indices = scores.flatten(1).topk(beam_size)
        best_tokens = best_indices % scores.size(2)
        best_log_probs = log_probs[:, i, :].gather(dim=1, index=best_tokens)
        # best_buckets = best_indices.div(scores.size(2), rounding_mode="floor") # pytorch>=1.10.0+
        best_buckets = best_indices.floor_divide(scores.size(2))  # pytorch<1.10.0+
        best_buckets = best_buckets.unsqueeze(2).expand(-1, -1, log_probs.size(1))

        # Gather the best buckets and their log-probabilities.
        searched_tokens = searched_tokens.gather(dim=1, index=best_buckets)
        searched_log_probs = searched_log_probs.gather(dim=1, index=best_buckets)

        # Update the predictions by inserting to the corresponding timestep.
        searched_scores = best_scores
        searched_tokens[:, :, i] = best_tokens
        searched_log_probs[:, :, i] = best_log_probs

    # Return the best beam-searched sequence and its probabilities.
    return searched_tokens[:, 0, :], searched_log_probs[:, 0, :].exp()


def get_spans_from_subword_bio_tags(ner_tags, offset_mapping, probs=None):
    """Extract the entities from NER-tagged subword tokens.
    This function detects the entities from BIO NER-tags and collects them with
    averaging their confidences (prediction probabilities). Using the averaged
    probabilities, you can filter the low-confidence entities.
    Args:
        ner_tags: The list of subword-token-level NER-tags.
        offset_mapping: The list of offsets which are positions of the tokens.
        probs: An optional prediction probabilities of the subword tokens. Default is
            `None`.
    Returns:
        A tuple of collected NER entities with their averaged entity confidencs
        (prediction probabilities).
    """
    probs = probs if probs is not None else np.zeros(offset_mapping.shape[0])
    entities, gathered_probs, entity, i = [], [], None, None
    for j, ner_tag in enumerate(ner_tags):
        if entity is not None and ner_tag != f"I-{entity}":
            entities.append((entity, offset_mapping[i][0], offset_mapping[j - 1][1]))
            gathered_probs.append(probs[i:j].mean())
            entity = None
        if ner_tag.startswith("B-"):
            entity, i = ner_tag[2:], j
    # Because BIO-naming does not ensure the end of the entities (i.e. E-tag), we cannot
    # automatically detect the end of the last entity in the above loop.
    if entity is not None:
        entities.append((entity, offset_mapping[i][0], offset_mapping[-1][1]))
        gathered_probs.append(probs[i:].mean())
    return entities, gathered_probs


TYPE_TO_SCHEME = {
    "BIO": get_spans_from_bio_tags,
    "BIOS": get_spans_from_bios_tags,
    'BIOB': get_spans_from_biob_tags,
}


def get_scheme(scheme_type):
    if scheme_type not in TYPE_TO_SCHEME:
        msg = ("There were expected keys in the `TYPE_TO_SCHEME`: "
               f"{', '.join(list(TYPE_TO_SCHEME.keys()))}, "
               f"but get {scheme_type}."
               )
        raise TypeError(msg)
    scheme_function = TYPE_TO_SCHEME[scheme_type]
    return scheme_function


if __name__ == "__main__":
    sentence = {'id': '0',
                'text': '大于三十岁的与临时居住地是松陵村东区473号楼的初婚东乡族同网吧的学历为高中的外来务工人员',
                'entities': [['AGE', 0, 5, '大于三十岁'],
                             ['EDU', 36, 38, '高中'],
                             ['TAG', 39, 45, '外来务工人员'],
                             ['PER', 13, 23, '松陵村东区473号楼']],
                'intent': 'KBQA'
                }
    entities = sentence['entities']
    # BertTokenizerFast:output,return_offsets_mapping=True
    # 需要注意：[CLS][SEP]特殊符号
    offset_mapping = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
                      (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 20),
                      (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
                      (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
                      (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45)]
    print(generate_bio_tags_from_spans(entities, offset_mapping))
    '''
    outputs:['B-AGE', 'I-AGE', 'I-AGE', 'I-AGE', 'I-AGE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-EDU', 'I-EDU', 'O', 'B-TAG', 'I-TAG', 'I-TAG', 'I-TAG', 'I-TAG', 'I-TAG']
    '''
