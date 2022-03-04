def get_spans_bios(tags, id2label=None):
    """Gets entities from sequence.
    note: BIOS
    Args:
        tags (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> tags = ['B-PER', 'I-PER', 'O', 'S-LOC']
        >>> get_spans_bios(tags)
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


def get_spans_bio(tags, id2label=None):
    """Gets entities from sequence.
    Args:
        tags (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> tags = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_spans_bio(tags)
        # output [['PER', 0,1], ['LOC', 3, 3]]
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


TYPE_TO_SCHEME = {
    "BIO": get_spans_bio,
    "BIOS": get_spans_bios,
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
