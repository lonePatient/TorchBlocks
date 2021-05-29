"""自定义分别tokenzier"""
from transformers import BertTokenizer


class CNTokenizer(BertTokenizer):
    '''
    基于char形式
    '''
    def __init__(self, vocab_file, delimiter='',  # 分隔符
                 unk_token='[UNK]', do_lower_case=False, **kwargs):
        super().__init__(vocab_file=str(vocab_file), do_lower_case=do_lower_case, **kwargs)
        self.vocab_file = str(vocab_file)
        self.delimiter = delimiter
        self.unk_token = unk_token

    def tokenize(self, text, **kwargs):
        if not isinstance(text, str):  # 必须str，否则不经过自定义tokenize，
            raise ValueError(f"'text' value type: expected to be str,but get {type(text)}")
        if self.do_lower_case:
            text = text.lower()
        if self.delimiter == '':
            w_list = list(text)
        else:
            w_list = text.split(self.delimiter)
        _tokens = []
        for c in w_list:
            if c in self.vocab:
                _tokens.append(c)
            else:
                _tokens.append(self.unk_token)
        return _tokens
