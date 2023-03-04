import unicodedata


class TokenTextMapping:
    def __init__(self):
        pass

    def _is_control(self, ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    def lowercase_and_normalize(self, text):
        """转小写，并进行简单的标准化
        """
        text = text.lower()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
        return text

    def stem(self, token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def _is_special(self, ch):
        """判断是不是有特殊含义的符号
        """
        special = ['[CLS]', '[SEP]', '[PAD]']
        # special = ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
        if ch in special:
            return True
        else:
            return False

    def __call__(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系"""
        text = text.lower()
        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = self.lowercase_and_normalize(ch)
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))
        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            elif token == '[unused1]' or token == '[UNK]':
                start = offset
                end = offset + 1
                token_mapping.append(char_mapping[start:end])
                offset = end
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end
        return token_mapping


if __name__ == "__main__":
    tokens = ['[CLS]', '大', '于', 'book', '##es', '岁', '的', '与', '临', '时', '居', '住', '[SEP]']
    text = '大于bookes岁的与临时居住'
    TokenTextMapping()(text, tokens)
    '''
    result:
         [[],
         [0],
         [1],
         [2, 3, 4, 5],
         [6, 7],
         [8],
         [9],
         [10],
         [11],
         [12],
         [13],
         [14],
         []]
    '''
