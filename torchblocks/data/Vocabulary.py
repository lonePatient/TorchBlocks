import os
import logging
from collections import Counter, OrderedDict

logger = logging.getLogger(__name__)

VOCAB_NAME = "vocab.txt"


class Vocabulary(object):
    def __init__(self,
                 pad_token="[PAD]",
                 unk_token="[UNK]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]",
                 max_size=None,
                 min_freq=None,
                 add_unused=False):

        self.max_size = max_size
        self.min_freq = min_freq
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.rebuild = True
        self.add_unused = add_unused
        self.word_counter = Counter()
        self.tokens_to_ids = OrderedDict()
        self.ids_to_tokens = OrderedDict()

    def reset(self):
        ctrl_symbols = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]
        for index, syb in enumerate(ctrl_symbols):
            self.tokens_to_ids[syb] = index

        if self.add_unused:
            for i in range(100):
                self.tokens_to_ids[f'[UNUSED{i}]'] = len(self.tokens_to_ids)

    def __len__(self):
        return len(self.tokens_to_ids)

    def update(self, word_list):
        '''
        依次增加序列中词在词典中的出现频率
        :param word_list:
        :return:
        '''
        self.word_counter.update(word_list)

    def add(self, word):
        '''
        增加一个新词在词典中的出现频率
        :param word:
        :return:
        '''
        self.word_counter[word] += 1

    def has_word(self, word):
        '''
        检查词是否被记录
        :param word:
        :return:
        '''
        return word in self.tokens_to_ids

    def to_index(self, word):
        '''
        将词转为数字. 若词不再词典中被记录, 将视为 unknown, 若 ``unknown=None`` , 将抛出
        :param word:
        :return:
        '''
        if word in self.tokens_to_ids:
            return self.tokens_to_ids[word]
        if self.unk_token is not None:
            return self.tokens_to_ids[self.unk_token]
        else:
            raise ValueError("word {} not in vocabulary".format(word))

    def unknown_idx(self):
        """
        unknown 对应的数字.
        """
        if self.unk_token is None:
            return None
        return self.tokens_to_ids[self.unk_token]

    def padding_idx(self):
        """
        padding 对应的数字
        """
        if self.pad_token is None:
            return None
        return self.tokens_to_ids[self.pad_token]

    def to_word(self, idx):
        """
        给定一个数字, 将其转为对应的词.

        :param int idx: the index
        :return str word: the word
        """
        return self.ids_to_tokens[idx]

    def build_vocab(self):
        self.reset()
        max_size = min(self.max_size, len(self.word_counter)) if self.max_size else None
        words = self.word_counter.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self.tokens_to_ids:
            words = filter(lambda kv: kv[0] not in self.tokens_to_ids, words)
        start_idx = len(self.tokens_to_ids)
        self.tokens_to_ids.update({w: i + start_idx for i, (w, _) in enumerate(words)})
        self.rebuild = False

    def build_reverse_vocab(self):
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.tokens_to_ids.items()])

    def clear(self):
        """
        删除Vocabulary中的词表数据。相当于重新初始化一下。
        :return:
        """
        self.word_counter.clear()
        self.tokens_to_ids = None
        self.ids_to_words = None
        self.rebuild = True
        self.reset()

    def save_vocab(self, vocab_path):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.

        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.

        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.tokens_to_ids.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            self.tokens_to_ids[token] = index
