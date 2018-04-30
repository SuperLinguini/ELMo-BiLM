import os
import numpy as np

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.data import Counter, WikiText2
from gluonnlp.data.dataset import SimpleDataset
from gluonnlp.data.utils import slice_sequence
from gluonnlp import Vocab


class WikiText2Character(WikiText2):
    """WikiText-2 word-level dataset for language modeling, from Salesforce research.

    From
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test'.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default '<eos>'
        The token to add at the end of each sentence. If None, nothing is added.
    root : str, default '~/.mxnet/datasets/wikitext-2'
        Path to temp folder for storing data.
    """
    def __init__(self, segment='train', skip_empty=True, bos='<bos>', eos='<eos>',
                 root=os.path.join('~', '.mxnet', 'datasets', 'wikitext-2')):
        super(WikiText2Character, self).__init__(segment, skip_empty, bos, eos, root)

    def batchify(self, vocab, batch_size, max_word_length=50, load=None):
        """Transform the dataset into N independent sequences, where N is the batch size.

        Parameters
        ----------
        vocab : gluonnlp.Vocab
            The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
            index according to the vocabulary.
        batch_size : int
            The number of samples in each batch.

        Returns
        -------
        NDArray of shape (num_tokens // N, N). Excessive tokens that don't align along
        the batches are discarded.
        """
        if load:
            return mx.nd.load('train_data')
        else:
            data = self._data[0]
            sample_len = len(data) // batch_size
            # return np.array(data[:sample_len*batch_size], dtype=object).reshape(batch_size, -1).T
            return vocab.dataset_to_char_ids(data[:sample_len*batch_size], batch_size, sample_len, max_word_length).swapaxes(0, 1), mx.nd.array(vocab[data[:sample_len*batch_size]]).reshape(batch_size, -1).T

    # def bptt_batchify(self, vocab, seq_len, batch_size, last_batch='keep', load_path=None, max_word_length=50, lazy=True):
    #     """Transform the dataset into batches of numericalized samples, in the way that the
    #     recurrent states from last batch connects with the current batch for each sample.
    #
    #     Each sample is of shape `(seq_len, batch_size)`. When `last_batch='keep'`, the first
    #     dimension of last sample may be shorter than `seq_len`.
    #
    #     Parameters
    #     ----------
    #     vocab : gluonnlp.Vocab
    #         The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
    #         index according to the vocabulary.
    #     seq_len : int
    #         The length of each of the samples for truncated back-propagation-through-time (TBPTT).
    #     batch_size : int
    #         The number of samples in each batch.
    #     last_batch : {'keep', 'discard'}
    #         How to handle the last batch if the remaining length is less than `seq_len`.
    #
    #         keep - A batch with less samples than previous batches is returned.
    #         discard - The last batch is discarded if its incomplete.
    #     """
    #     if load_path:
    #         batches = mx.nd.load(load_path)
    #     else:
    #         data = self.batchify(vocab, batch_size)
    #         batches = slice_sequence(data, seq_len+1, overlap=1)
    #         if last_batch == 'keep':
    #             sample_len = len(self._data[0]) // batch_size
    #             has_short_batch = _slice_pad_length(sample_len*batch_size, seq_len+1, 1) > 0
    #             if has_short_batch:
    #                 batches.append(data[seq_len*len(batches):, :])
    #     # TODO: +1 for masks
    #     return SimpleDataset(batches).transform(lambda x:
    #                                             (vocab.array_to_char_ids(x[:min(len(x)-1, seq_len), :], max_word_length),
    #                                              mx.nd.array(vocab[x[1:, :]])))


def _slice_pad_length(num_items, length, overlap=0):
    """Calculate the padding length needed for sliced samples in order not to discard data.

    Parameters
    ----------
    num_items : int
        Number of items in dataset before collating.
    length : int
        The length of each of the samples.
    overlap : int, default 0
        The extra number of items in current sample that should overlap with the
        next sample.

    Returns
    -------
    Length of paddings.

    """
    if length <= overlap:
        raise ValueError('length needs to be larger than overlap')

    step = length-overlap
    span = num_items-length
    residual = span % step
    if residual:
        return step - residual
    else:
        return 0


class UnicodeCharsVocabulary(Vocab):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    """
    def __init__(self, counter=None, max_word_length=50, max_size=None, min_freq=1, unknown_token='<unk>',
                 padding_token='<pad>', bos_token='<bos>', eos_token='<eos>', reserved_tokens=None):
        super(UnicodeCharsVocabulary, self).__init__(counter, max_size, min_freq, unknown_token, padding_token,
                                                     bos_token, eos_token, reserved_tokens)
        self._max_word_length = max_word_length

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260 # <padding>

        if counter:
            self.num_words = self.__len__()

            self._word_char_ids = np.zeros([self.num_words, max_word_length],
                dtype=np.int32)

            # the charcter representation of the begin/end of sentence characters
            def _make_bos_eos(c):
                r = np.zeros([self.max_word_length], dtype=np.int32)
                r[:] = self.pad_char
                r[0] = self.bow_char
                r[1] = c
                r[2] = self.eow_char
                return r
            self.bos_chars = _make_bos_eos(self.bos_char)
            self.eos_chars = _make_bos_eos(self.eos_char)

            for i, word in enumerate(self._token_to_idx):
                self._word_char_ids[i] = self._convert_word_to_char_ids(word)

            self._word_char_ids[self._token_to_idx[self.bos_token]] = self.bos_chars
            self._word_char_ids[self._token_to_idx[self.eos_token]] = self.eos_chars

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def size(self):
        return self.num_words

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = self.pad_char

        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)]
        code[0] = self.bow_char
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        code[k + 1] = self.eow_char

        return code

    def word_to_char_ids(self, word):
        if word in self._token_to_idx:
            return self._word_char_ids[self._token_to_idx[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def array_to_char_ids(self, input_array, max_word_length):
        char_array = mx.nd.full((input_array.shape[0], input_array.shape[1], max_word_length), self.pad_char)

        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                word = input_array[i][j]
                if word in self._token_to_idx:
                    char_array[i][j] = self._word_char_ids[self._token_to_idx[word]]
                else:
                    word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length - 2)]
                    char_array[i][j][0] = self.bow_char
                    for k, chr_id in enumerate(word_encoded, start=1):
                        char_array[i][j][k] = chr_id
                    char_array[i][j][k + 1] = self.eow_char

        # TODO: Check what to do about masks
        char_array += 1
        return char_array

    def dataset_to_char_ids(self, dataset, batch_size, sample_len, max_word_length):
        char_dataset = mx.nd.full((batch_size, sample_len, max_word_length), self.pad_char)

        for i, word in enumerate(dataset):
            if word in self._token_to_idx:
                char_dataset[i // sample_len][i % sample_len] = self._word_char_ids[self._token_to_idx[word]]
            else:
                word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length - 2)]
                char_dataset[i // sample_len][i % sample_len][0] = self.bow_char
                for k, chr_id in enumerate(word_encoded, start=1):
                    char_dataset[i // sample_len][i % sample_len][k] = chr_id
                char_dataset[i // sample_len][i % sample_len][k + 1] = self.eow_char

        # TODO: Check what to do about masks
        char_dataset += 1

        return char_dataset

    def encode_chars(self, sentence, reverse=False, split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])

    def __getitem__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.

        If `unknown_token` of the vocabulary is None, looking up unknown tokens results in KeyError.

        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        if isinstance(tokens, (list, tuple)):
            return [self._token_to_idx[token] for token in tokens]
        elif isinstance(tokens, np.ndarray):
            vfunc = np.vectorize(self._token_to_idx.__getitem__)
            return vfunc(tokens)
        else:
            return self._token_to_idx[tokens]


# train_dataset, val_dataset, test_dataset = [WikiText2Character(segment=segment,
#                                                                bos='<bos>', eos='<eos>',
#                                                                skip_empty=False)
#                                             for segment in ['train', 'val', 'test']]
#
# vocab = UnicodeCharsVocabulary(nlp.data.Counter(train_dataset[0]), 50)
#
# val_data, test_data = [x.bptt_batchify(vocab, 35, 80,
#                                                    last_batch='keep', load_path=None)
#                                    for x in [val_dataset, test_dataset]]
# val_data[0]
# print('hi')