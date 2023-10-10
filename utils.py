import collections.abc
import nltk
from rouge import Rouge


class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""

    def __init__(self):
        words = {'<BOS>', '<EOS>', '<UNK>'}
        self.num_to_word = list(words)
        self.word_to_num = {word: num for num, word in enumerate(self.num_to_word)}

    def add(self, word):
        if word in self: return
        num = len(self.num_to_word)
        self.num_to_word.append(word)
        self.word_to_num[word] = num

    def discard(self, word):
        raise NotImplementedError()

    def update(self, words):
        self |= words

    def __contains__(self, word):
        return word in self.word_to_num

    def __len__(self):
        return len(self.num_to_word)

    def __iter__(self):
        return iter(self.num_to_word)

    def numberize(self, word):
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else:
            return self.word_to_num['<UNK>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]


def read_parallel(data_path, num_bos=1, num_eos=1):
    """Read data from the files named by `ffilename` and `efilename`.

    The files should have the same number of lines.

    Arguments:
      - ffilename: str
      - efilename: str
    Returns: list of pairs of lists of strings. <BOS> and <EOS> are added to all sentences.
    """
    ffilename, efilename = data_path
    data = []
    for (fline, eline) in zip(open(ffilename, encoding='utf-8'), open(efilename, encoding='utf-8')):
        fwords = ['<BOS>'] * num_bos + fline.lower().split() + ['<EOS>'] * num_eos
        ewords = ['<BOS>'] * num_bos + eline.lower().split() + ['<EOS>'] * num_eos
        data.append((fwords, ewords))
    return data


def get_data_path(language, split):
    return f"data/{language}/{split}/code.original_subtoken", f"data/{language}/{split}/javadoc.original"


def get_k(language, fun_len):
    if language == 'java':
        # k = min(max(round(0.0147 * fun_len + 17.72), 4), 696)
        k = round(0.0143 * fun_len + 16.0064)
    else:
        # k = min(max(round(fun_len * 9.46 / 48.09), 3), 50)
        k = round(0.0067 * fun_len + 9.1413)
    return k if k < fun_len else fun_len


def calculate_metrics(reference, hypothesis):
    # nltk.download('wordnet')
    precision = len(set(hypothesis) & set(reference)) / len(set(hypothesis))
    recall = len(set(hypothesis) & set(reference)) / len(set(reference))
    bleu = nltk.translate.bleu_score.sentence_bleu([' '.join(reference)], ' '.join(hypothesis))
    meteor = nltk.translate.meteor_score.meteor_score([reference], hypothesis)
    rouge_l = Rouge().get_scores(' '.join(hypothesis), ' '.join(reference))[0]['rouge-l']['f']
    return {'precision': precision, 'recall': recall, 'bleu': bleu, 'meteor': meteor, 'rouge-l': rouge_l}
