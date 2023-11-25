import collections.abc
import nltk
from rouge import Rouge


class Vocab(collections.abc.MutableSet):
    """Set-like data structure that can change words into numbers and back."""

    def __init__(self):
        words = {'<bos>', '<eos>', '<unk>', '<pad>'}
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
            return self.word_to_num['<unk>']

    def denumberize(self, num):
        """Convert a number into a word."""
        return self.num_to_word[num]


def read_parallel(data_path, num_bos=1, num_eos=1, max_fun_len=1600, max_com_len=205):
    # print("max fun len:", max_fun_len, "max com len:", max_com_len)
    ffilename, efilename = data_path
    data = []
    for (fline, eline) in zip(open(ffilename, encoding='utf-8'), open(efilename, encoding='utf-8')):
        fwords = ['<bos>'] * num_bos + fline.lower().split() + ['<eos>'] * num_eos
        ewords = ['<bos>'] * num_bos + eline.lower().split() + ['<eos>'] * num_eos
        if len(fwords) > max_fun_len or len(ewords) > max_com_len:
            continue
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
    try:
        precision = len(set(hypothesis) & set(reference)) / len(set(hypothesis))
        recall = len(set(hypothesis) & set(reference)) / len(set(reference))
        bleu = nltk.translate.bleu_score.sentence_bleu([' '.join(reference)], ' '.join(hypothesis))
        meteor = nltk.translate.meteor_score.meteor_score([reference], hypothesis)
        rouge_l = Rouge().get_scores(' '.join(hypothesis), ' '.join(reference))[0]['rouge-l']['f']
        return {'precision': precision, 'recall': recall, 'bleu': bleu, 'meteor': meteor, 'rouge-l': rouge_l}
    except:
        print("Error in calculating metrics", reference, hypothesis)
        return {'precision': 0, 'recall': 0, 'bleu': 0, 'meteor': 0, 'rouge-l': 0}
