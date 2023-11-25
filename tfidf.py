from utils import *
from tqdm import tqdm
from config import *


class TFIDF:
    def __init__(self, corpus, vocab):
        self.corpus = corpus
        self.vocab = vocab
        self.idf = self._compute_idf()

    def _compute_idf(self):
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for doc in tqdm(self.corpus):
            for word in set(doc):
                idf[self.vocab.numberize(word)] += 1
        idf = np.log(len(self.corpus) / (idf + 1))
        return idf

    def _compute_tf(self, doc):
        tf = np.zeros(len(self.vocab), dtype=np.float32)
        for word in doc:
            tf[self.vocab.numberize(word)] += 1
        tf = tf / len(doc)
        return tf

    def compute_tfidf(self, doc):
        tf = self._compute_tf(doc)
        tfidf = tf * self.idf
        return tfidf


def test(data, model, vocab):
    avg_precision, avg_recall, avg_bleu, avg_meteor, avg_rouge_l = 0, 0, 0, 0, 0
    for i in tqdm(range(len(data))):
        fun, com = data[i]
        score = model.compute_tfidf(fun)
        k = get_k(language, len(set(fun)))
        topk_words = [vocab.num_to_word[idx] for idx in np.argsort(score)[-k:]]
        metrics = calculate_metrics(com, topk_words)
        avg_precision += metrics['precision']
        avg_recall += metrics['recall']
        avg_bleu += metrics['bleu']
        avg_meteor += metrics['meteor']
        avg_rouge_l += metrics['rouge-l']
    return {'precision': avg_precision / len(data), 'recall': avg_recall / len(data), 'bleu': avg_bleu / len(data),
            'meteor': avg_meteor / len(data), 'rouge-l': avg_rouge_l / len(data)}


if __name__ == '__main__':
    train_data = read_parallel(get_data_path(language, 'train'), 1, 1)
    dev_data = read_parallel(get_data_path(language, 'dev'), 0, 0)
    test_data = read_parallel(get_data_path(language, 'test'), 0, 0)

    fun_vocab = Vocab()
    com_vocab = Vocab()
    for fun, com in tqdm(train_data):
        fun_vocab |= fun
        com_vocab |= com
    fun_corpus = [_[0] for _ in train_data]

    print("Training...")
    print("Computing IDF...")
    tfidf_fun = TFIDF(fun_corpus, fun_vocab)

    print("Dev...")
    results = test(dev_data, tfidf_fun, fun_vocab)
    print("Precision: {:.4f}, Recall: {:.4f}, BLEU: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}".format(
        results['precision'], results['recall'], results['bleu'], results['meteor'], results['rouge-l']))

    print("Testing...")
    results = test(test_data, tfidf_fun, fun_vocab)
    print("Precision: {:.4f}, Recall: {:.4f}, BLEU: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}".format(
        results['precision'], results['recall'], results['bleu'], results['meteor'], results['rouge-l']))
