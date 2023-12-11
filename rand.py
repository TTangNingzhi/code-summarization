from utils import *
from tqdm import tqdm
from config import *


def test(data):
    avg_precision, avg_recall, avg_bleu, avg_meteor, avg_rouge_l = 0, 0, 0, 0, 0
    for i in tqdm(range(len(data))):
        fun, com = data[i]
        k = get_k(language, len(set(fun)))
        k_words = np.random.choice(list(set(fun)), k, replace=False)
        # print("Sampled keywords:", ' '.join(k_words))
        metrics = calculate_metrics(com, k_words)
        avg_precision += metrics['precision']
        avg_recall += metrics['recall']
        avg_bleu += metrics['bleu']
        avg_meteor += metrics['meteor']
        avg_rouge_l += metrics['rouge-l']
    return {'precision': avg_precision / len(data), 'recall': avg_recall / len(data), 'bleu': avg_bleu / len(data),
            'meteor': avg_meteor / len(data), 'rouge-l': avg_rouge_l / len(data)}


if __name__ == '__main__':
    train_data = read_parallel(get_data_path(language, 'train'), 0, 0)
    dev_data = read_parallel(get_data_path(language, 'dev'), 0, 0)
    test_data = read_parallel(get_data_path(language, 'test'), 0, 0)

    print("Training...")

    print("Dev...")
    results = test(dev_data)
    print("Precision: {:.4f}, Recall: {:.4f}, BLEU: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}".format(
        results['precision'], results['recall'], results['bleu'], results['meteor'], results['rouge-l']))

    print("Testing...")
    results = test(test_data)
    print("Precision: {:.4f}, Recall: {:.4f}, BLEU: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}".format(
        results['precision'], results['recall'], results['bleu'], results['meteor'], results['rouge-l']))
