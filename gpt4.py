from utils import *
from tqdm import tqdm
from config import *

if __name__ == '__main__':
    data = read_parallel(get_data_path(language, 'test'), 0, 0)
    with open('out/' + language + '/gpt4/gpt4.100.txt', 'r') as f:
        preds = [line.strip().split() for line in f.readlines()]
    avg_precision, avg_recall, avg_bleu, avg_meteor, avg_rouge_l = 0, 0, 0, 0, 0
    for i in tqdm(range(len(preds))):
        fun, com = data[i]
        metrics = calculate_metrics(com, preds[i])
        avg_precision += metrics['precision']
        avg_recall += metrics['recall']
        avg_bleu += metrics['bleu']
        avg_meteor += metrics['meteor']
        avg_rouge_l += metrics['rouge-l']
    results = {'precision': avg_precision / len(preds), 'recall': avg_recall / len(preds), 'bleu': avg_bleu / len(preds),
               'meteor': avg_meteor / len(preds), 'rouge-l': avg_rouge_l / len(preds)}
    print("Precision: {:.4f}, Recall: {:.4f}, BLEU: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}".format(
        results['precision'], results['recall'], results['bleu'], results['meteor'], results['rouge-l']))
