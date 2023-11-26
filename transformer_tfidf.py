from config import *
from utils import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import math
import os

sample_method = 'topk'  # 'topk', 'ancestral', 'greedy'


class TransformerTFIDF(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, tfidf, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(TransformerTFIDF, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_embed = nn.Embedding(len(src_vocab), d_model)
        self.tgt_embed = nn.Embedding(len(tgt_vocab), d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=1700)
        self.tfidf = tfidf
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout, batch_first=True)
        self.out = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src_weight = torch.zeros_like(src, dtype=torch.float32)
        for i in range(src.size(0)):
            src_weight[i] = self.tfidf.compute_tfidf_weight(src[i])
        src = self.src_embed(src) * math.sqrt(self.transformer.d_model) * src_weight.unsqueeze(-1)
        tgt = self.tgt_embed(tgt) * math.sqrt(self.transformer.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask,
                                  memory_key_padding_mask)
        output = self.out(output)
        return output

    def translate(self, src_sentence, tgt_max_len=200):
        src_sentence = ['<bos>'] + src_sentence.split() + ['<eos>']
        src_tokens = [self.src_vocab.numberize(token) for token in src_sentence]
        src_tokens = torch.tensor(src_tokens).unsqueeze(0)
        tgt_tokens = [self.tgt_vocab.numberize('<bos>')]

        for i in range(tgt_max_len):
            tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0)
            tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(tgt_tensor.device)
            output = self(src_tokens, tgt_tensor, tgt_mask=tgt_mask)

            # beam search with size 4 and length penalty 0.6 (?)
            if sample_method == 'topk':
                topk = torch.topk(output[0, -1, :], k=100)
                next_token = topk[1][torch.multinomial(torch.softmax(topk[0], dim=-1), num_samples=1).item()].item()
            elif sample_method == 'ancestral':
                next_token = torch.multinomial(torch.softmax(output[0, -1, :], dim=-1), num_samples=1).item()
            elif sample_method == 'greedy':
                next_token = torch.argmax(output[0, -1, :]).item()
            else:
                next_token = self.tgt_vocab.numberize('<eos>')

            tgt_tokens.append(next_token)
            if next_token == self.tgt_vocab.numberize('<eos>'):
                break

        translated_sentence = [self.tgt_vocab.denumberize(token) for token in tgt_tokens[1:-1]]
        return translated_sentence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


class TFIDF:
    def __init__(self, corpus, vocab):
        self.corpus = corpus
        self.vocab = vocab
        self.idf = self._compute_idf()

    def _compute_idf(self):
        df = torch.zeros(len(self.vocab), dtype=torch.float32)
        for doc in tqdm(self.corpus):
            for word in set(doc):
                df[self.vocab.numberize(word)] += 1
        df[self.vocab.numberize('<pad>')] = len(self.corpus)
        idf = torch.log(len(self.corpus) / (df + 1))
        return idf

    def _compute_tf(self, numbered_doc):
        tf = torch.bincount(torch.tensor(numbered_doc), minlength=len(self.vocab)).float()
        tf = tf / len(numbered_doc)
        return tf

    def _compute_tfidf(self, numbered_doc):
        tf = self._compute_tf(numbered_doc)
        tfidf = tf * self.idf
        return tfidf

    def compute_tfidf_weight(self, numbered_doc):
        tfidf = self._compute_tfidf(numbered_doc)
        weight = tfidf[numbered_doc]
        weight = weight / weight.mean()
        weight.requires_grad = False
        return weight


class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = torch.tensor([self.src_vocab.numberize(token) for token in src])
        tgt = torch.tensor([self.tgt_vocab.numberize(token) for token in tgt])
        return src, tgt

    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_batch = pad_sequence(src_batch, padding_value=self.src_vocab.numberize('<pad>'))
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.tgt_vocab.numberize('<pad>'))
        return src_batch, tgt_batch


def test(data, model, out=None):
    avg_precision, avg_recall, avg_bleu, avg_meteor, avg_rouge_l = 0, 0, 0, 0, 0
    for i in tqdm(range(len(data))):
        fun, com = data[i]
        translated_sentence = model.translate(' '.join(fun), tgt_max_len=100)
        metrics = calculate_metrics(com, translated_sentence)
        avg_precision += metrics['precision']
        avg_recall += metrics['recall']
        avg_bleu += metrics['bleu']
        avg_meteor += metrics['meteor']
        avg_rouge_l += metrics['rouge-l']
        if out is not None:
            print(' '.join(translated_sentence), file=out)
    return {'precision': avg_precision / len(data), 'recall': avg_recall / len(data), 'bleu': avg_bleu / len(data),
            'meteor': avg_meteor / len(data), 'rouge-l': avg_rouge_l / len(data)}


if __name__ == '__main__':
    train_data = read_parallel(get_data_path(language, 'train'), 1, 1)
    dev_data = read_parallel(get_data_path(language, 'dev'), 0, 0)
    test_data = read_parallel(get_data_path(language, 'test'), 0, 0)

    out_dir = 'out/' + language + '/transformer-tfidf/' + sample_method

    fun_vocab = Vocab()
    com_vocab = Vocab()
    for fun, com in tqdm(train_data):
        fun_vocab |= fun
        com_vocab |= com
    fun_corpus = [_[0] for _ in train_data]
    tfidf_fun = TFIDF(fun_corpus, fun_vocab)

    train_dataset = TranslationDataset(train_data, fun_vocab, com_vocab)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn,
                              generator=torch.Generator(device=device))
    dev_dataset = TranslationDataset(dev_data, fun_vocab, com_vocab)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False, collate_fn=dev_dataset.collate_fn)

    model = TransformerTFIDF(fun_vocab, com_vocab, tfidf_fun, d_model=256, nhead=4, num_encoder_layers=4,
                             num_decoder_layers=1, dim_feedforward=4 * 256, dropout=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=com_vocab.numberize('<pad>'))
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)

    best_dev_loss = math.inf
    best_dev_epoch = 0

    for epoch in range(10):
        model.train()
        total_loss = 0
        for src, tgt in tqdm(train_loader):
            optimizer.zero_grad()
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
            output = model(src, tgt[:, :-1],
                           tgt_mask=model.transformer.generate_square_subsequent_mask(tgt.size(1) - 1).to(tgt.device),
                           src_key_padding_mask=(src == fun_vocab.numberize('<pad>')),
                           tgt_key_padding_mask=(tgt[:, :-1] == com_vocab.numberize('<pad>')))
            tgt_for_loss = tgt[:, 1:].reshape(-1)
            output_flat = output.reshape(-1, output.size(-1))
            loss = criterion(output_flat, tgt_for_loss)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_loss += loss.item()
        print(f"Epoch {epoch}, Train Loss: {total_loss / len(train_loader)}")

        model.eval()
        with torch.no_grad():
            for fun, com in dev_data[:5]:
                print("=========================")
                print("Source: ", ' '.join(fun))
                print("Target: ", ' '.join(com))
                print("Translation: ", ' '.join(model.translate(' '.join(fun))))

            total_loss = 0
            for src, tgt in tqdm(dev_loader):
                src = src.transpose(0, 1)
                tgt = tgt.transpose(0, 1)
                output = model(src, tgt[:, :-1],
                               tgt_mask=model.transformer.generate_square_subsequent_mask(tgt.size(1) - 1).to(
                                   tgt.device),
                               src_key_padding_mask=(src == fun_vocab.numberize('<pad>')),
                               tgt_key_padding_mask=(tgt[:, :-1] == com_vocab.numberize('<pad>')))
                tgt_for_loss = tgt[:, 1:].reshape(-1)
                output_flat = output.reshape(-1, output.size(-1))
                loss = criterion(output_flat, tgt_for_loss)
                total_loss += loss.item()

            print(f"Epoch {epoch}, Dev Loss: {total_loss / len(dev_loader)}")

            if total_loss < best_dev_loss:
                best_dev_loss = total_loss
                best_dev_epoch = epoch
                torch.save(model.state_dict(), os.path.join(out_dir, f'dev.{epoch}.pt'))
                print(f"Saved model at epoch {epoch}.")

    model.eval()
    print("Loading best model at epoch {}".format(best_dev_epoch))
    model.load_state_dict(torch.load(os.path.join(out_dir, f'dev.{best_dev_epoch}.pt')))
    with torch.no_grad():
        out_file = open(os.path.join(out_dir, f'test.out'), 'w')
        results = test(test_data, model, out_file)
        print("Test - Precision: {:.4f}, Recall: {:.4f}, BLEU: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}".format(
            results['precision'], results['recall'], results['bleu'], results['meteor'], results['rouge-l']))
