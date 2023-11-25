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


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_embed = nn.Embedding(len(src_vocab), d_model)
        self.tgt_embed = nn.Embedding(len(tgt_vocab), d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=1700)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          dropout, batch_first=True)
        self.out = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.src_embed(src) * math.sqrt(self.transformer.d_model)
        tgt = self.tgt_embed(tgt) * math.sqrt(self.transformer.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask,
                                  memory_key_padding_mask)
        output = self.out(output)
        return output

    def translate(self, src_sentence, tgt_max_len=200):
        self.eval()
        src_tokens = [self.src_vocab.numberize(token) for token in src_sentence.split()]
        src_tokens = torch.tensor(src_tokens)
        tgt_tokens = [self.tgt_vocab.numberize('<bos>')]

        for i in range(tgt_max_len):
            tgt_tensor = torch.tensor(tgt_tokens)
            output = self(src_tokens.unsqueeze(0), tgt_tensor.unsqueeze(0))
            next_token = torch.argmax(output[0, -1]).item()
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
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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
        translated_sentence = model.translate(' '.join(fun))
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

    out_dir = 'out/' + language + '/transformer'

    fun_vocab = Vocab()
    com_vocab = Vocab()
    for fun, com in tqdm(train_data):
        fun_vocab |= fun
        com_vocab |= com

    train_dataset = TranslationDataset(train_data, fun_vocab, com_vocab)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=train_dataset.collate_fn,
                              generator=torch.Generator(device=device))

    model = Transformer(fun_vocab, com_vocab, d_model=256, nhead=1, num_encoder_layers=4, num_decoder_layers=1,
                        dim_feedforward=256, dropout=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=com_vocab.numberize('<pad>'))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for src, tgt in tqdm(train_loader):
            optimizer.zero_grad()
            output = model(src.T, tgt.T,
                           tgt_mask=model.transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device),
                           src_key_padding_mask=(src.T == fun_vocab.numberize('<pad>')),
                           tgt_key_padding_mask=(tgt.T == com_vocab.numberize('<pad>')))
            loss = criterion(output.view(-1, output.size(-1)), tgt.T.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

        for fun, com in dev_data[:10]:
            print("Source: ", ' '.join(fun))
            print("Target: ", ' '.join(com))
            print("Translation: ", ' '.join(model.translate(' '.join(fun))))
            print()

        out_file = open(os.path.join(out_dir, f'dev.{epoch}.out'), 'w')
        results = test(dev_data, model, out_file)
        print("Dev - Precision: {:.4f}, Recall: {:.4f}, BLEU: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}".format(
            results['precision'], results['recall'], results['bleu'], results['meteor'], results['rouge-l']))

    out_file = open(os.path.join(out_dir, f'test.out'), 'w')
    test(test_data, model, out_file)
    print("Test - Precision: {:.4f}, Recall: {:.4f}, BLEU: {:.4f}, METEOR: {:.4f}, ROUGE-L: {:.4f}".format(
        results['precision'], results['recall'], results['bleu'], results['meteor'], results['rouge-l']))
