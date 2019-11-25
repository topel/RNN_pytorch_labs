import torch
import torch.nn.functional as F
import numpy as np
import math
import os.path
import re

#############################
# PTB
#############################
def check_dataset_exists(path_data='../'):
    flag_idx2word = os.path.isfile(path_data + 'librivox_fr/idx2word.pt') 
    flag_test_data = os.path.isfile(path_data + 'librivox_fr/test_data.pt') 
    flag_train_data = os.path.isfile(path_data + 'librivox_fr/train_data.pt') 
    flag_word2idx = os.path.isfile(path_data + 'librivox_fr/word2idx.pt') 
    if flag_idx2word==False or flag_test_data==False or flag_train_data==False or flag_word2idx==False:
        print('Librivox_fr dataset manquant')
#         data_folder = 'librivox_fr/data_raw'
#         corpus = Corpus(path_data+data_folder)
#         batch_size=20
#         train_data = batchify(corpus.train, batch_size)
#         val_data = batchify(corpus.valid, batch_size)
#         test_data = batchify(corpus.test, batch_size)
#         vocab_size = len(corpus.dictionary)
#         torch.save(train_data,path_data + 'librivox_fr/train_data.pt')
#         torch.save(test_data,path_data + 'librivox_fr/test_data.pt')
#         torch.save(corpus.dictionary.idx2word,path_data + 'librivox_fr/idx2word.pt')
#         torch.save(corpus.dictionary.word2idx,path_data + 'librivox_fr/word2idx.pt')
    return path_data


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
        
        
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


path_data = '../'
_ = check_dataset_exists(path_data)

word2idx  =  torch.load(path_data + 'librivox_fr/word2idx.pt')
idx2word  =  torch.load(path_data + 'librivox_fr/idx2word.pt')





def normalize_gradient(net):

    grad_norm_sq=0

    for p in net.parameters():
        grad_norm_sq += p.grad.data.norm()**2

    grad_norm=math.sqrt(grad_norm_sq)
   
    if grad_norm<1e-4:
        net.zero_grad()
        print('norme du gradient proche de zéro')
    else:
        for p in net.parameters():
             p.grad.data.div_(grad_norm)

    return grad_norm


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('Nombre de paramètres du réseau : {} ({:.2f} millions)'.format(
        nb_param, nb_param/1e6)
         )


def sentence2vector_librivox_fr(sentence):
    words = sentence.split()
    x = torch.LongTensor(len(words),1)
    for idx, word in enumerate(words):
        word = re.sub("'", "_", word)
        if word not in word2idx:
            print('Vous avez entrer un mot hors-vocabulaire.')
            print('--> Enlever lettres majuscules et ponctuation')
            print("mot --> <unk> avec index 0")
            x[idx,0]=0            
        else:
            x[idx,0]=word2idx[word]
    return x


def show_next_word(scores):
    num_word_display=30
    prob=F.softmax(scores,dim=2)
    p=prob[-1].squeeze()
    p,word_idx = torch.topk(p,num_word_display)

    for i,idx in enumerate(word_idx):
        percentage= p[i].item()*100
        word=  idx2word[idx.item()]
        print(  "{:.1f}%\t".format(percentage),  word ) 

