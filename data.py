# Modify from https://github.com/S-Abdelnabi/awt
import os
import torch

from collections import Counter
from transformers import RobertaTokenizer


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        if os.path.exists(os.path.join(path, "token.txt")):
            with open(os.path.join(path, "token.txt"), "r") as f:
                self.tokens = set([x.strip() for x in f.readlines()])
                for t in self.tokens:
                    self.dictionary.add_word(t)
        else:
            self.tokens = None
        # self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    
    def split(self, line):
        result = line.split()
        if self.tokens is not None:
            result = [x if x in self.tokens else "<unk>" for x in result]
        return result
    
        # result = [x.replace("Ċ", "").replace("Ġ", "") for x in self.tokenizer.tokenize(line)]
        # return [x for x in result if x]
        
        # words = []
        # index = 0
        # while index < len(line):
        #     word = ""
        #     if line[index].strip() == "":
        #         index += 1
        #         continue
        #     if not line[index].isalnum() and not line[index] == "_":
        #         word += line[index]
        #         index += 1
        #     else:
        #         while index < len(line) and (line[index].isalnum() or line[index] == "_"):
        #             word += line[index]
        #             index += 1
        #     words.append(word)
        # return words

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = self.split(line)# + ['<eos>']
                tokens += len(words)
                for word in words:
                    if word != "Ċ":
                        self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = self.split(line)# + ['<eos>']
                for word in words:
                    if word != "Ċ":
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

        return ids
