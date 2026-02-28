import os
import torch
from typing import Union, List, Tuple, Optional
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

class WordTokenizer():
    def __init__(self, min_samples=5):
        self.dictionary_encode = dict()
        self.dictionary_decode = dict()
        self.vocab_size = 0
        
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        self.min_samples = min_samples
    
    def fit(self, file_path):
        self.vocab_size = 4
        for i, key in enumerate(["<pad>", "<bos>", "<eos>", "<unk>"]):
            self.dictionary_encode[key] = i
            self.dictionary_decode[i] = key
        
        counter = dict()
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                for word in line.split(" "):
                    if word == "\n":
                        continue
                    if word not in counter:
                        counter[word] = 1
                    else:
                        counter[word] += 1
                
        for word in counter:
            if counter[word] >= self.min_samples:
                    token = word
                    self.dictionary_encode[token] = self.vocab_size
                    self.dictionary_decode[self.vocab_size] = token
                    self.vocab_size += 1
        
        assert "\n" not in self.dictionary_encode
        return self
    
    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            return list(map(lambda x: self._code_token(x, self.dictionary_encode), texts.split(" ")))
        
        res = []
        for line in texts:
            res.append(list(map(lambda x: self._code_token(x, self.dictionary_encode), line.split(" "))))
        return res
    
    def decode(self, ids: Union[List[int], List[List[int]]]):
        if isinstance(ids[0], int):
            return " ".join(list(map(lambda x: self._code_token(x, self.dictionary_decode), ids))[1:-1])
        
        res = []
        for line in ids:
            res.append(" ".join(list(map(lambda x: self._code_token(x, self.dictionary_decode), line))[1:-1]))
        return res
        
    def _code_token(self, token, dict):
        if token in dict:
            return dict[token]
        
        if isinstance(token, int):
            return "<unk>"
        return self.unk_id
    
    
class TextDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, data_file: str, target_file: Optional[str], train: bool = True, max_length: int = 128, min_samples=5):
        """
        Dataset with texts, supporting BPE tokenizer
        :param data_file: txt file containing texts
        :param train: whether to use train or validation split
        :param sp_model_prefix: path prefix to save tokenizer model
        :param vocab_size: sentencepiece tokenizer vocabulary size
        :param normalization_rule_name: sentencepiece tokenizer normalization rule
        :param model_type: sentencepiece tokenizer model type
        :param max_length: maximal length of text in tokens
        """
        
        with open(data_file, encoding="utf-8") as file:
            self.texts_de = file.readlines()
            for i in range(len(self.texts_de)):
                self.texts_de[i] = self.texts_de[i].replace("\n", "")

        if target_file is not None:
            with open(target_file, encoding="utf-8") as file:
                self.texts_en = file.readlines()
                for i in range(len(self.texts_en)):
                    self.texts_en[i] = self.texts_en[i].replace("\n", "")
        
        self.tokenizer_de = WordTokenizer(min_samples=min_samples).fit(data_file)
        self.tokenizer_en = WordTokenizer(min_samples=min_samples).fit(target_file)

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Split texts to train and validation fixing self.TRAIN_VAL_RANDOM_SEED
        The validation ratio is self.VAL_RATIO
        """
        
        self.indices_de = self.tokenizer_de.encode(self.texts_de)
        self.indices_en = self.tokenizer_en.encode(self.texts_en)

        self.pad_id_en, self.unk_id_en, self.bos_id_en, self.eos_id_en = \
            self.tokenizer_en.pad_id, self.tokenizer_en.unk_id, \
            self.tokenizer_en.bos_id, self.tokenizer_en.eos_id
        
        self.pad_id_de, self.unk_id_de, self.bos_id_de, self.eos_id_de = \
            self.tokenizer_de.pad_id, self.tokenizer_de.unk_id, \
            self.tokenizer_de.bos_id, self.tokenizer_de.eos_id    
        
        self.max_length = max_length
        self.vocab_size_en = self.tokenizer_en.vocab_size
        self.vocab_size_de = self.tokenizer_de.vocab_size

    def text2ids(self, texts: Union[str, List[str]], language="de") -> Union[List[int], List[List[int]]]:
        """
        Encode a text or list of texts as tokenized indices
        :param texts: text or list of texts to tokenize
        :return: encoded indices
        """
        if language == "de":
            return self.tokenizer_de.encode(texts)
        else:
            return self.tokenizer_en.encode(texts)

    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]], language="en") -> Union[str, List[str]]:
        """
        Decode indices as a text or list of tokens
        :param ids: 1D or 2D list (or torch.Tensor) of indices to decode
        :return: decoded texts
        """
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        if language == "de":
            return self.tokenizer_de.decode(ids)
        else:
            return self.tokenizer_en.decode(ids)

    def __len__(self):
        """
        Size of the dataset
        :return: number of texts in the dataset
        """
        return len(self.indices_en)

    def __getitem__(self, item: int):
        """
        Add specials to the index array and pad to maximal length
        :param item: text id
        :return: encoded text indices and its actual length (including BOS and EOS specials)
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Take corresponding index array from self.indices,
        add special tokens (self.bos_id and self.eos_id) and 
        pad to self.max_length using self.pad_id.
        Return padded indices of size (max_length, ) and its actual length
        """
        indices_en = self.indices_en[item]
        length_en = min(self.max_length, len(indices_en) + 2)
        
        if len(indices_en) + 2 > self.max_length:
            indices_en = indices_en[:self.max_length - 2]
        
        indices_en = [self.bos_id_en] + indices_en + [self.eos_id_en]
        indices_en += [self.pad_id_en for _ in range(self.max_length - len(indices_en))]
        
        indices_de = self.indices_de[item]
        length_de  = min(self.max_length, len(indices_de ) + 2)
        
        if len(indices_de) + 2 > self.max_length:
            indices_de  = indices_de[:self.max_length - 2]
        
        indices_de  = [self.bos_id_de] + indices_de  + [self.eos_id_de]
        indices_de  += [self.pad_id_de for _ in range(self.max_length - len(indices_de))]
        
        
        return (torch.tensor(indices_de), torch.tensor(indices_en)), (length_de, length_en)
