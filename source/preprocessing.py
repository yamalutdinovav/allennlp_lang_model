import pandas as pd
import re
import os
from typing import *
from sklearn.model_selection import train_test_split
import youtokentome as yttm
import random


def add_mistakes(word: str) -> str:
    alphabet = list('абвгдежзиклмнопрстуфхцчшщъыьэюя')
    word_list = list(word)
    action = random.randint(1, 3)
    if action == 1: # добавим лишнюю букву
        index = random.randint(1, len(word) - 1)
        word_list[index] = random.choice(alphabet)
    elif action == 2: # уберем букву из слова
        index = random.randint(1, len(word) - 1)
        del word_list[index]
    else:  # перемешаем буквы в слове
        first_idx = random.randint(1, len(word) - 1)
        second_idx = random.randint(1, len(word) - 1)
        word_list[first_idx], word_list[second_idx] = word_list[second_idx], word_list[first_idx]
    return ''.join(word_list)


class MessageDataPreprocessor:
    def __init__(self, path: str, subwords: bool = False, bpe_path: str = None) -> None:
        self.df = pd.read_csv(os.path.normpath(path + '/data.csv'), encoding='utf-8')
        self.path = path
        self.is_preprocessed = False
        if subwords:
            self.write_preprocessed_sentences()
            self.bpe_path = bpe_path
            self.bpe = yttm.BPE.train(os.path.normpath(self.path + '/preprocessed_data.csv'),
                                      os.path.normpath(self.bpe_path + '/bpe.model'),
                                      vocab_size=5000,
                                      coverage=0.999,
                                      n_threads=-1,
                                      )

    def _preprocess(self) -> None:
        phone_regex = re.compile(r'(\d{3})(\s|-|.)*(\d{2,3})(\s|-|.)*(\d{2,3})(\s|-|.)*(\d{2,3})')
        age_regex = re.compile(r'(\d{2}|\d{2}-\d{2})')

        self.df.msg = self.df.msg.apply(lambda x: x.lower())
        self.df.msg = self.df.msg.apply(lambda x: phone_regex.sub('[phone]', x))
        self.df.msg = self.df.msg.apply(lambda x: age_regex.sub('[age]', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'sms', 'смс', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'[:;.?!,/=()]', ' ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'(\s|^)(п|пар|паринь)(\s|$)', ' парень ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'(\s|^)(д|дев|)(\s|$)', ' девушка ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'(\s|^)(ж|жен|женщ)(\s|$)', ' женщина ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'(\s|^)(м|муж|мужч|мущина)(\s|$)', ' мужчина ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'[\s]г(\s|$)', ' год ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'[\s]л(\s|$)', ' лет ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'ищю', 'ищу', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'симпот', 'симпат', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'(\s|^)поз(н|нк|нак)?(\s|$)', ' познакомлюсь ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'(\s|^)(с о|сер[\s]+отн)(\s|$)', ' серьезных отношений ', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'(^\s+|\s+$)', '', x))
        self.df.msg = self.df.msg.apply(lambda x: re.sub(r'\s+', ' ', x))
        self.is_preprocessed = True

    def _get_preprocessed_sentences(self) -> List[str]:
        if not self.is_preprocessed:
            self._preprocess()
        return self.df.msg.values

    def write_preprocessed_sentences(self) -> None:
        dataset = self._get_preprocessed_sentences()
        preprocessed_path = os.path.normpath(self.path + '/preprocessed_data.csv')
        with open(preprocessed_path, 'w', encoding='utf-8') as file:
            for msg in dataset:
                if len(re.split(r'\s+', msg)) > 1:
                    file.write(msg + '\n')

    def split(self) -> Tuple[str, str]:
        dataset = self._get_preprocessed_sentences()
        train_path = os.path.normpath(self.path + '/train_data.csv')
        test_path = os.path.normpath(self.path + '/test_data.csv')
        train_data, test_data = train_test_split(dataset, random_state=42)
        with open(train_path, 'w', encoding='utf-8') as train:
            for msg in train_data:
                if len(re.split(r'\s+', msg)) > 1:
                    train.write(msg + '\n')
        with open(test_path, 'w', encoding='utf-8') as test:
            for msg in test_data:
                if len(re.split(r'\s+', msg)) > 1:
                    test.write(msg + '\n')
        return train_path, test_path

    def add_mistakes(self, mistakes_rate: float) -> None:
        test_path = os.path.normpath(self.path + '/test_data.csv')
        with open(test_path, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                new_line = []
                for word in line.strip().split():
                    if len(word) > 2:
                        rv = random.random()
                        if rv <= mistakes_rate:
                            new_word = add_mistakes(word)
                        else:
                            new_word = word
                    new_line.append(new_word)
                lines.append(' '.join(new_line))
            spoiled_path = os.path.normpath(self.path + '/test_data_spoiled.csv')
            with open(spoiled_path, 'w', encoding='utf-8') as output:
                for line in lines:
                    output.write(line + '\n')
