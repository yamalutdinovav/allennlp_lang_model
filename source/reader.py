from typing import *
from overrides import overrides

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.dataset_readers.language_modeling import DatasetReader

import youtokentome as yttm

class MessageDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 subwords: bool = False,
                 bpe_path: str = None
                 ) -> None:
        super().__init__(lazy=False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.subwords = subwords
        if subwords:
            self.bpe = yttm.BPE(bpe_path)

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:
        if self.subwords:
            tokenized_sentence = [Token(subword)
                                  for subword in self.bpe.encode(sentence, output_type=yttm.OutputType.SUBWORD)]
        else:
            tokenized_sentence = self._tokenizer.tokenize(sentence)
        instance = Instance({
            "source": TextField(tokenized_sentence,
                                  self._token_indexers),
        })
        return instance

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for sentence in f:
                yield self.text_to_instance(sentence)
