import torch
import os
from torch.optim import Adam
from itertools import chain

from allennlp.training.trainer import Trainer
from allennlp.data.iterators import BucketIterator
from allennlp.common.file_utils import cached_path
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, MultiHeadSelfAttention
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import LanguageModel
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from source.preprocessing import MessageDataPreprocessor
from source.reader import MessageDatasetReader


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def train_model(model: str,
                data_path: str,
                model_path: str,
                config: Config,
                seed: int,
                subwords: bool = False,
                mistakes_rate: float = 0):

    train_path = os.path.normpath(data_path + '/train_data.csv')
    test_path = os.path.normpath(data_path + '/test_data.csv')
    spoiled_path = os.path.normpath(data_path + '/train_data_spoiled.csv')
    vocab_path = os.path.normpath(model_path + "/vocab")

    torch.manual_seed(seed)

    bpe_train_needed = subwords and not os.path.exists(os.path.normpath(data_path + '/bpe.model'))
    if not os.path.exists(os.path.normpath(data_path + '/train_data.csv')) or bpe_train_needed:
        preprocessor = MessageDataPreprocessor(data_path, subwords=subwords, bpe_path=model_path)
        preprocessor.split()
    if mistakes_rate > 0 and not os.path.exists(spoiled_path):
        preprocessor = MessageDataPreprocessor(data_path)
        preprocessor.add_mistakes(mistakes_rate)


    reader = MessageDatasetReader(tokenizer=WordTokenizer(), subwords=subwords,
                                  bpe_path=os.path.normpath(model_path + '/bpe.model'))

    train_dataset = reader.read(cached_path(train_path))
    validation_dataset = reader.read(cached_path(test_path))
    if not os.path.exists(vocab_path):
        vocab = Vocabulary.from_instances(chain(train_dataset, validation_dataset))
        vocab.save_to_files(vocab_path)
    else:
        vocab = Vocabulary.from_files(vocab_path)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size("tokens"),
                                embedding_dim=config.embedding_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    chosen_model = {
        'lstm': PytorchSeq2SeqWrapper(torch.nn.LSTM(config.embedding_dim, config.hidden_dim, batch_first=True)),
        'transformer': MultiHeadSelfAttention(num_heads=2, input_dim=config.embedding_dim,
                                              attention_dim=16, values_dim=16, attention_dropout_prob=config.dropout)
    }

    lang_model = LanguageModel(vocab=vocab, text_field_embedder=word_embeddings,
                               contextualizer=chosen_model[model])

    iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("source", "num_tokens")])
    iterator.index_with(vocab)

    optimizer = Adam(lang_model.parameters())

    if model == 'lstm' and subwords:
        serialization_dir = os.path.normpath(model_path + '/lstm_subwords')
    elif model == 'lstm' and not subwords:
        serialization_dir = os.path.normpath(model_path + '/lstm_words')
    elif model == 'transformer' and subwords:
        serialization_dir = os.path.normpath(model_path + '/transformer_subwords')
    else:
        serialization_dir = os.path.normpath(model_path + '/transformer_words')

    if torch.cuda.is_available():
        cuda_device = 0
        lang_model = lang_model.cuda(cuda_device)
    else:
        cuda_device = -1

    if mistakes_rate > 0:
        train_dataset = None
        validation_dataset =

    trainer = Trainer(
        model=lang_model,
        optimizer=optimizer,
        iterator=iterator,
        patience=config.patience,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        num_epochs=config.num_epochs,
        serialization_dir=serialization_dir,
        cuda_device=cuda_device,
    )
    trainer.train()
