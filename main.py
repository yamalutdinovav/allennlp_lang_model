import argparse
from source.model import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='transformer', type=str, choices=['lstm', 'transformer'],
                        help='Model to train (lstm or transformer)')
    parser.add_argument('--data-path', default='./data/', type=str, dest='data_path',
                        help="Path to the directory containing the dataset")
    parser.add_argument('--model-path', default='./models/', type=str, dest='model_path',
                        help="Path to the directory where models will be saved")
    parser.add_argument('--epochs', default=50, type=int,
                        help="Number of epochs")
    parser.add_argument('--batchsize', default=2, type=int,
                        help="Batch size of the trainer")
    parser.add_argument('--subwords', action='store_true',
                        help="Defines whether the model will be learned on words or subwords")
    parser.add_argument('--seed', default=42, type=int,
                        help="The value of `torch.manual_seed`")
    parser.add_argument('--embedding-dim', default=64, type=int, dest='embedding_dim',
                        help="Dimension of the embedding layer")
    parser.add_argument('--hidden-dim', default=64, type=int, dest='hidden_dim',
                        help="Dimension of the hidden layer of LSTM model")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout rate')

    namespace = parser.parse_args()

    config = Config(
        embedding_dim=namespace.embedding_dim,
        hidden_dim=namespace.hidden_dim,
        batch_size=namespace.batchsize,
        num_epochs=namespace.epochs,
        dropout=namespace.dropout,
        patience=3
    )
    train_model(model=namespace.model,
                data_path=namespace.data_path,
                model_path=namespace.model_path,
                config=config,
                seed=namespace.seed,
                subwords=namespace.subwords,
   )

