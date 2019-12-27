import youtokentome as yttm
yttm.BPE.train(data='../data/data.csv',model='../models/bpe.model', vocab_size=10000, n_threads=-1, coverage=0.9999)
