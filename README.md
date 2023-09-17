# Rationale

Neural network-based language models are ideally suited for compressing text, as they can efficiently predict the next word in a sentence.
Instead of storing all words in a sentence, we can instead only store the index of the next predicted words:

An apple a day keeps the doctor away (37 characters)
-> 40, 9, 6, 3, 1, 1, 1, 1 (9 digits)

Likewise, given a list of indices to select the next words from the same neural network, we can decode the digits back to words.

However, this requires us to store the full neural network weights, throwing away any compression gains. Interestingly, it is possible to decode the digits without ever storing the neural weights.

The [notebook.ipynb](notebook.ipynb) demonstrates this proof of concept:
- Encoder neural network: Compresses data.txt -> compressed.txt using a simple LSTM neural network
- Decoder neural network: Decompresses compressed.txt WITHOUT transmitting the neural network weights
- The idea works by having the encoder compressing text while it is training, and the decoder mirroring the process exactly by decompressing and training on the decompressed text. This way, both neural networks always share the same state over time, removing the need to store the weights externally.

The idea comes from this [2019 NNCP paper](https://bellard.org/nncp/nncp.pdf), which holds the currently world record for smallest compressed version of Wikipedia file (~1 GB -> 100 MB), cleverly avoiding needing to store the NN weights in the file, and is explained in this [HackerNews post](https://news.ycombinator.com/item?id=27244810).

## Notebook proof-of-concept
See [notebook.ipynb](notebook.ipynb)

## Overview
We encode sequences of digits like "000000", "000001", etc., and store the compressed [data](data.txt) in [compressed.txt](compressed.txt). Despite not saving the neural network's weights, we can then decompress this data, retrieving the original sequences. This is achieved by ensuring that both the encoder and decoder evolve identically during their respective processes.

## Key Concepts
Both encoder and decoder start with the same initial model. As they process the sequences, they update their models identically, ensuring synchronized evolution.

Encoder:
- Initialize a neural network with all weights set to the same value (need weight updates to be deterministic)
- (Nb: Save the first sequence without compression)
- For each digit in a sequence, predict the next digit using a neural network model (learning a probability distribution)
- Update the neural network based on the loss
- When done with a sequence, compress the next one based on the learned probability distribution so far using an [Arithmetic Compressor](https://pypi.org/project/arithmetic-compressor/)

Decoder (where the magic happens):
- Initialize the same neural network with the same fixed value
- (Nb: load the first uncompressed sentence)
- Predict each digit of a sequence using the current state of the neural network
- Update the neural network based on the loss (mirroring the encoder state)
- Decompress the next sequence based on the learned probability distribution.
- With the now decompressed sequence, train on it, and learn to decompress the next one, until all sequences are decoded

## Read more
- [NNCP v2: Lossless Data Compression with
Transformer](https://bellard.org/nncp/nncp_v2.1.pdf)
- [Lossless Data Compression with Neural Networks](https://bellard.org/nncp/nncp.pdf)
- [Arithmetic encoding](https://en.wikipedia.org/wiki/Arithmetic_coding)

