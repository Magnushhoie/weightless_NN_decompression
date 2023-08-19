# Neural Network Decompression Without Storing Weights
This repository demonstrates a basic implementation for a method of data de-compression using neural networks, inspired by [NNCP](https://bellard.org/nncp/) and this [HackerNews post](https://news.ycombinator.com/item?id=27244810). The fascinating aspect of this technique is the ability to decompress data WITHOUT the need to store or transmit the neural network's weights.

## Overview
We encode sequences of digits like "000000", "000001", etc., and store the compressed [data](data.txt) in [compressed.txt](compressed.txt). Despite not saving the neural network's weights, we can then decompress this data, retrieving the original sequences. This is achieved by ensuring that both the encoder and decoder evolve identically during their respective processes.

## Key Concepts
Encoder:
- Initialize a neural network with all weights set to the same value (need weight updates to be deterministic)
- (Nb: Save the first sequence without compression)
- For each digit in a sequence, predict the next digit using a neural network model (learning a probability distribution)
- Update the neural network based on the loss
- When done with a sequence, compress the next one based on the predicted/learned probability distribution so far using an [Arithmetic Compressor](https://pypi.org/project/arithmetic-compressor/)

Decoder (where the magic happens):
- Initialize the same neural network with the same fixed value
- (Nb: load the first uncompressed sentence)
- Predict each digit of a sequence using the current state of the neural network
- Update the neural network based on the loss
- Decompress the next sequence based on the predicted/learned probability distribution.
- With the de-compressed sentence, train on it, and learn to de-compress the next one, until all sequences are decoded

Both encoder and decoder start with the same initial model. As they process the sequence, they update their models identically, ensuring synchronized evolution.

## Usage
See [notebook.ipynb](notebook.ipynb)

## Read more
- [NNCP v2: Lossless Data Compression with
Transformer](https://bellard.org/nncp/nncp_v2.1.pdf)
- [Lossless Data Compression with Neural Networks](https://bellard.org/nncp/nncp.pdf)
- [Arithmetic encoding](https://en.wikipedia.org/wiki/Arithmetic_coding)

