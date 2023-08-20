# Neural Network Decompression Without Storing Weights
The [notebook.ipynb](notebook.ipynb) demonstrates this workflow:
- Encoder: Compress data.txt -> compressed.txt using a simple LSTM neural network
- Decoder: Decompress compressed.txt WITHOUT transmitting the neural network weights
- Relies on the encoder training+compressing one sequence at a time, and decoder mirroring the steps exactly by decompressing+training one sequence at a time (sharing states w/o needing to transmit weights)

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

