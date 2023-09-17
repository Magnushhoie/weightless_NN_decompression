# Rationale

Neural network-based language models are ideally suited for compressing text, as they can efficiently predict the next word in a sentence.
Instead of storing all words directly, we can instead store only the index of the word in the predicted probability distribution, across the sentence:

An apple a day keeps the doctor away (37 characters)
-> 40, 9, 6, 3, 1, 1, 1, 1 (9 digits)

<img width="480" alt="image" src="https://github.com/Magnushhoie/weightless_NN_decompression/assets/39849954/4fe62e9c-bdc7-4904-86b3-4a75e371e646">

Likewise, given a list of indices to select the next words from the same neural network, we can decode the digits back to words.

However, this requires us to store the full neural network weights, throwing away any compression gains. Interestingly, it is possible to decode the digits without ever storing the neural weights.

The [notebook.ipynb](notebook.ipynb) demonstrates this proof of concept:
- Encoder neural network: Compresses data.txt -> compressed.txt using a simple LSTM neural network
- Decoder neural network: Decompresses compressed.txt WITHOUT transmitting the neural network weights
- The idea works by having the encoder compressing text while it is training, and the decoder mirroring the process exactly by decompressing and training on the decompressed text. This way, both neural networks always share the same state over time, removing the need to store the weights externally.

The idea comes from this [2019 NNCP paper](https://bellard.org/nncp/nncp.pdf), which holds the currently world record for smallest compressed version of Wikipedia file (~1 GB -> 100 MB). Under normal circumstances the compressed file would also have to contain the decoder neural weights, but with this technique this requirement is removed. You can read more in this [HackerNews post](https://news.ycombinator.com/item?id=27244810).

## Notebook proof-of-concept
See [notebook.ipynb](notebook.ipynb)

## Implementation details
We encode sequences of digits like "000000", "000001", etc., and store the compressed [data](data.txt) in [compressed.txt](compressed.txt). Instead of using the index of the most likely next word, we'll be even more efficient and use an [Arithmetic Compressor](https://pypi.org/project/arithmetic-compressor/).

Despite not saving the neural network's weights, we can then decompress this data, retrieving the original sequences. This is achieved by ensuring that both the encoder and decoder evolve identically during their respective processes.

Both encoder and decoder start with the same initial model. As they process the sequences, they update their models identically, ensuring synchronized evolution.

Encoder neural network:
- Initialize a neural network with all weights set to the same value (we need the weight updates to be deterministic)
- (Nb: We save the first sequence without compressing it)
- For each digit in a sequence, predict the next digit using a neural network model (learning a probability distribution)
- Update the neural network based on the loss
- When done with a sequence, compress the next one based on the learned probability distribution so far using an Arithmetic Compressor.

Decoder neural network:
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

