# CS6910 Assignment 3 - RNN Network for Sequence-to-Sequence Learning and Attention Networks.

# Description

This repository contains the code for training a Recurrent Neural Network (RNN) model for sequence-to-sequence learning tasks

The RNN architecture used in this project consists of an encoder-decoder framework, where the encoder processes the input sequence and encodes it into a fixed-size context vector, which is then decoded by the decoder to generate the output sequence.

# Features

- Implementation of a sequence-to-sequence model using PyTorch.
- Support for RNN cell types including LSTM, GRU, and RNN.
- Customizable hyperparameters such as learning rate, batch size, embedding dimension, number of layers, hidden size, cell type, dropout rate, and bidirectional encoding.
- Training, validation, and testing functionality with evaluation metrics including loss and accuracy.
- Implementation of sequence-to-sequence model with attention using PyTorch.
- Visualization tools for attention heatmaps and model predictions.

<br>

### Contents

- `Train_Vanilla.py`: Containing main function for training the RNN network.

- `Train_Attention.py`: Containing function for training the RNN network with attention.

- `requirements.txt`: File containing required libraries for the assignment.

- `aksharantar_sampled`: Dataset used for the RNN network.

- `predictions_attention`: Folder containing predictions (csv) on validation and test datasets.

- `Noto_Sans_Devanagari`: Folder containing the fonts used for the heatmap entries.

<br>

### Usage
To train the neural network, please follow the steps given below:

- Import the required libraries:
   ```
   pip install -r requirements.txt

- Please put your Wandb API key in `Train_Vanilla.py` and `Train_Attention.py` before running the file to track the runs.

   
- Run the vanilla code on default parameters (check the end of file for default parameters).
   ```
   Python Train_Vanilla.py

- Run the attention code on default parameters (check the end of file for default parameters):
  ```
  Python Train_Attention.py 
   
- Use your parameters:
    - Example: `Python Train_Vanilla.py --batch_size 64 --learning_rate 0.001 --cell_type LSTM` to run the RNN with a batch size 64, learning rate of 0.001 and cell type of LSTM.

<br>

Link to the wandb report: [WandB report link](https://wandb.ai/cs23m024-gaurav/CS6910_Assignment_3/reports/Assignment-3--Vmlldzo3OTU0MDE2)

<br>

### Dataset

I have used the `Aksharantar dataset` released by AI4Bharat.
This dataset contains pairs of the following form: 
```
x,y
ajanabee,अजनबी
```
a word in the native script and its corresponding transliteration in the Latin script.

<br>

### Hyperparameters and their default values
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | CS6910_Assignment_3 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | CS23M024  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-bs`, `--batch_size` | 64 | Batch size used to train RNN network. | 
| `-lr`, `--learning_rate` | 0.001 | Learning rate used to optimize model parameters | 
| `-nel`, `--num_enc_layers` | 3 | Number of layers in encoder |
| `-ndl`, `--num_dec_layers` | 3 | number of layers in decoder | 
| `-ed`, `--emb_dim` | 256 | Embedding size | 
| `-hs`, `--hidden_size` | 512 | Hidden size |
| `-ct`, `--cell_type` | LSTM | Cell type choices: ['LSTM', 'GRU', 'RNN'] |
| `-bd`, `--bidirectional` | True | Bidirectional |
| `-dp`, `--dropout` | Dropout | dropout value | 

<br>
