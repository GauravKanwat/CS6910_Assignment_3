import random
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import hyperparameter_config

import wandb
wandb.login(key="0f6963d23192cbab4399ad9ec6e7475c7a0d6345")


device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device


# Defining unicode characters for Hindi and English
hin_start, hin_end = 0x0900, 0x0980
eng_start, eng_end = 0x0061, 0x007B


# generate all characters in a given range and return them as a list
def generate_all_characters(start, end):
    all_characters = [chr(char_code) for char_code in range(start, end)]
    return all_characters

class CreateVocab():
    def __init__(self,input_language,output_language):
        self.language1 = input_language
        self.language2 = output_language
        self.SOS_token = "<"
        self.EOS_token = ">"
        self.SOS_token_index = 0
        self.EOS_token_index = 1
        self.max_length = 30

        # Generate all the characters for Hindi and English
        hindi_characters = [chr(alpha) for alpha in range(hin_start, hin_end)]
        english_characters = [chr(alpha) for alpha in range(eng_start, eng_end)]

        # Create dictionaries for character to index mapping
        output_char_to_index = {self.SOS_token : self.SOS_token_index, self.EOS_token : self.EOS_token_index}
        input_char_to_index = {self.SOS_token : self.SOS_token_index, self.EOS_token : self.EOS_token_index}
        
        # Add characters from hindi_characters and english_characters with their respective indices
        output_char_to_index.update({char: index + 2 for index, char in enumerate(hindi_characters)})
        input_char_to_index.update({char: index + 2 for index, char in enumerate(english_characters)})

        # Create dictionaries for index to character mapping
        output_index_to_char = {index: char for char, index in output_char_to_index.items()}
        input_index_to_char = {index: char for char, index in input_char_to_index.items()}

        # Store the updated dictionaries
        self.updated_inp_char_to_index = input_char_to_index
        self.inp_lang_char_to_index = input_char_to_index

        # loop through the hindi characters and add them to the dictionary
        self.updated_inp_char_to_index.update({char: index + 2 for index, char in enumerate(hindi_characters)})

        self.updated_out_char_to_index = output_char_to_index
        self.out_lang_char_to_index = output_char_to_index

        # loop through the english indexes and add them to the dictionary
        self.updated_inp_char_to_index.update({char: index + 2 for index, char in enumerate(english_characters)})

        self.updated_inp_index_to_char = input_index_to_char
        self.inp_lang_index_to_char = input_index_to_char

        # loop through the hindi indexes and add them to the dictionary
        self.updated_inp_index_to_char.update({char: index + 2 for index, char in enumerate(hindi_characters)})

        self.updated_out_index_to_char = output_index_to_char
        self.out_lang_index_to_char = output_index_to_char

        # loop through the english indexes and add them to the dictionary
        self.updated_out_index_to_char.update({char: index + 2 for index, char in enumerate(english_characters)})

    # Function to convert the word to index
    def word_to_index(self, lang, word):
        if lang == self.language1:
            word_len = len(word)
            indexes = [self.inp_lang_char_to_index[letter] for letter in word]
        elif lang == self.language2:
            word_len = len(word)
            indexes = [self.SOS_token_index]+[self.out_lang_char_to_index[letter] for letter in word]
        word_len = len(indexes)
        indexes += [self.EOS_token_index] * (self.max_length - len(indexes))
        return torch.tensor(indexes, dtype=torch.long, device=device)
    
    # Function to convert the words to indexes
    def pair_to_index(self,pair):
        input_tensor = self.word_to_index(self.language1, pair[self.language1])
        input = input_tensor.view(-1,1)
        target_tensor = self.word_to_index(self.language2, pair[self.language2])
        pairs = (input_tensor, target_tensor)
        return pairs
    
    # Function to return the pair of words
    def return_pair(self, pair):
        input_tensor = self.word_to_index(self.language1, pair[0])
        input = input_tensor.view(-1,1)
        target_tensor = self.word_to_index(self.language2, pair[1])
        pairs = (input_tensor, target_tensor)
        return pairs
    
    # Function to convert the data to indexes
    def data_to_index(self, Data):
        indexes = []
        for i in range(Data.shape[0]):
            index = i
            indexes.append(self.pair_to_index(Data.iloc[index]))
        print("Data indexing done")
        return indexes
    
    # Function to convert the indexes to words
    def index_to_word(self, Lang, word):
        if Lang == self.language1:
            letters = [self.inp_lang_index_to_char[letter.item()] for letter in word if ((letter.item() != self.EOS_token_index) and (letter.item() != self.SOS_token_index))]
        elif Lang == self.language2:
            letters = [self.out_lang_index_to_char[letter.item()] for letter in word if ((letter.item() != self.EOS_token_index) and (letter.item() != self.SOS_token_index))]
        word = ''.join(letters)
        return word
    
    # Function to convert the indexes to pair of words
    def index_to_pair(self, pair):
        input_word = self.index_to_word(self.language1, pair[0])
        target_word = self.index_to_word(self.language2, pair[1])
        return (input_word, target_word)



# -------------- Define the Encoder and Decoder classes ------------------->
class Encoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, hidden_size, num_layers, bidirectional,cell_type, dropout):
        super(Encoder,self).__init__()

        self.inp_dim = inp_dim
        self.embedding = nn.Embedding(inp_dim, emb_dim)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dropout)
        
        if self.cell_type == "LSTM":
            self.rnn = nn.LSTM(emb_dim,hidden_size,num_layers,bidirectional=self.bidirectional,dropout=(dropout if num_layers>1 else 0))
        elif self.cell_type == "RNN":
            self.rnn = nn.RNN(emb_dim,hidden_size,num_layers,bidirectional=self.bidirectional,dropout=(dropout if num_layers>1 else 0))
        elif self.cell_type == "GRU":
            self.rnn = nn.GRU(emb_dim,hidden_size,num_layers,bidirectional=self.bidirectional,dropout=(dropout if num_layers>1 else 0))

    # Forward pass of the encoder returns the encoder states, hidden and cell states
    def forward(self,x):
        embedding = self.dropout(self.embedding(x))
        if self.cell_type == "LSTM":
            input = embedding
            outputs,(hidden,cell) = self.rnn(embedding)
            embedding = embedding.permute(1,0,2)
        else:
            input = embedding
            outputs,hidden = self.rnn(embedding)
            embedding = embedding.permute(1,0,2)
            cell = None
        return outputs,hidden,cell
    
class Decoder(nn.Module):
    def __init__(self, inp_dim, emb_dim, hidden_size, output_size, num_layers, bidirectional, cell_type, dropout):
        super(Decoder,self).__init__()
        self.inp_dim = inp_dim
        self.embedding = nn.Embedding(inp_dim,emb_dim)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if self.bidirectional else 1  

        if self.cell_type == "LSTM":
            self.cell = nn.LSTM((hidden_size * self.num_directions + emb_dim),hidden_size,num_layers,bidirectional=self.bidirectional, dropout=(dropout if num_layers>1 else 0))
        elif self.cell_type == "RNN":
            self.cell = nn.RNN((hidden_size * self.num_directions + emb_dim),hidden_size,num_layers,bidirectional=self.bidirectional,dropout=(dropout if num_layers>1 else 0))
        elif self.cell_type == "GRU":
            self.cell = nn.GRU((hidden_size * self.num_directions + emb_dim),hidden_size,num_layers,bidirectional=self.bidirectional,dropout=(dropout if num_layers>1 else 0))
        
        self.attn_combine = nn.Linear(hidden_size * (self.num_directions + 1), 1)
        self.fc = nn.Linear(self.num_directions * hidden_size, output_size)
        self.hidden = hidden_size
        self.weights = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.output = output_size
        self.out = nn.Linear(hidden_size * self.num_directions, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # Forward pass of the decoder returns the predictions, hidden, cell states and attention weights
    def forward(self,x,encoder_states,hidden,cell):
        x = x.unsqueeze(0)
        self.out = nn.Linear(self.hidden_size * self.num_directions, self.output_size)
        embedding = self.dropout(self.embedding(x))
        sequence_length = encoder_states.shape[0]
        decoder_hidden = hidden
        embedded = embedding.size(2)
        hidden_reshaped = hidden.repeat(int(sequence_length / self.num_directions),1,1)
        decoder_hidden = decoder_hidden.permute(1,0,2)
        attn_combine = self.relu(self.attn_combine(torch.cat((hidden_reshaped,encoder_states),dim=2)))
        decoder_hidden_reshaped = decoder_hidden.repeat(1,sequence_length,1)
        attention = self.weights(attn_combine) 
        attention = attention.permute(1,2,0)
        embedded = embedding.size(2)
        encoder_states = encoder_states.permute(1,0,2)
        
        # Calculate the context vector
        encoder_context = torch.bmm(attention,encoder_states).permute(1,0,2)
        decoder_context = torch.cat((encoder_context,embedding),dim=2)
        context = torch.bmm(attention,encoder_states).permute(1,0,2)
        decoder_context = torch.cat((context,embedding),dim=2)
        cell_input = torch.cat((context,embedding),dim=2)
        embedded = embedding.size(1)

        if self.cell_type == "RNN" or self.cell_type == "GRU":
            outputs,hidden = self.cell(cell_input,hidden)
            cell = None
        else:
            outputs,(hidden,cell) = self.cell(cell_input,(hidden,cell))

        predictions = self.fc(outputs)
        outputs = predictions.squeeze(0)
        predictions = self.softmax(predictions[0])
        return predictions, hidden, cell, attention

# Define the Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder, vocab):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
    
    # Forward pass of the model returns the decoder outputs
    def forward(self,source,target,teacher_forcing_ratio=0.5):
        self.target_len = target.shape[0]
        batch_size = source.shape[1]
        target_vocab_size = len(self.vocab.out_lang_char_to_index)
        input_length = batch_size * target_vocab_size
        outputs = torch.zeros(self.target_len,batch_size,target_vocab_size).to(device)
        encoder_states,hidden,cell = self.encoder(source)
        
        # Assigning the starting tensor i.e "<" (SOS token) to the input
        x = target[0]
        decoder_input = torch.full((1, batch_size), self.vocab.SOS_token_index, dtype=torch.long)
        batch = x.size(0)
        for t in range(1,self.target_len):
            input = x
            output,hidden,cell,_ = self.decoder(x,encoder_states,hidden,cell)
            outputs[t] = output
            input = output.argmax(1)
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            input = target[t] if use_teacher_forcing else output.argmax(1)
            predicted_sequence = output.argmax(1)
            x = target[t] if use_teacher_forcing else predicted_sequence
            batch = x.size(0)
        return outputs
    
    # Function to return the total and correct predictions
    def calculate_accuracy(self,predicted_batch,target_batch):
        correct,total=0,0
        for i in range(target_batch.shape[0]):
            # converting the index to words to find the predicted and target words
            predicted = self.vocab.index_to_word(self.language2,predicted_batch[i])
            target = self.vocab.index_to_word(self.language2,target_batch[i])
            total+=1
            if predicted == target:
                crct +=1
        return correct, total
    
    # Function to return the outputs and attention weights
    def prediction(self, source, attn_weights=False):
        batch_size = source.shape[1]
        target = torch.zeros(1,batch_size).to(device).long()
        target_vocab_size = len(self.vocab.out_lang_char_to_index)

        outputs = torch.zeros(self.target_len,batch_size,target_vocab_size).to(device)
        encoder_states,hidden,cell = self.encoder(source)
        
        # Starting the decoder with the SOS token
        x = target[0]
        decoder_input = torch.full((1, batch_size), self.vocab.SOS_token_index, dtype=torch.long)
        if attn_weights:
            # Attention_Weights -> (batch_size, target_len, target_len)
            Attention_Weights = torch.zeros([batch_size,self.target_len,self.target_len]).to(device)
            weights = torch.zeros([batch_size,self.target_len,self.target_len]).to(device)
            for t in range(1,self.target_len):
                input = x
                output, hidden, cell, attention_weights = self.decoder(x,encoder_states,hidden,cell)
                outputs[t] = output
                input = output.argmax(1)
                predicted_sequence = output.argmax(1)
                target = predicted_sequence
                x = predicted_sequence
                Attention_Weights[:,:,t] = attention_weights.permute(1,0,2).squeeze()
        else:
            Attention_Weights = None
            # weights = torch.zeros([batch_size,self.target_len,self.target_len]).to(device)
            for t in range(1,self.target_len):
                input = x
                output,hidden,cell,_ = self.decoder(x,encoder_states,hidden,cell)
                outputs[t] = output
                input = output.argmax(1)
                predicted_sequence = output.argmax(1)
                target = predicted_sequence
                x = predicted_sequence
        return outputs, Attention_Weights
    

# Initialize the dataset, addition of SOS and EOS tokens and create dictionaries for character to index mapping
def initializeDataset():
    SOS_token = "<"
    EOS_token = ">"
    SOS_token_index = 0
    EOS_token_index = 1
    hindi_characters = generate_all_characters(hin_start, hin_end)
    english_characters = generate_all_characters(eng_start, eng_end)

    output_char_to_index = {SOS_token : SOS_token_index, EOS_token : EOS_token_index}
    input_char_to_index = {SOS_token : SOS_token_index, EOS_token : EOS_token_index}

    # Add characters from hindi_characters and english_characters with their respective indices
    output_char_to_index.update({char: index + 2 for index, char in enumerate(hindi_characters)})
    input_char_to_index.update({char: index + 2 for index, char in enumerate(english_characters)})

    # Create dictionaries for index to character mapping
    output_index_to_char = {index: char for char, index in output_char_to_index.items()}
    input_index_to_char = {index: char for char, index in input_char_to_index.items()}

    dataset_path = "./aksharantar_sampled/hin/"

    data_train = load_dataset(dataset_path, "train")
    data_val = load_dataset(dataset_path, "valid")
    data_test = load_dataset(dataset_path, "test")

    # data_train_X = np.array(data_train["English"])
    # data_train_y = np.array(data_train["Hindi"])

    return data_train, data_val, data_test, input_char_to_index, output_char_to_index, SOS_token_index, EOS_token_index


# Function to return the correct and total predictions
def return_accurate(batch, predictions, vocab, correct, total):
    for i in range(batch[1].shape[0]):
        predicted_sequences = vocab.index_to_pair((batch[0][i],predictions.T[i]))
        true_sequences = vocab.index_to_pair((batch[0][i],batch[1][i]))
        if predicted_sequences[1] == true_sequences[1]:
            correct +=1
        total+=1
    return correct, total


# Function to train the model and return the training loss
def train(data, model, epoch_loss, optimizer, criterion):
    epoch_loss = 0
    for batch in tqdm(data):
            target_sequence = batch[1].T.to(device)
            input_sequence = batch[0].T.to(device)
            input = input_sequence
            output = model(input_sequence,target_sequence)
            target_sequence = target_sequence[1:].reshape(-1)
            output = output[1:].reshape(-1, output.shape[2])
            
            optimizer.zero_grad()
            loss = criterion(output,target_sequence)
            loss.backward()
            input = input_sequence[1:].reshape(-1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            optimizer.step()
            epoch_loss += loss.item()
    return epoch_loss


# Function to validate the model and return the correct and total predictions
def validate(model, criterion, vocab, val_data):
    total, correct = 0, 0
    epoch_loss = 0

    for batch in tqdm(val_data):
        input_sequence = batch[0].T
        target_sequence = batch[1].T
        input_sequence, target_sequence = input_sequence.to(device), target_sequence.to(device)
        x = input_sequence - 1
        output_sequence,_ = model.prediction(input_sequence)
        pred_sequence = output_sequence.argmax(2)
        x = x[1:].reshape(-1)
        predictions = pred_sequence.squeeze()
        target_sequence = target_sequence[1:].reshape(-1)
        output_sequence = output_sequence[1:].reshape(-1,output_sequence.shape[2])
        x = x[1:].reshape(-1)
        loss = criterion(output_sequence,target_sequence)
        epoch_loss += loss.item()
        
        correct, total = return_accurate(batch, predictions, vocab, correct, total)
    return correct, total, epoch_loss


# train and validate the model, it initializes training and validation functions and prints the training and validation loss
def train_and_validate(model, vocab, train_data, val_data, num_epochs, optimizer, criterion):
    train_loss = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        epoch_loss = train(train_data, model, epoch_loss, optimizer, criterion)
        train_loss.append(epoch_loss)
        epoch_loss = epoch_loss/len(train_data)
        
        correct, total, val_epoch_loss = validate(model, criterion, vocab, val_data)
        val_epoch_loss = val_epoch_loss/len(val_data)
        val_accuracy = correct/total
        print("train_loss ", epoch_loss, "val_loss ", val_epoch_loss, "val_accuracy ", (val_accuracy * 100))
        wandb.log({"train_loss":epoch_loss, "val_loss":val_epoch_loss, "val_accuracy": (val_accuracy * 100)})


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x, y = sample[0], sample[1]
        if self.transform is not None:
            x, y = self.transform(x, y)
        return self.data[idx]

# Load the dataset using the path and split
def load_dataset(dataset_path, split):
    file_path = os.path.join(dataset_path, f"hin_{split}.csv")
    data = pd.read_csv(file_path, header=None, names=["English", "Hindi"])
    data = data.astype(object).replace(np.nan, '', regex=True)  # Handle NaN values
    return data

# Prepare the data for training, validation and testing
def prepare_data(data_train, data_val, data_test, batch_size):
    vocab = CreateVocab("English","Hindi")
    data_train_num = vocab.data_to_index(data_train)
    data_val_num = vocab.data_to_index(data_val)
    data_test_num = vocab.data_to_index(data_test)

    train_data=CustomDataset(data_train_num)
    valid_data=CustomDataset(data_val_num)
    test_data=CustomDataset(data_test_num)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_dataset = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataset, valid_dataset, test_dataset, vocab


def main(args):

    num_epochs = 10              # Number of epochs

    # Sweep configuration
    sweep_config={
        'method':'bayes',
        'name':args.wandb_project,
        'entity':args.wandb_entity,
        'metric' : {
            'name':'val_accuracy',
            'goal':'maximize'
            },
            'parameters':{ 
                'learning_rate' : {
                    'values' : [args.learning_rate]
                },
                'batch_size' : {
                    'values' : [args.batch_size]
                },
                'emb_dim' : {
                    'values' : [args.emb_dim]
                },
                'num_enc_layers' : {
                    'values' : [args.num_enc_layers]
                },
                'num_dec_layers' : {
                    'values' : [args.num_dec_layers]
                },
                'hidden_size' : {
                    'values' : [args.hidden_size]
                },
                'cell_type' : {
                    'values' : [args.cell_type]
                },
                'bidirectional' : {
                    'values' : [args.bidirectional]
                },
                'dropout' : {
                    'values' : [args.dropout]
                }
            }
        }

    
    # Function to run the sweep configuration on training and validation datasets
    def training():
        with wandb.init():
            config = wandb.config
            wandb.run.name='emb_dim_'+str(wandb.config.emb_dim)+'_num_enc_layers_'+str(wandb.config.num_enc_layers)+'_num_dec_layers_'+str(wandb.config.num_dec_layers)+'_hs_'+str(wandb.config.hidden_size)+'_cell_type_'+config.cell_type+'_bidirectional_'+str(config.bidirectional)+'_lr_'+str(config.learning_rate)+'_bs_'+str(config.batch_size)+'_dropout_'+str(config.dropout)
            # learning_rate = 0.001
            # batch_size = 128
            
            # Defining hyperparameters
            learning_rate = config.learning_rate
            batch_size = config.batch_size
            data_train, data_val, data_test, input_char_to_index, output_char_to_index, SOS_token_index, EOS_token_index = initializeDataset()
            input_encoder = len(input_char_to_index)
            input_decoder = len(output_char_to_index)
            output_size = len(output_char_to_index)
            train_dataset, valid_dataset, test_dataset, vocab = prepare_data(data_train, data_val, data_test, batch_size)

            num_enc_layers = config.num_enc_layers
            num_dec_layers = config.num_dec_layers
            emb_dim = config.emb_dim
            hidden_size = config.hidden_size
            bidirectional = config.bidirectional
            cell_type = config.cell_type.upper()
            dropout = config.dropout

            encoder = Encoder(input_encoder, emb_dim, hidden_size, 
                            num_enc_layers, bidirectional,cell_type, dropout).to(device)
            decoder = Decoder(input_decoder, emb_dim, hidden_size, 
                            output_size, num_enc_layers, bidirectional, cell_type, dropout).to(device)

            # Initialize the model, optimizer and criterion
            model = Seq2Seq(encoder, decoder, vocab).to(device)
            optimizer = optim.Adam(model.parameters(), lr = learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Train and validate the model
            train_and_validate(model, vocab, train_dataset, valid_dataset, num_epochs, optimizer, criterion)

    sweep_id=wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)

    wandb.agent(sweep_id, training, count=1)

if __name__ == "__main__":
    args = hyperparameter_config.configParse()
    main(args)