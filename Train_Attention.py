import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import torch.nn as nn

# Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size, emb_dim, num_layers, bidirectional, cell_type, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        if cell_type == "LSTM":
            self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(emb_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        elif cell_type == "RNN":
            self.rnn = nn.RNN(emb_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)

    def forward(self, input, cell1=None):
        embedded = self.dropout(self.embedding(input))
        if self.cell == "LSTM":
            output, (hidden, cell1) = self.cell(embedded)
        else:
            output, hidden = self.cell(embedded)
        return output, hidden, cell1 if cell1 is not None else None
    
# Define the Decoder class with attention
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_size, num_layers, cell_type, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, hidden_size)
        self.num_layers = num_layers
        
        if cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        elif cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)
        elif cell_type == "RNN":
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, dropout=dropout)
        
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell=None):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        x, y = sample[0], sample[1]
        if self.transform is not None:
            x, y = self.transform(x, y)
        return x, y

dataset = "./aksharantar_sampled/hin"

# Dataset
train_dataset = CustomDataset(os.path.join(dataset, f"hin_train.csv"))
test_dataset = CustomDataset(os.path.join(dataset, f"hin_test.csv"))
val_dataset = CustomDataset(os.path.join(dataset, f"hin_valid.csv"))

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

import torch
from torch.utils.data import Dataset

SOS_token = '<'
EOS_token = '>'

class TransliterationDataset(Dataset):
    def __init__(self, data_path, split, input_vocab, output_vocab, max_seq_length):
        self.data_path = data_path
        self.split = split
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_seq_length = max_seq_length
        
        # Load data
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_sequence, output_sequence = self.data[index]
        
        # Convert input and output sequences to tensors
        input_tensor = self.sequence_to_tensor(input_sequence, self.input_vocab, self.max_seq_length)
        output_tensor = self.sequence_to_tensor(output_sequence, self.output_vocab, self.max_seq_length)
        
        return {'input_sequence': input_tensor, 'output_sequence': output_tensor}

    def load_data(self):
        # Load data from CSV file or any other source
        # Preprocess and return the data as a list of tuples (input_sequence, output_sequence)
        file_name = f'hin_{self.split}.csv'
        data = pd.read_csv(os.path.join(self.data_path, file_name), header=None)

        # Preprocess data and create list of tuples (input_sequence, output_sequence)
        data_tuples = []
        for row in data.itertuples(index=False):
            input_sequence, output_sequence = str(row[0]), str(row[1])  # Convert to string in case of numeric data
            data_tuples.append((input_sequence, output_sequence))

        return data_tuples

    def sequence_to_tensor(self, sequence, vocab, max_seq_length):
        # Convert sequence of characters to tensor of indices based on vocab
        # Pad or truncate sequence to max_seq_length
        tensor = [vocab[SOS_token]] + [vocab[char] for char in sequence]
        tensor = tensor[:max_seq_length] + [vocab[EOS_token]] * (max_seq_length - len(tensor))
        return torch.tensor(tensor)

def create_dataset(data_path, split, input_vocab, output_vocab, max_seq_length):
    dataset = TransliterationDataset(data_path, split, input_vocab, output_vocab, max_seq_length)
    return dataset

def loadDataset(batch_size):
    # Define dataset path and splits
    dataset_path = './aksharantar_sampled/hin/'
    splits = ['train', 'test', 'valid']

    # Load the dataset and create vocabulary mappings
    input_characters = set()
    output_characters = set()

    def create_vocab_mapping(characters):
        char_to_index = {SOS_token: 0, EOS_token: len(characters)+1}
        char_to_index.update({char: index+1 for index, char in enumerate(characters)})
        index_to_char = {index: char for char, index in char_to_index.items()}
        return char_to_index, index_to_char


    for split in splits:
        file_name = f'hin_{split}.csv'
        data = pd.read_csv(os.path.join(dataset_path, file_name), header=None)  # Read CSV without headers
        input_characters.update(set(''.join(data.iloc[:, 0].tolist())))  # Access first column
        output_characters.update(set(''.join(data.iloc[:, 1].tolist())))  # Access second column

    input_char_to_index, input_index_to_char = create_vocab_mapping(input_characters)             # 26 values
    output_char_to_index, output_index_to_char = create_vocab_mapping(output_characters)          # 65 values

#     print(input_char_to_index)
#     print(output_char_to_index)

    # # Define max sequence length
    # maxLengthEng, maxLengthHin = findMax(datasets['train'])
    # print(maxLengthEng, maxLengthHin)
    # max_seq_length = max(maxLengthEng, maxLengthHin) + 5
    max_seq_length = 30

    # Create datasets for train, test, and validation
    datasets = {}
    for split in splits:
        datasets[split] = create_dataset(dataset_path, split, input_char_to_index, output_char_to_index, max_seq_length)

    # Example usage:
    train_data = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    val_data = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False)
    test_data = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
    
    return train_data, val_data, test_data, input_char_to_index, output_char_to_index, input_index_to_char, output_index_to_char

def tensor_to_words(tensor, char_to_idx, idx_to_char, SOS_token, EOS_token):
    words = []
    for i in range(tensor.size(0)):
        word = []
        for j in range(tensor.size(1)):
            char = idx_to_char[tensor[i][j].item()]
            if char == SOS_token:
                continue  # Skip the SOS token
            if char == EOS_token:
                break  # Stop at the EOS token
            word.append(char)
        words.append(''.join(word))
    return words

def word_accuracy(predicted_words, target_words, correct_words, words):
    for pred_word, target_word in zip(predicted_words, target_words):
        if pred_word == target_word:
            # print("pred: ", pred_word, "target: ", target_word)
            correct_words += 1
        words += 1
    return correct_words, words

def trainEncoderDecoder(device, model, train_data, val_data, test_data, output_char_to_index, output_index_to_char, num_epochs = 5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        total_correct_words = 0
        total_words = 0
        # Loop through each batch in the train_data
        for batch in tqdm(train_data):
            # Zero the gradients
            optimizer.zero_grad()

            # Extract input and target sequences from batch
            input_sequences = batch['input_sequence']
            target_sequences = batch['output_sequence']

            input_sequences = input_sequences.to(device)
            target_sequences = target_sequences.to(device)

            # Forward pass: encode input sequences and decode to get predicted output sequences
            outputs = model(input_sequences, target_sequences)

            # Compute the loss
            loss = criterion(outputs.view(-1, outputs.shape[-1]), target_sequences.view(-1))

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update the parameters of the model
            optimizer.step()

            # Add the batch loss to the total loss for this epoch
            total_loss += loss.item()

            # Calculate word accuracy
            predicted_sequences = outputs.argmax(dim=-1)
            predicted_words = tensor_to_words(predicted_sequences, output_char_to_index, output_index_to_char, SOS_token, EOS_token)
            target_words = tensor_to_words(target_sequences, output_char_to_index, output_index_to_char, SOS_token, EOS_token)
            batch_correct_words, batch_words = word_accuracy(predicted_words, target_words, total_correct_words, total_words)
            
            total_correct_words += batch_correct_words
            total_words += batch_words
     
        # Calculate average loss and accuracy for this epoch
        average_loss = total_loss / len(train_data)
        accuracy = total_correct_words / total_words

        # Print average loss and accuracy for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss}, Accuracy: {accuracy * 100}")
        # wandb.log({"train_loss": average_loss, "train_accuracy": accuracy * 100}, step=epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_words = 0
        val_words = 0

        # Loop through each batch in the val_data
        for batch in tqdm(val_data):
            # Extract input and target sequences from batch
            input_sequences = batch['input_sequence'].to(device)
            target_sequences = batch['output_sequence'].to(device)

            # Forward pass: encode input sequences and decode to get predicted output sequences
            outputs = model(input_sequences, target_sequences)

            # Compute the loss
            loss = criterion(outputs.view(-1, outputs.shape[-1]), target_sequences.view(-1))

            # Add the batch loss to the total loss for this epoch
            val_loss += loss.item()

            # Calculate accuracy
            predicted_sequences = outputs.argmax(dim=-1)
            # predicted_words = tensor_to_words(predicted_sequences, output_char_to_index, output_index_to_char, SOS_token, EOS_token)
            # target_words = tensor_to_words(target_sequences, output_char_to_index, output_index_to_char, SOS_token, EOS_token)
            
            batch_correct_words, batch_words = word_accuracy(predicted_words, target_words, val_correct_words, val_words)
            
            val_correct_words += batch_correct_words
            val_words += batch_words
  
        val_average_loss = val_loss / len(val_data)
        val_accuracy = val_correct_words / val_words

        # Print average loss and accuracy for validation
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_average_loss}, Validation Accuracy: {val_accuracy * 100}")
        # wandb.log({"val_loss": val_average_loss, "val_accuracy": val_accuracy * 100}, step=epoch)

        #testing
        if epoch == num_epochs-1:
            model.eval()
            test_loss = 0.0
            test_correct_words = 0
            test_words = 0

            # Loop through each batch in the test DataLoader
            for batch in tqdm(test_data):
                # Extract input and target sequences from batch
                input_sequences = batch['input_sequence'].to(device)
                target_sequences = batch['output_sequence'].to(device)

                # Forward pass: encode input sequences and decode to get predicted output sequences
                outputs = model(input_sequences, target_sequences)

                # Compute the loss
                loss = criterion(outputs.view(-1, outputs.shape[-1]), target_sequences.view(-1))

                # Add the batch loss to the total loss for testing
                test_loss += loss.item()

                # Calculate accuracy
                predicted_sequences = outputs.argmax(dim=-1)
                predicted_words = tensor_to_words(predicted_sequences, output_char_to_index, output_index_to_char, SOS_token, EOS_token)
                target_words = tensor_to_words(target_sequences, output_char_to_index, output_index_to_char, SOS_token, EOS_token)
                
                batch_correct_words, batch_words = word_accuracy(predicted_words, target_words, test_correct_words, test_words)
                
                test_correct_words += batch_correct_words
                test_words += batch_words

            # Calculate average loss and accuracy for testing
            test_average_loss = test_loss / len(test_data)
            test_accuracy = test_correct_words / test_words

        #     # Print average loss and accuracy for testing
            print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_average_loss}, Test Accuracy: {test_accuracy * 100}")
        #     # wandb.log({"test_loss": test_average_loss, "test_accuracy": test_accuracy * 100})

def main():
    # Instantiate the EncoderDecoder
    batch_size = 32
    train_data, val_data, test_data, input_char_to_index, output_char_to_index, input_index_to_char, output_index_to_char = loadDataset(batch_size)
    input_dim = len(input_char_to_index)  # Size of input vocabulary
    output_dim = len(output_char_to_index)
    emb_dim = 64
    hidden_size = 256  #  hidden layer size
    num_layers = 3  # Number of layers in the encoder and decoder
    bidirectional = False  # Whether the encoder is bidirectional
    cell_type = "LSTM"
    dropout = 0.2
    num_epochs = 1
    learning_rate = 0.0001
    
    encoder = Encoder(input_dim, hidden_size, emb_dim, num_layers, bidirectional, cell_type, dropout)
    decoder = Decoder(output_dim, hidden_size, num_layers, cell_type, dropout)
    
    model = EncoderDecoder(encoder, decoder, device)
    model.to(device)
    
    trainEncoderDecoder(device, model, train_data, val_data, test_data, output_char_to_index, output_index_to_char, num_epochs, learning_rate)
    model.to(device)
    
    # sweep_config = {
    #     'method' : 'bayes',
    #     'project' : "DL_Assignment_3",
    #     'name' : 'Word accuracy (local)',
    #     'metric' : {
    #         'name' : 'val_accuracy', 
    #         'goal' : 'maximize'
    #     },
    #     'parameters' : {
    #         'emb_dim': {
    #             'values' : [16, 32, 64]
    #         },
    #         'hidden_size': {
    #             'values' : [128, 256, 512]
    #         },
    #         'num_layers' : {
    #             'values' : [1, 2, 3]
    #         },
    #         'bidirectional' : {
    #             'values' : [True, False]
    #         },
    #         'cell_type' : {
    #             'values' : ["LSTM", "GRU", "RNN"]
    #         },
    #         'num_epochs' : {
    #             'values' : [2, 5, 10]
    #         },
    #         'learning_rate': {
    #             'values' : [1e-3, 1e-4]
    #         },
    #         'batch_size': {
    #             'values' : [16, 32, 64]
    #         },
    #         'dropout' : {
    #             'values' : [0.0, 0.2, 0.4]
    #         }
    #     }
    # }

    # def train():
    #     with wandb.init(project="DL_Assignment_3") as run:

    #         config = wandb.config
    #         run_name = "inp_emb_dim_" + str(config.emb_dim) + "_hs_" + str(config.hidden_size) + "_num_enc_layers_" + str(config.num_layers) + "_num_dec_layers_" + str(config.num_layers) + "_bidirectional_" + str(config.bidirectional) + "_cell_type_" + str(config.cell_type) + "_num_epochs_" + str(config.num_epochs) + "_lr_" + str(config.learning_rate) +"_drouput_" + str(config.dropout)
    #         wandb.run.name = run_name
            
    #         train_data, val_data, test_data, input_char_to_index, output_char_to_index, input_index_to_char, output_index_to_char = loadDataset(config.batch_size)
            
    #         input_dim = len(input_char_to_index)  # Size of input vocabulary
    #         output_dim = len(output_char_to_index)
            
    #         encoder = Encoder(input_dim, config.hidden_size, config.emb_dim, config.num_layers, 
    #                           config.bidirectional, config.cell_type, config.dropout)
    #         decoder = Decoder(output_dim, config.hidden_size, config.num_layers, 
    #                           config.bidirectional, config.cell_type, config.dropout)

    #         model = EncoderDecoder(encoder, decoder, device)
    #         model.to(device)

    #         trainEncoderDecoder(device, model, train_data, val_data, test_data, output_char_to_index, output_index_to_char, config.num_epochs, config.learning_rate)
    #         model.to(device)

    # sweep_id = wandb.sweep(sweep=sweep_config)
    # wandb.agent(sweep_id, function=train, count=50)
    # wandb.finish()
    # train()

if __name__=="__main__":
    main()