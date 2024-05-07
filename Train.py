import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import torch.nn as nn

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
        
class Encoder:
    def __init__(self, input_size, hidden_size, num_layers, rnn_type="lstm"):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers)
        
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded)
        return output, hidden
    
class Decoder:
    def __init__(self, hidden_size, output_size, num_layers, rnn_type="lstm"):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers)

        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, hidden = self.rnn(embedded, hidden)
        output = nn.functional.log_softmax(self.out(output), dim=2)
        return output, hidden
    
class Seq2SeqModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, encoder_layers=1, decoder_layers=1, encoder_type="lstm", decoder_type="lstm"):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, encoder_layers, encoder_type)
        self.decoder = Decoder(hidden_size, output_size, decoder_layers, decoder_type)
        
    def forward(self, input_seq, target_seq):
        _, encoder_hidden = self.encoder(input_seq)
        decoder_output, _ = self.decoder(target_seq, encoder_hidden)
        return decoder_output
    

model = Seq2SeqModel(input_size, hidden_size, output_size, encoder_layers, decoder_layers, encoder_type, decoder_type)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (input_seq, output_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(input_seq, output_seq)
        output = output.view(-1, output_size)
        target_seq = output_seq.view(-1)
        
        loss = criterion(output, target_seq)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    printf(f"Epoch: {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")