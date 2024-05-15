import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import random
import wandb
wandb.login(key="0f6963d23192cbab4399ad9ec6e7475c7a0d6345")

device = "mps" if torch.backends.mps.is_available() else "cpu"
device

# Hindi Unicode Hex Range is 2304:2432. Source: https://en.wikipedia.org/wiki/Devanagari_(Unicode_block)
SOS_token = 0
EOS_token = 1
hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
english_alphabets = [chr(alpha) for alpha in range(97, 123)]
hindi_alphabet_size = len(hindi_alphabets)
english_alphabet_size = len(english_alphabets)
hindi_alpha2index = {"<": 0,">": 1}
english_alpha2index = {"<": 0,">": 1}
for index, alpha in enumerate(hindi_alphabets):
    hindi_alpha2index[alpha] = index+2
for index, alpha in enumerate(english_alphabets):
    english_alpha2index[alpha] = index+2
hindi_index2alpha = {0 : "<", 1 : ">"}
english_index2alpha = { 0 : "<", 1 : ">"}
for index, alpha in enumerate(hindi_alphabets):
    hindi_index2alpha[index+2] = alpha
for index, alpha in enumerate(english_alphabets):
    english_index2alpha[index+2] = alpha 

dataset_path = "./aksharantar_sampled/hin/"
data_train = pd.read_csv(dataset_path + "hin_train.csv",header= None)
data_train = pd.DataFrame(np.array(data_train),columns=["English","Hindi"])
data_val = pd.read_csv(dataset_path + "hin_valid.csv",header= None)
data_val = pd.DataFrame(np.array(data_val),columns=["English","Hindi"])
data_test = pd.read_csv(dataset_path + "hin_test.csv",header= None)
data_test = pd.DataFrame(np.array(data_test),columns=["English","Hindi"])

data_train_X = np.array(data_train["English"])
data_train_y = np.array(data_train["Hindi"])

class Tokenize():
    def __init__(self,Lang_From,Lang_To):
        # Hindi Unicode Hex Range is 2304:2432. Source: https://en.wikipedia.org/wiki/Devanagari_(Unicode_block)
        self.L1 = Lang_From
        self.L2 = Lang_To
        self.SOS_token = 0
        self.EOS_token = 1
        hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
        english_alphabets = [chr(alpha) for alpha in range(97, 123)]
        hindi_alphabet_size = len(hindi_alphabets)
        english_alphabet_size = len(english_alphabets)
        hindi_alpha2index = {"<": 0,">": 1}
        english_alpha2index = {"<": 0,">": 1}
        for index, alpha in enumerate(hindi_alphabets):
            hindi_alpha2index[alpha] = index+2
        for index, alpha in enumerate(english_alphabets):
            english_alpha2index[alpha] = index+2
        hindi_index2alpha = {0 : "<", 1 : ">"}
        english_index2alpha = { 0 : "<", 1 : ">"}
        for index, alpha in enumerate(hindi_alphabets):
            hindi_index2alpha[index+2] = alpha
        for index, alpha in enumerate(english_alphabets):
            english_index2alpha[index+2] = alpha 

        self.Lang_From_Alpha_2_Index = english_alpha2index
        self.Lang_To_Alpha_2_Index = hindi_alpha2index
        self.Lang_From_Index_2_Alpha = english_index2alpha
        self.Lang_To_Index_2_Alpha = hindi_index2alpha

    def tensorFromWord(self,Lang, word):
        if Lang == "L1":
            indexes = [self.Lang_From_Alpha_2_Index[letter] for letter in word]
        elif Lang == "L2":
            indexes = [self.SOS_token]+[self.Lang_To_Alpha_2_Index[letter] for letter in word]
        #print([self.EOS_token]*(30-len(indexes)))
        indexes+=[self.EOS_token]*(30-len(indexes))
        return torch.tensor(indexes, dtype=torch.long, device=device)#.view(-1, 1)
    def tensorsFromPair(self,pair):
        input_tensor = self.tensorFromWord("L1",pair[self.L1])
        target_tensor = self.tensorFromWord("L2",pair[self.L2])
        return (input_tensor, target_tensor)
    def tensorsFromData(self,Data):
        Tensors_Val = []
        for i in tqdm(range(Data.shape[0])):
            Tensors_Val.append(self.tensorsFromPair(Data.iloc[i]))
        return Tensors_Val
    def WordFromtensors(self,Lang, word):
        if Lang == "L1":
            letters = [self.Lang_From_Index_2_Alpha[letter.item()] for letter in word if ((letter.item() != EOS_token) and (letter.item() != SOS_token))]
        elif Lang == "L2":
            letters = [self.Lang_To_Index_2_Alpha[letter.item()] for letter in word if ((letter.item() != EOS_token) and (letter.item() != SOS_token))]
        #print([self.EOS_token]*(30-len(indexes)))
        word = ''.join(letters)
        return word
    def PairFromtensors(self,pair):
        input_word = self.WordFromtensors("L1",pair[0])
        target_word = self.WordFromtensors("L2",pair[1])
        return (input_word, target_word)
    '''def DataFromtensors(self,Data):
        Tensors_Val = []
        for i in tqdm(range(Data.shape[0])):
            Tensors_Val.append(self.PairFromtensors(data_train.iloc[i]))
        return Tensors_Val'''
    
T = Tokenize("English","Hindi")
data_train_num = T.tensorsFromData(data_train)
data_val_num = T.tensorsFromData(data_val)
data_test_num = T.tensorsFromData(data_test)

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
train_set=CustomDataset(data_train_num)
valid_set=CustomDataset(data_val_num)
test_set=CustomDataset(data_test_num)

class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers, dropouts,cell_type,bidirectional):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropouts)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        if num_layers >1:
            if self.cell_type == "LSTM":
                self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=dropouts,bidirectional=self.bidirectional)
            elif self.cell_type == "RNN":
                self.rnn = nn.RNN(embedding_size,hidden_size,num_layers,dropout=dropouts,bidirectional=self.bidirectional)
            elif self.cell_type == "GRU":
                self.rnn = nn.GRU(embedding_size,hidden_size,num_layers,dropout=dropouts,bidirectional=self.bidirectional)
        else:
            if self.cell_type == "LSTM":
                self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,bidirectional=self.bidirectional)
            elif self.cell_type == "RNN":
                self.rnn = nn.RNN(embedding_size,hidden_size,num_layers,bidirectional=self.bidirectional)
            elif self.cell_type == "GRU":
                self.rnn = nn.GRU(embedding_size,hidden_size,num_layers,bidirectional=self.bidirectional)
                
    def forward(self,x):
        # X : (seq_length,N)
        embedding = self.dropout(self.embedding(x))
        # embedding : seq_length,N,embedding_size)
        if self.cell_type == "LSTM":
            outputs,(hidden,cell) = self.rnn(embedding)
        else:
            outputs,hidden = self.rnn(embedding)
            cell = None
        return outputs,hidden,cell
    
class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,num_layers,dropouts,cell_type,bidirectional):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropouts)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.cell_type = cell_type
        self.bidirectional = bidirectional
        if num_layers>1:            
            if self.cell_type == "LSTM":
                self.rnn = nn.LSTM((hidden_size*(1+self.bidirectional*1)+embedding_size),hidden_size,num_layers,dropout=dropouts,bidirectional=self.bidirectional)
            elif self.cell_type == "RNN":
                self.rnn = nn.RNN((hidden_size*(1+self.bidirectional*1)+embedding_size),hidden_size,num_layers,dropout=dropouts,bidirectional=self.bidirectional)
            elif self.cell_type == "GRU":
                self.rnn = nn.GRU((hidden_size*(1+self.bidirectional*1)+embedding_size),hidden_size,num_layers,dropout=dropouts,bidirectional=self.bidirectional)
        else:
            if self.cell_type == "LSTM":
                self.rnn = nn.LSTM((hidden_size*(1+self.bidirectional*1)+embedding_size),hidden_size,num_layers,bidirectional=self.bidirectional)
            elif self.cell_type == "RNN":
                self.rnn = nn.RNN((hidden_size*(1+self.bidirectional*1)+embedding_size),hidden_size,num_layers,bidirectional=self.bidirectional)
            elif self.cell_type == "GRU":
                self.rnn = nn.GRU((hidden_size*(1+self.bidirectional*1)+embedding_size),hidden_size,num_layers,bidirectional=self.bidirectional)
        self.energy = nn.Linear(hidden_size*(2+self.bidirectional*1),1)
        self.fc = nn.Linear((1+self.bidirectional*1)*hidden_size,output_size)
        self.weights = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self,x,encoder_states,hidden,cell):
        # x :(N) but we want (1,N)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding : (1,N,embedding_size)
        sequence_length = encoder_states.shape[0]
        #print("Seq = ",sequence_length)
        #print("Enc = ",encoder_states.shape)
        #print("hidd = ",hidden.shape)
        h_reshaped = hidden.repeat(int(sequence_length/(1+self.bidirectional)),1,1)
        #print(int(sequence_length/(1+self.bidirectional)))
        #print(h_reshaped.shape,encoder_states.shape)
        energy = self.tanh(self.energy(torch.cat((h_reshaped,encoder_states),dim=2)))
        attention_weights = self.weights(energy) 
        # attention : seq_length,N,1
        attention = attention_weights.permute(1,2,0)
        # attention : N,1,seq_length
        encoder_states = encoder_states.permute(1,0,2)
        # (N,1,hidden_size*2) --> (1,N,hidden_size*2)
        context_vector = torch.bmm(attention,encoder_states).permute(1,0,2)
        rnn_input = torch.cat((context_vector,embedding),dim=2)

        if self.cell_type == "LSTM":
            outputs,(hidden,cell) = self.rnn(rnn_input,(hidden,cell))
        else:
            outputs,hidden = self.rnn(rnn_input,hidden)
            cell = None
        # outputs : (1,N,hidden_size)
        predictions = self.fc(outputs)
        #predictions : (1,N,output_vocab_size)
        predictions = self.softmax(predictions[0])
        #predictions = predictions.squeeze(0)
        
        return predictions,hidden,cell,attention_weights

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,source,target,teacher_forceing=0.5):
        batch_size = source.shape[1]
        self.target_len = target.shape[0]
        target_vocab_size = len(hindi_alpha2index)
        
        outputs = torch.zeros(self.target_len,batch_size,target_vocab_size).to(device)
        encoder_states,hidden,cell = self.encoder(source)
        
        # Start Token
        x = target[0]
        for t in range(1,self.target_len):
            output,hidden,cell,_ = self.decoder(x,encoder_states,hidden,cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random()<teacher_forceing else best_guess
        return outputs
    def predict(self,source,track_attn_weights=False):
        batch_size = source.shape[1]
        target_vocab_size = len(hindi_alpha2index)
        
        outputs = torch.zeros(self.target_len,batch_size,target_vocab_size).to(device)
        encoder_states,hidden,cell = self.encoder(source)
        
        # Start Token
        x = 0*source[0]
        
        if track_attn_weights == False:
            Attention_Weights = None
            for t in range(1,self.target_len):
                output,hidden,cell,_ = self.decoder(x,encoder_states,hidden,cell)
                outputs[t] = output
                best_guess = output.argmax(1)
                x = best_guess
        else:
            Attention_Weights = torch.zeros([batch_size,self.target_len,self.target_len]).to(device)
            for t in range(1,self.target_len):
                output,hidden,cell,attention_weights = self.decoder(x,encoder_states,hidden,cell)
                outputs[t] = output
                best_guess = output.argmax(1)
                x = best_guess
                #print(Attention_Weights.shape,attention_weights.shape)
                Attention_Weights[:,:,t] = attention_weights.permute(1,0,2).squeeze()

        return outputs, Attention_Weights
    
    def find_crct_Tot(self,predicted_batch,target_batch):
        crct,Total=0,0
        for i in range(target_batch.shape[0]):
            Pred = T.WordFromtensors("L2",predicted_batch[i])
            Targ = T.WordFromtensors("L2",target_batch[i])
            Total+=1
            if Pred == Targ:
                crct +=1
        return crct,Total

def train_and_tune(config=None):
  # Initialize a new wandb run
  with wandb.init(config=config):
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    config = wandb.config
    wandb.run.name='emb_dim_'+str(wandb.config.emb_dim)+'_num_enc_layers_'+str(wandb.config.num_enc_layers)+'_num_dec_layers_'+str(wandb.config.num_dec_layers)+'_hs_'+str(wandb.config.hidden_size)+'_cell_type_'+config.cell_type+'_bidirectional_'+str(config.bidirectional)+'_lr_'+str(config.learning_rate)+'_bs_'+str(config.batch_size)+'_dropout_'+str(config.dropout)



    # Training Params
    num_epochs = 10
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    train_data_set=DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_data_set=DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_data_set=DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Model Params
    input_size_encoder = len(english_alpha2index)
    input_size_decoder = len(hindi_alpha2index)
    output_size = len(hindi_alpha2index)
    emb_dim = config.emb_dim
    hidden_size = config.hidden_size
    num_enc_layers = config.num_enc_layers
    num_dec_layers = config.num_dec_layers
    enc_dropout = config.dropout
    dec_dropout = config.dropout
    cell_type = config.cell_type
    bidirectional = config.bidirectional


    encoder_net = Encoder(input_size_encoder,emb_dim,hidden_size,num_enc_layers,enc_dropout,cell_type,bidirectional).to(device)
    decoder_net = Decoder(input_size_decoder,emb_dim,hidden_size,output_size,num_enc_layers,dec_dropout,cell_type,bidirectional).to(device)

    model = Seq2Seq(encoder_net, decoder_net).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    pad_idx = EOS_token
    criterion = nn.CrossEntropyLoss()#ignore_index=pad_idx)
    Loss_log = []
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for batch in tqdm(train_data_set):
            inp_data = batch[0].T.to(device)
            target = batch[1].T.to(device)
            #print(inp_data.shape)
            #print(inp_data)
            output = model(inp_data,target)
            #output : (trg_len,batch_size,output_dim)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            
            optimizer.zero_grad()
            loss = criterion(output,target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
            optimizer.step()
            epoch_loss += loss.item()
        Loss_log.append(epoch_loss)
        Train_epoch_loss = epoch_loss/len(train_data_set)

        Predictions_List = []
        Total = 0
        crct = 0
        Val_epoch_loss = 0
        for batch in tqdm(valid_data_set):
            inp_data = batch[0].T.to(device)
            target = batch[1].T.to(device)
            output,_ = model.predict(inp_data)
            #print(output_val[2])
            best_guess = output.argmax(2)
            predictions = best_guess.squeeze()
            #print(predictions.shape)
            output = output[1:].reshape(-1,output.shape[2])
            target = target[1:].reshape(-1)
            loss = criterion(output,target)
            Val_epoch_loss += loss.item()
            for i in range(batch[1].shape[0]):
                Pairs_P = T.PairFromtensors((batch[0][i],predictions.T[i]))
                Pairs_T = T.PairFromtensors((batch[0][i],batch[1][i]))
                Total+=1
                if Pairs_P[1] == Pairs_T[1]:
                    crct +=1
        Val_epoch_loss=Val_epoch_loss/len(valid_data_set)
        Val_Accuracy = crct/Total
        print("train_loss ", Train_epoch_loss,"val_loss ", Val_epoch_loss,"val_accuracy ", (Val_Accuracy * 100))
        wandb.log({"train_loss":Train_epoch_loss,"val_loss":Val_epoch_loss,"val_accuracy": (Val_Accuracy * 100)})

sweep_config={'method':'bayes',
              'name':'Sweep on Kaggle',
              'metric' : {
                  'name':'val_accuracy',
                  'goal':'maximize'},
              'parameters':{ 
                  'learning_rate':{'values':[0.001,0.0001]},
                  'batch_size':{'values':[32,64]},
                  'emb_dim':{'values':[128,256,512]} ,
                  'num_enc_layers':{'values':[1,2,3]},
                  'num_dec_layers':{'values':[1,2,3]},
                  'hidden_size':{'values':[256,512]},
                  'cell_type':{'values':["RNN","GRU","LSTM"]},
                  'bidirectional':{'values':[True,False]},
                  'dropout':{'values':[0.2,0.3]} }}
# import pprint
# pprint.pprint(sweep_config)
sweep_id=wandb.sweep(sweep_config,project="Testing_3")

wandb.agent(sweep_id, train_and_tune,count=1)