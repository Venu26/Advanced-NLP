from datasets import load_dataset
import torch.utils.data as data_utils
import torch
from torch.utils.data import Dataset, DataLoader
import time
import gensim.downloader as api
from torch import nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 100
hidden_size = 100
num_layers = 1
cnt = 0
#torch.autograd.set_detect_anomaly(True)

dataset = load_dataset("cnn_dailymail",'1.0.0')
class wordsDataset(Dataset):
    def __init__(self):
        self.x = dataset['train']['article']
        self.y = dataset['train']['highlights']
        self.n_samples = 100
        
    def __getitem__(self,index):
        return self.x[index],self.y[index]    
        
    def __len__(self):
        return self.n_samples 
    
#print(dataset["train"])
model = torch.load('glove_twitter_100.pt')


dict = {}
Dict = {}

for i in range(len(dataset['train']['article'])):
    print(i/10)
    if i > 10:
        break
    for j in dataset['train']['article'][i].split():
        if j in dict.keys():
            continue
        else:
            dict[j] = cnt+1
            Dict[cnt+1] = j
            cnt = cnt + 1
            
dict["UNK"] = cnt + 1
Dict[cnt+1] = "UNK"
cnt = cnt+1
def getEmbeddings(x):
    if x == "SOS":
        return torch.ones(100)
    vector = model[x] if x in model else torch.zeros(100)
    
    if type(vector) != torch.Tensor:
        return torch.from_numpy(vector)
    return vector


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = nn.LSTM(input_size=input_size,hidden_size=hidden_size,bidirectional=True,batch_first=True)
    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size)
        c0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size)
        output,(hn,cn)= self.encoder(x,(h0,c0))
        return output,(hn,cn)

encoder = Encoder()

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder,self).__init__()
#         self.output_size = cnt
#         self.input_size = input_size+hidden_size*2
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.decoder = nn.LSTM(input_size=input_size+hidden_size*2,hidden_size=hidden_size,batch_first=True)
#         self.power = nn.Linear(hidden_size*3,1)
#         self.softmax = nn.Softmax(dim=0)
#         self.relu = nn.ReLU()
#         self.fc = nn.Linear(hidden_size,cnt)
        
#     def forward(self,x,encoder_states,hidden,cell):
#         length = encoder_states.shape[0]
#         #print(length)
#         h_reshaped = hidden.repeat(length,1,1)
#         print(encoder_states.shape)
#         print(h_reshaped.shape)
        
#         weights = self.relu(self.power(torch.cat((h_reshaped,encoder_states),dim=2)))
#         attention = self.softmax(weights)
#         attention = attention.permute(1,0,2)
#         encoder_states = encoder_states.permute(1,0,2)
#         context_vector = torch.bmm(attention,encoder_states).permute(1,0,2)
#         input = torch.cat((context_vector,getEmbeddings(x)),dim=2)
#         outputs, (hidden,cell) = self.decoder(input, (hidden,cell))
#         predictions = self.fc(outputs).squeeze(0)
#         return predictions, hidden, cell
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = nn.Linear(hidden_size*4+encoder_states[],1)
        #self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=input_size+hidden_size*2,hidden_size=hidden_size*2,batch_first=False)
        self.pgen = nn.Linear(hidden_size*4+input_size,1)
        self.softmax = nn.Softmax(dim=0)
        self.linear1 = nn.Linear(hidden_size*4,hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2= nn.Linear(hidden_size,cnt)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size*2,cnt)
    def forward(self,x,encoder_states,hidden,cell,dicting,sum_attn):
        p = torch.zeros(1,1,100)
        #print(hidden.shape)
        #time.sleep(10)
        p[0] = x
        t = torch.zeros(len(encoder_states[0]))
        for i in range(len(encoder_states[0])):
            print(sum_attn.shape)
            print(sum_attn[0][0])
            f = torch.cat((hidden[0][0],encoder_states[0][i],sum_attn[0][0]))
            print(f.shape)
            t[i] = self.attention(f)
            t[i] = self.relu(t[i])
        
        t1 = t
        for i in range(len(t)):
            t1[i] = torch.exp(t[i])/torch.sum(torch.exp(t))
        ci = torch.zeros(1,hidden_size*2)
        for i in range(len(t1)):
            ci = ci + t1[i]*encoder_states[0][i]
        ci1 = torch.zeros(1,1,200)
        ci1[0] = ci
        ci = ci1
        for i in range(len(t1)):
            sum_attn[i] = sum_attn[i] + t1[i]
        lstm_input = torch.cat((p,ci),dim=2)
        vocab_input = torch.cat((hidden,ci),dim=2)
        Pvocab = 10*self.relu1(self.linear2(self.relu1(self.linear1(vocab_input))))
        Pvocab = self.softmax(Pvocab[0][0])
        
        #print(Pvocab.shape)
        outputs, (hidden,cell) = self.lstm(lstm_input,(hidden,cell))
        #print(hidden.shape)
        #print(ci.shape)
        #print(x.shape)
        h = torch.cat((hidden,ci,p),dim=2)
        Pgen = self.sigmoid(self.pgen(h))
        #print(Pgen)
        Pw = torch.zeros(cnt)
        
        for i in range(1,cnt):
            sum = 0
            #print(Dict[i])
            if Dict[i] in dicting:
                fggds = dicting[Dict[i]]
                #print(dicting[Dict[i]])
            else:
                dicting[Dict[i]] = []
            for j in range(len(dicting[Dict[i]])):
                #print(t1[dicting[Dict[i]][j]])
                sum = sum + t1[dicting[Dict[i]][j]]
                #print(sum)
            #time.sleep(10)
            #print(Pw[i-1])
            #print(Pvocab[0][0][i-1])
            Pw[i-1] = Pgen*Pvocab[i-1] + (1-Pgen)*sum
            # print(Pvocab[i-1])
            # print(sum)
        
        predictions = self.fc(outputs[0][0])         
        return predictions,hidden,cell,Pw,sum_attn
    
decoder = Decoder()

class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,source,target,features):
        batch_size = source.shape[0]
        #print(target.shape)
        # N, tar_len, 100
        target_len = target.shape[1]
        outputs = torch.zeros(batch_size,target_len,cnt)
        encoder_states,(hn,cn) = self.encoder(source)
        # N, seq_len,200
        x = "SOS"
        hidden = hn
        cell = cn
        
        x = hidden[0][0]
        y = hidden[1][0]
        hidden = torch.cat((x,y),dim=0)
        hidden1 = torch.zeros(1,200)
        hidden1[0] = hidden
        hidden2 = torch.zeros(1,1,200)
        hidden2[0] = hidden1
        hidden = hidden2 
        x = cell[0][0]
        y = cell[1][0]
        cell = torch.cat((x,y),dim=0)
        cell1=torch.zeros(1,200)
        cell1[0] = cell
        cell2 = torch.zeros(1,1,200)
        cell2[0] = cell1
        cell = cell2        
        x = getEmbeddings(x)
        p = torch.zeros(1,100)
        p[0] = x
        dicting = {}
        cnt2 = 0
        for i in features.split():
            if i not in dicting:
                dicting[i] = []
                dicting[i].append(cnt2)
            else:
                dicting[i].append(cnt2)
            cnt2 = cnt2 + 1
            
        sum_attn = torch.zeros(1,1,len(encoder_states[0]))  
        sum_attn1 = torch.zeros(1,1,len(encoder_states[0]))     
        for t in range(1,target_len):
            arr = []    
                        
            output,hidden,cell,Pw,sum_attn =self.decoder(p,encoder_states,hidden,cell,dicting,sum_attn)
            sum = 0
            for j in range(len(sum_attn[0][0])):
                sum = sum + min(sum_attn[0][0][j],sum_attn1[0][0][j])
            outputs[0][t] = Pw
            p = target[0][t-1]
            p.unsqueeze(0)
            
        return outputs,sum
            




dataset1 = wordsDataset()
dataloader = DataLoader(dataset1,batch_size=1,shuffle = True)
dataiter = iter(dataloader)
data = dataiter.next()

features, labels = data
loss = nn.NLLLoss()
Final_model = Seq2Seq(encoder=encoder,decoder=decoder)

optimiser = torch.optim.SGD(Final_model.parameters(), lr=0.1)
for i,(features, labels) in enumerate(dataloader):
    if i > 10:
        break
    input = torch.zeros(1,len(features[0].split()),100)
    cnt1 = 0
    for p in features[0].split():
        try:
            x = p.lower()
        except:
            x = p
        t = getEmbeddings(x)
        input[0][cnt1] = t
        cnt1 = cnt1+1
    output = torch.zeros(1,len(labels[0].split()),100)
    cnt1 = 0
    for p in labels[0].split():
        try:
            x= p.lower()
        except:
            x = p
        t = getEmbeddings(x)
        output[0][cnt1] = t
        cnt1 = cnt1 + 1
    outputs = Final_model(input,output,features[0])
    print(outputs)
    #time.sleep(10)
    # for i in range(len(outputs[0])):
    #     for j in range(len(outputs[0][i])):
    #         #print(outputs[0][i][j],end = " ")
    #     #print("")
    #     #time.sleep(10)
    #p = torch.argmax(outputs[0],dim=1)
    #print(p)
    #time.sleep(10)
    #print(len(output[0]))
    #print(len(outputs[0]))
    #print(len(p))
    lossing = 0
    words = labels[0].split()
    target = torch.zeros(len(words),dtype=torch.long)
    for j in range(len(words)):
        print(words[j])
        if words[j] in dict.keys():
            #print(dict[words[j]])
            target[j] = dict[words[j]]-1
        else:
            target[j] = dict["UNK"]-1
    print(target.dtype)
    print(target)
    prob = torch.zeros(len(words))
    for j in range(len(words)):
        prob[j] = outputs[0][j][target[j]]
        if prob[j] == 0:
            prob[j] = 0.0001
    #outputs[0] = nn.LogSoftmax(outputs[0])
    for j in range(len(outputs[0])):
        lossing = lossing + torch.sum(-1*torch.log(prob))/len(words)
    lossing = lossing + 1*sum
    # for j in range(len(outputs[0])):
    #     x = outputs[0][j]
    #     #print(int(x))
    #     #print(dict[words[j]])
    #     print(x)
    #     if words[j] in Dict:
    #         lossing = lossing + loss(x,dict[words[j]])
    #     else:
    #         lossing = lossing + loss(x,dict["UNK"])
    #     print(Dict[int(x)+1],end = "")
    #     print(" ",end="")
    print(lossing)
    lossing.backward()
    optimiser.step()
    optimiser.zero_grad()
        
    time.sleep(1)
    
model.save("model2.pt")
