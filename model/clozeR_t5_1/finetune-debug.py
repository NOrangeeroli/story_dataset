from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
from tools import start_debugger_on_exception
from dataset import DataSetBert
import numpy as np
import torchtext
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
start_debugger_on_exception()
train_dataset = DataSetBert(data_file= './data/IMDBs.csv')
val_dataset = DataSetBert(data_file= './data/IMDBs.csv')
from torch.utils.data import DataLoader
device = torch.device('cuda:6') 
train_dataloader = DataLoader(train_dataset, batch_size=11, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=11, shuffle=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)  
model.train()
model.to(device)  
from transformers import AdamW
MAX_SEQ_LEN = 128
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                           fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('index', label_field), ('text', text_field), ('label', label_field)]

# TabularDataset

train, valid, test = TabularDataset.splits(path='./', train='data/IMDBs.csv', validation='data/IMDBs.csv',
                                                   test='data/IMDBs.csv', format='CSV', fields=fields, skip_header=True)

# Iterators

train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.text),
                                    device=device, train=True, sort=True, sort_within_batch=True)
optimizer = AdamW(model.parameters(), lr=1e-5)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
epochs = 40
#optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps = total_steps)
for epoch in range(epochs):
    losses = []
    accs = []
    for (_,train_features,train_labels),_ in train_iter:
        #train_features = batch['source']
        #train_labels = batch['target']
        #encoding = tokenizer(train_features, return_tensors='pt', padding=True, truncation=True)

        #input_ids = encoding['input_ids']
        input_ids = train_features
        #attention_mask = encoding['attention_mask']
        input_ids=input_ids.to(device)  
        train_labels= train_labels.type(torch.LongTensor)
        train_labels=train_labels.to(device)  
        #attention_mask=attention_mask.to(device)
        
        outputs = model(input_ids, labels = train_labels)
        # import pdb;pdb.set_trace()
        
        loss = outputs.loss
        optimizer.zero_grad()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        _, predicted = torch.max(outputs.logits, 1)
        acc = (predicted == train_labels).sum().item()/len(train_labels)
        accs.append(acc)
    train_loss = np.mean(losses)
    train_acc = np.mean(accs)
    print(f'epoch: {epoch} loss {train_loss} acc {train_acc}')
