from transformers import RobertaForSequenceClassification,RobertaConfig,BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
from tools import start_debugger_on_exception
from dataset import DataSetBert
import numpy as np
start_debugger_on_exception()
train_dataset = DataSetBert(data_file= './data/data_train/train.csv')
val_dataset = DataSetBert(data_file= './data/data_train/val.csv')
test_dataset = DataSetBert(data_file= './data/data_train/test.csv')
from torch.utils.data import DataLoader
device = torch.device('cuda:3') 
train_dataloader = DataLoader(train_dataset, batch_size=11, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=11, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=11, shuffle=True)
model = BertForSequenceClassification.from_pretrained('uer/roberta-base-chinese-extractive-qa')
from transformers import RobertaTokenizer, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('uer/roberta-base-chinese-extractive-qa')
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#import pdb;pdb.set_trace()
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.config.max_position_embeddings= 1024
model.to(device)  
model.train()
model.to(device)  
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
epochs = 400
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps = total_steps)
def step(model,dataloader,tokenizer,optimizer,if_train=True):
    losses = []
    accs = []
    nb = 0
    '''
    if if_train:
        model.train()
    else:
        model.eval()
    '''
    for batch in dataloader:
        # import pdb;pdb;set_trace()
        nb += 1
        train_features = batch['source']
        train_labels = batch['target']
        encoding = tokenizer(train_features, return_tensors='pt', padding=True, truncation=True)

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        input_ids=input_ids.to(device)
        train_labels=train_labels.to(device)
        attention_mask=attention_mask.to(device)

        outputs = model(input_ids, labels=train_labels)
        #import pdb;pdb.set_trace()

        loss = outputs[0]
        
        
        losses.append(loss.item())
        if if_train:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
        _, predicted = torch.max(outputs.logits, 1)
        acc = (predicted == train_labels).sum().item()/len(train_labels)
        if acc>1:
            import pdb;pdb.set_trace()
        accs.append(acc)
    return losses,accs
log = open('log','w')
for epoch in range(epochs):
    train_dataloader = DataLoader(train_dataset, batch_size=11, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=11, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=11, shuffle=True)
    model.train()
    losses,accs = step(model,train_dataloader,tokenizer,optimizer,if_train=True)
    train_loss = np.mean(losses)
    train_acc = np.mean(accs)
    model.eval()
    
    with torch.no_grad():
        val_losses, val_accs = step(model,val_dataloader,tokenizer,optimizer,if_train=False)
        test_losses, test_accs = step(model,test_dataloader,tokenizer,optimizer,if_train=False)
    val_loss = np.mean(val_losses)
    val_acc = np.mean(val_accs)
    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accs)
    log_line= f'epoch: {epoch} loss {train_loss} acc {train_acc} val_loss {val_loss} val_acc {val_acc} test_loss {test_loss} test_acc {test_acc}'
    log.write(log_line+'\n')
    print(log_line)
log.close()
