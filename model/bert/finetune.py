from transformers import BertForSequenceClassification
import torch
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
model.train()
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
text_batch = ["我是好人", "我是坏人"]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
labels = torch.tensor([1,0]).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
