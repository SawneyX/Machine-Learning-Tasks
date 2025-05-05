# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from torch.optim import AdamW
from torch.nn import MSELoss

# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = 5  # TODO: Set the number of epochs

train_val = pd.read_csv("train.csv")
test_val = pd.read_csv("test_no_score.csv")

# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, data_frame, tokenizer=GPT2Tokenizer.from_pretrained('gpt2'), max_length=512):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.titles = data_frame['title'].values
        self.sentences = data_frame['sentence'].values
        self.scores = data_frame['score'].values if 'score' in data_frame.columns else None

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        title = self.titles[index]
        sentence = self.sentences[index]
        text = title + " " + sentence
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        sample = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        if self.scores is not None:
            score = self.scores[index]
            sample['score'] = torch.tensor(score, dtype=torch.float)
        return sample


train_dataset = ReviewDataset(train_val)
test_dataset = ReviewDataset(test_val)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=16, pin_memory=True)

# Additional code if needed

# TODO: Fill out MyModule
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.config = GPT2Config.from_pretrained('gpt2')
        self.regression_head = nn.Linear(self.config.hidden_size, 1)   #Only 1 output value

    def forward(self, sample):
        
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']
        
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states[:, -1, :]  # (batch_size, hidden_size)
        score = self.regression_head(pooled_output)
        return score



model = MyModule().to(DEVICE)

# TODO: Setup loss function, optimiser, and scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = MSELoss()
scheduler = None

model.train()
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in tqdm(train_loader, total=len(train_loader)):
        batch = batch.to(DEVICE)

        # TODO: Set up training loop
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        scores = batch['score'].to(DEVICE)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.squeeze(), scores)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item()}')



model.eval()
with torch.no_grad():
    results = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = batch.to(DEVICE)

        # TODO: Set up evaluation loop
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # We collect the predictions here
        results.append(outputs.cpu().numpy())
        

    with open("result.txt", "w") as f:
        for val in np.concatenate(results):
            f.write(f"{val}\n")
