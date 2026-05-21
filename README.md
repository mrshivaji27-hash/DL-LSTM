# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The goal of this project is to develop a Long Short-Term Memory (LSTM) based model to perform Named Entity Recognition (NER) on text data. The model will identify and classify words in a sentence into predefined categories such as Person, Organization, Location, Date, Time, Money, Percent, Facility, and GPE. Using the NER dataset containing Word, POS, and Tag columns, the model will be trained to recognize entities and predict the correct tags for unseen text.
<img width="773" height="936" alt="image" src="https://github.com/user-attachments/assets/e5497985-db5d-4ba4-926f-c38562bcfb07" />


## DESIGN STEPS
### STEP 1: 

Load data, create word/tag mappings, and group sentences.

### STEP 2: 
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.


### STEP 3: 

Define dataset and DataLoader for batching.

### STEP 4: 


Build a bidirectional LSTM model for sequence tagging.
### STEP 5: 

Train the model over multiple epochs, tracking loss.

### STEP 6: 


Evaluate model accuracy, plot loss curves, and visualize predictions on a sample.


## PROGRAM

### Name: shivaji.k

### Register Number:212224233002

```python

   class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100, dropout=0.1):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)           
        x = self.dropout(x)
        x, _ = self.lstm(x)         
        out = self.fc(x)     
        return out
        

vocab_size = len(word2idx) + 1   
tagset_size = len(tag2idx)


    
        


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMTagger(len(word2idx) + 1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids)
                loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
                val_loss += loss.item()
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")
    return train_losses, val_losses



```
### OUTPUT

## Loss Vs Epoch Plot
<img width="833" height="625" alt="image" src="https://github.com/user-attachments/assets/926c5cee-9b87-43ca-b6ba-b37dd9f96f4d" />



### Sample Text Prediction
<img width="483" height="505" alt="image" src="https://github.com/user-attachments/assets/d71aa267-1539-4886-909c-4efc31be8bf8" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
