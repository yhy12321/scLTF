import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ======================================================
# Transformer-based model for cell representation learning and clustering
# Includes model definition, training, evaluation, early stopping, and prediction
# ======================================================



class TransformerModel(nn.Module):
    """
        Transformer encoder model for cell-level representation learning
        and downstream cell type / cluster classification.
        """
    def __init__(self, input_dim, num_classes, num_heads, num_layers, d_model=128, dropout=0.4):
        super(TransformerModel, self).__init__()
        # Project input gene expression features to Transformer hidden dimension
        self.input_linear = nn.Linear(input_dim, d_model)
        self.d_model = d_model
        # Layer normalization to stabilize training
        self.layer_norm = nn.LayerNorm(self.d_model)
        # Single Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=num_heads, dropout=dropout)
        # Stacked Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # Final classification layer
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = self.input_linear(x)            # Feature projection
        x = self.layer_norm(x)              # Normalization
        x = self.transformer_encoder(x)     # Transformer encoding
        x = self.classifier(x)              # Classification
        return x

def validate_model(model, val_loader, criterion, device):
    """
        Compute average validation loss.
        """
    model.eval()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_count += inputs.size(0)

    average_loss = total_loss / total_count
    return average_loss

def evaluate_model(model, val_loader, criterion, device):
    """
        Evaluate model performance in terms of loss and classification accuracy.
        """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    average_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    return average_loss, accuracy

def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    """
        Train the Transformer model using mini-batch gradient descent.
        """
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

class EarlyStopping:
    """
        Early stopping strategy to prevent overfitting based on validation loss.
        """
    def __init__(self, patience=5, delta=0, verbose=False, path='best_model.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = 'best_model.pt'
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)

class GeneExpressionDataset(Dataset):
    """
        PyTorch Dataset for gene expression features and corresponding labels.
        """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def predict_full_dataset(model, data, cell_names, file_path, device):
    """
        Predict cluster labels for all cells and save results to a CSV file.
        """
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in DataLoader(data, batch_size=32):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    predicted_clusters_df = pd.DataFrame({
        'Cell Name': cell_names,
        'Predicted Cluster': predictions
    })
    predicted_clusters_df.to_csv(file_path, index=False)