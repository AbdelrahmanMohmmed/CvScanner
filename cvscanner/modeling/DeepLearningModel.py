import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
import streamlit as st

# ---------------- CV CLASSIFIER ----------------
class CVClassifier(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=10, dropout_rate=0.3):
        super(CVClassifier, self).__init__()
        self.pretrained_model = AutoModel.from_pretrained(model_name)

        # Freeze all layers initially
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers for fine-tuning
        for param in self.pretrained_model.encoder.layer[-4:].parameters():
            param.requires_grad = True

        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.pretrained_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ---------------- DATASET ----------------
class CVDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ---------------- DATA PREP ----------------
def prepare_data_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        category_data = json.load(f)
    texts, labels = [], []
    for category, resumes in category_data.items():
        for resume in resumes:
            texts.append(resume)
            labels.append(category)
    return texts, labels

# ---------------- TRAINING ----------------
def train_deep_learning_model(json_file_path, model_name='bert-base-uncased', 
                              num_epochs=10, batch_size=16, learning_rate=2e-5):
    texts, labels = prepare_data_from_json(json_file_path)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)

    st.write(f"Found {len(texts)} resumes across {num_classes} categories")
    st.write("Categories:", list(label_encoder.classes_))

    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = CVClassifier(model_name=model_name, num_classes=num_classes)

    train_dataset = CVDataset(train_texts, train_labels, tokenizer)
    val_dataset = CVDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: {device}")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct, total, val_loss = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        st.write(f"**Epoch {epoch+1}/{num_epochs}** | "
                 f"Train Loss: {avg_train_loss:.4f} | "
                 f"Val Loss: {avg_val_loss:.4f} | "
                 f"Accuracy: {accuracy:.2f}%")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'accuracy': accuracy,
                'label_encoder': label_encoder,
                'tokenizer_name': model_name,
                'num_classes': num_classes
            }, 'best_cv_classifier.pth')
            st.success(f"Saved new best model with accuracy: {accuracy:.2f}%")

    st.success(f"Training complete! Best accuracy: {best_accuracy:.2f}%")
    return model, label_encoder, tokenizer

# ---------------- PREDICTION ----------------
def predict_cv_category(cv_text, model, label_encoder, tokenizer, device='cpu'):
    model.eval()
    model.to(device)
    encoding = tokenizer(cv_text, truncation=True, padding='max_length',
                         max_length=512, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    predicted_category = label_encoder.inverse_transform([predicted_class])[0]

    # Top predictions
    top_probs, top_indices = torch.topk(probabilities[0], k=min(5, len(label_encoder.classes_)))
    categories = label_encoder.inverse_transform(top_indices.cpu().numpy())
    confidences = top_probs.cpu().numpy()

    top_predictions = [(category, float(conf)) for category, conf in zip(categories, confidences)]

    return predicted_category, confidence, top_predictions

# ---------------- LOAD TRAINED MODEL ----------------
def load_trained_model(model_path):
    """
    Load a trained model for inference safely with PyTorch >=2.6
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from torch.serialization import safe_globals

    # Add safe globals that may exist in the checkpoint
    allowed_globals = [LabelEncoder, np._core.multiarray._reconstruct]

    with safe_globals(allowed_globals):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    num_classes = checkpoint['num_classes']
    model_name = checkpoint.get('tokenizer_name', 'bert-base-uncased')
    model = CVClassifier(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    label_encoder = checkpoint['label_encoder']
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, label_encoder, tokenizer

# ---------------- QUICK TRAIN ----------------
def quick_train(json_file_path, epochs=3):
    """Quick training for testing"""
    return train_deep_learning_model(
        json_file_path=json_file_path,
        model_name='distilbert-base-uncased',
        num_epochs=epochs,
        batch_size=8,
        learning_rate=3e-5
    )
