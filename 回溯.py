import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import json
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义Dataset类
class EventDetectionDataset(Dataset):
    def __init__(self, data, word_to_idx, max_len=512):
        self.data = data
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        events = item['event_chain']

        # 使用词袋模型或字符嵌入
        input_ids = self.text_to_ids(text)

        # Convert event chain to a one-hot encoding for multi-label classification
        event_labels = self.get_event_labels(events)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'event_labels': torch.tensor(event_labels, dtype=torch.float)
        }

    def text_to_ids(self, text):
        words = text.split()  # 假设以空格为分词依据
        input_ids = [self.word_to_idx.get(word, 0) for word in words]  # 如果词不在词典中，使用0（未知词）
        input_ids = input_ids[:self.max_len]  # 限制最大长度
        input_ids += [0] * (self.max_len - len(input_ids))  # 填充至最大长度
        return input_ids

    def get_event_labels(self, events):
        event_types = ['Marry', 'Be_Born', 'Divorce', 'Child_Custody', 'Alimony', 'Property_Distribution']
        event_labels = [1 if event in events else 0 for event in event_types]
        return event_labels


# 定义简单的Embedding + LSTM模型
class EventDetectionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_labels):
        super(EventDetectionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=256, num_layers=1, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(256 * 2, num_labels)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)  # shape: (batch_size, seq_len, embedding_dim)
        lstm_output, (h_n, c_n) = self.lstm(embedded)
        last_hidden_state = lstm_output[:, -1, :]  # shape: (batch_size, hidden_size*2)
        logits = self.fc(last_hidden_state)  # shape: (batch_size, num_labels)
        return logits


# 训练过程
def train_model(model, train_loader, val_loader, epochs=3, lr=2e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            event_labels = batch['event_labels'].to(device)

            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits, event_labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluate after each epoch
        val_accuracy = evaluate_model(model, val_loader)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Accuracy: {val_accuracy}')

    # Plot training and validation accuracy
    plot_training_results(train_losses, val_accuracies)


def evaluate_model(model, val_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            event_labels = batch['event_labels'].to(device)

            logits = model(input_ids)
            preds = torch.sigmoid(logits) > 0.5  # Multi-label thresholding

            all_preds.append(preds.cpu().numpy())
            all_labels.append(event_labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy


def plot_training_results(train_losses, val_accuracies):
    # Plot the training loss and validation accuracy over epochs
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots()

    # Plot train loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(epochs, train_losses, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for validation accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:blue')
    ax2.plot(epochs, val_accuracies, color='tab:blue', label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Title and legend
    plt.title('Training Loss and Validation Accuracy Over Epochs')
    fig.tight_layout()
    plt.show()


# 预测过程
def predict_model(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)

            logits = model(input_ids)
            preds = torch.sigmoid(logits) > 0.5  # Multi-label thresholding
            predictions.append(preds.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    return predictions


# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


# 输出提交格式
def generate_submission(predictions, test_data, output_file='submission.json'):
    event_types = ['Marry', 'Be_Born', 'Divorce', 'Child_Custody', 'Alimony', 'Property_Distribution']
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (item, pred) in enumerate(zip(test_data, predictions)):
            event_chain = [event_types[idx] for idx, val in enumerate(pred) if val == 1]
            submission_item = {
                'id': item['id'],
                'text': item['text'],
                'event_chain': event_chain
            }
            f.write(json.dumps(submission_item, ensure_ascii=False) + '\n')


# 主流程
if __name__ == "__main__":
    # 加载数据
    train_data = load_data('train_dataset.json')
    val_data = load_data('dev_dataset.json')
    test_data = load_data('test_dataset.json')

    # 创建词汇表
    word_to_idx = {}
    for data in train_data + val_data + test_data:
        for word in data['text'].split():
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx) + 1  # Start index from 1

    # 创建数据集和数据加载器
    train_dataset = EventDetectionDataset(train_data, word_to_idx)
    val_dataset = EventDetectionDataset(val_data, word_to_idx)
    test_dataset = EventDetectionDataset(test_data, word_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # 初始化模型
    model = EventDetectionModel(vocab_size=len(word_to_idx) + 1, embedding_dim=100, num_labels=6).to(device)

    # 训练模型
    train_model(model, train_loader, val_loader)

    # 预测并生成提交文件
    predictions = predict_model(model, test_loader)
    generate_submission(predictions, test_data)
