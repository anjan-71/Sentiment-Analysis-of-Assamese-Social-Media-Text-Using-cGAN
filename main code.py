import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
import matplotlib.pyplot as plt

nltk.download('punkt')


# --- Load Assamese Dataset ---
df = pd.read_excel(r"C:\Users\Anjan Boro\OneDrive\Desktop\Main Paper\Program\Dataset Positive Negative.xlsx")
 # Change filename
df.dropna(inplace=True)

# --- Preprocess Text ---


def preprocess(texts):
    tokens = [word_tokenize(str(t).lower()) for t in texts]
    all_words = [word for sent in tokens for word in sent]
    vocab = {word: i+2 for i,
             (word, _) in enumerate(Counter(all_words).items())}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    encoded = [[vocab.get(word, 1) for word in sent] for sent in tokens]
    return encoded, vocab


df['label'] = df['label'].astype(int)
encoded_texts, vocab = preprocess(df['text'])
max_len = max(len(x) for x in encoded_texts)
padded = [x + [0]*(max_len-len(x)) for x in encoded_texts]

# --- Dataset ---


class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.x = torch.tensor(texts, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]


x_train, x_test, y_train, y_test = train_test_split(
    padded, df['label'].tolist(), test_size=0.2)
train_data = SentimentDataset(x_train, y_train)
test_data = SentimentDataset(x_test, y_test)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# --- Generator ---


class Generator(nn.Module):
    def __init__(self, noise_dim, embed_dim, vocab_size, hidden_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        self.fc = nn.Linear(noise_dim, max_len * embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, noise):
        x = self.fc(noise).view(-1, max_len, self.embed_dim)
        x, _ = self.rnn(x)
        logits = self.out(x)
        return logits

# --- Discriminator ---


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.real_fake = nn.Linear(hidden_dim, 1)
        self.sentiment = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.rnn(x)
        h = h.squeeze(0)
        real_fake = torch.sigmoid(self.real_fake(h))
        sentiment = torch.sigmoid(self.sentiment(h))
        return real_fake, sentiment


# --- Initialize ---
vocab_size = len(vocab)
embed_dim = 64
hidden_dim = 128
noise_dim = 100
G = Generator(noise_dim, embed_dim, vocab_size, hidden_dim)
D = Discriminator(vocab_size, embed_dim, hidden_dim)

bce = nn.BCELoss()
g_opt = optim.Adam(G.parameters(), lr=0.0002)
d_opt = optim.Adam(D.parameters(), lr=0.0002)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G, D = G.to(device), D.to(device)

epoch_accuracies = []
d_losses = []
g_losses = []


# --- Training ---
for epoch in range(10):
    total_correct = 0
    total_samples = 0
    epoch_d_loss = 0
    epoch_g_loss = 0

    for real_x, labels in train_loader:
        real_x, labels = real_x.to(device), labels.to(device).unsqueeze(1)

        # --- Train Discriminator ---
        noise = torch.randn(real_x.size(0), noise_dim).to(device)
        fake_logits = G(noise).argmax(-1).detach()

        D.zero_grad()
        real_validity, real_sent = D(real_x)
        fake_validity, _ = D(fake_logits)

        d_loss = bce(real_validity, torch.ones_like(real_validity)) + \
            bce(fake_validity, torch.zeros_like(fake_validity)) + \
            bce(real_sent, labels)
        d_loss.backward()
        d_opt.step()

        # --- Train Generator ---
        noise = torch.randn(real_x.size(0), noise_dim).to(device)
        fake_logits = G(noise).argmax(-1)
        G.zero_grad()
        fake_validity, fake_sent = D(fake_logits)
        g_loss = bce(fake_validity, torch.ones_like(fake_validity))
        g_loss.backward()
        g_opt.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()

        total_correct += (real_sent > 0.5).eq(labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    epoch_accuracies.append(accuracy)
    d_losses.append(epoch_d_loss / len(train_loader))
    g_losses.append(epoch_g_loss / len(train_loader))

    print(
        f"Epoch {epoch+1}/10 | Accuracy: {accuracy:.4f} | D Loss: {d_losses[-1]:.4f} | G Loss: {g_losses[-1]:.4f}")


# --- Evaluation ---
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        _, sentiment_preds = D(x)
        preds = (sentiment_preds > 0.5).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

print("\nClassification Report:\n", classification_report(
    all_labels, all_preds, target_names=["Negative", "Positive"]))


# --- Plot Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                       "Negative", "Positive"]).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
