import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

'''
Using mps to accelerate the training process since I compile the file
on my laptop whose operating system is MacOS.
'''
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


chars = "0123456789+=_"
vocab = {ch: i for i, ch in enumerate(chars)}
inv_vocab = {i: ch for ch, i in vocab.items()}
vocab_size = len(vocab)

pad_token = vocab["_"]

max_len = 12

def generate_sample():
    a = random.randint(0, 999)
    b = random.randint(0, 999)
    return f"{a:03d}+{b:03d}={a+b:04d}"

def encode(s):
    ids = [vocab[c] for c in s]
    return ids  # No more padding is needed.

def build_dataset(n_samples=30000):
    X, Y = [], []

    for _ in range(n_samples):
        s = generate_sample()
        ids = encode(s)

        x = ids[:-1]
        y = ids[1:]

        eq_pos = s.index("=")
        for i in range(eq_pos):
            y[i] = pad_token

        X.append(x)
        Y.append(y)

    return torch.tensor(X), torch.tensor(Y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)

        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, d_model=256, n_heads=4, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(T, device=x.device)
        pos = pos.unsqueeze(0).expand(B, T)

        x = self.embed(x) + self.pos_embed(pos)

        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)

        for block in self.blocks:
            x = block(x, mask)

        return self.fc(x)

def train_model():
    X, Y = build_dataset()

    model = TinyTransformerLM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 256
    epochs = 15

    X = X.to(device)
    Y = Y.to(device)

    losses = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X.size(0))

        total_loss = 0

        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i+batch_size]
            x_batch = X[idx]
            y_batch = Y[idx]

            logits = model(x_batch)

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y_batch.reshape(-1),
                ignore_index=pad_token
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (X.size(0) // batch_size)
        losses.append(avg_loss)

        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    return model, losses


def generate(model, prompt):
    model.eval()

    ids = [vocab[c] for c in prompt]

    for _ in range(4):  #The length for the output must be fixed in this case.
        x = torch.tensor(ids).unsqueeze(0).to(device)
        logits = model(x)

        next_token = logits[0, -1].argmax().item()
        ids.append(next_token)

    return ''.join(inv_vocab[i] for i in ids)

if __name__ == "__main__":
    model, losses = train_model()

    # plot the graph
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


    tests = [
        "123+456=",
        "110+590=",
        "045+067=",
        "0+0="
    ]

    print("\nTest Results:")
    for t in tests:
        print(f"{t} -> {generate(model, t)}")