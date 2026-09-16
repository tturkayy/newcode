import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OpticalAPSKDataset
from model import APSKDemodulator1D


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Veri Kümeleri
    train_dataset = OpticalAPSKDataset(num_samples=25000, symbol_len=24)
    val_dataset = OpticalAPSKDataset(num_samples=5000, symbol_len=24)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # Model, Kayıp Fonksiyonu ve Optimizer
    model = APSKDemodulator1D(in_channels=1, symbol_len=24, num_classes=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    epochs = 15
    print("\n--- 16-APSK Neural Demodulator Eğitimi Başlıyor ---")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_acc = (correct_train / total_train) * 100.0
        train_loss /= total_train

        # Doğrulama (Validation)
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                preds = torch.argmax(outputs, dim=1)
                correct_val += (preds == y_batch).sum().item()
                total_val += y_batch.size(0)

        val_acc = (correct_val / total_val) * 100.0
        ser = 1.0 - (correct_val / total_val) # Symbol Error Rate

        scheduler.step(val_acc)

        print(f"Epoch [{epoch:02d}/{epochs}] | Kayıp: {train_loss:.4f} | Egt Dogruluk: %{train_acc:.2f} | Val Dogruluk: %{val_acc:.2f} | SER: {ser:.4e}")

    # Ağırlıkları Kaydet
    torch.save(model.state_dict(), "apsk_demodulator.pth")
    print("\nModel ağırlıkları kaydedildi: apsk_demodulator.pth")


if __name__ == "__main__":
    train()