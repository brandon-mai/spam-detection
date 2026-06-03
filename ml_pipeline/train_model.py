import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MLPValidator(nn.Module):
    def __init__(self, in_features=24, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
    def forward(self, x):
        return self.net(x)

def get_numpy_weights(model):
    w = {}
    for name, param in model.named_parameters():
        np_name = name.replace("net.", "").replace(".", "_")
        w[np_name] = param.detach().cpu().numpy()
    return w

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load data
    data = np.load("scripts/shot_dataset.npz")
    feats = data["features"]
    labels = data["labels"].astype(np.float32)
    meta = data["meta_game"]

    games = np.unique(meta)
    np.random.seed(42)
    np.random.shuffle(games)
    split_idx = int(len(games) * 0.8)
    train_games = set(games[:split_idx])
    
    train_mask = np.array([g in train_games for g in meta])
    val_mask = ~train_mask

    X_tr, y_tr = feats[train_mask], labels[train_mask]
    X_va, y_va = feats[val_mask], labels[val_mask]

    print(f"  train: {len(X_tr)} shots ({len(train_games)} games), val: {len(X_va)} shots ({len(games)-len(train_games)} games)")
    print(f"  train pos: {y_tr.mean()*100:.1f}%, val pos: {y_va.mean()*100:.1f}%")

    ds_tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    ds_va = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32).unsqueeze(1))
    
    loader_tr = DataLoader(ds_tr, batch_size=256, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=512, shuffle=False)

    pos_w = (1.0 - y_tr.mean()) / max(y_tr.mean(), 1e-5)
    print(f"  pos_weight (neg/pos): {pos_w:.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(device))

    ensemble_weights = {}
    N_MODELS = 5
    EPOCHS = 60
    SEEDS = [42, 100, 7, 999, 123]

    for m_idx in range(N_MODELS):
        seed = SEEDS[m_idx]
        print(f"\\n--- Training model {m_idx+1}/{N_MODELS} (seed={seed}) ---")
        torch.manual_seed(seed)
        
        model = MLPValidator(hidden=128).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        best_acc = 0.0
        best_state = None
        
        for ep in range(1, EPOCHS + 1):
            model.train()
            losses = []
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            if ep % 10 == 0 or ep == EPOCHS:
                model.eval()
                val_probs = []
                val_y = []
                with torch.no_grad():
                    for bx, by in loader_va:
                        bx = bx.to(device)
                        probs = torch.sigmoid(model(bx)).cpu().numpy()
                        val_probs.extend(probs)
                        val_y.extend(by.numpy())
                
                val_probs = np.array(val_probs).flatten()
                val_y = np.array(val_y).flatten()
                preds = (val_probs >= 0.30).astype(int)
                acc = (preds == val_y).mean()
                
                if acc > best_acc:
                    best_acc = acc
                    best_state = get_numpy_weights(model)
                
                print(f"  epoch {ep:3d} | t_loss={np.mean(losses):.4f} v_acc(0.3)={acc:.3f} best={best_acc:.3f}")

        # save best
        for k, v in best_state.items():
            ensemble_weights[f"m{m_idx}_{k}"] = v

    ensemble_weights["n_models"] = np.array([N_MODELS])
    
    out_path = "weights/hellburner_v2/weights.npz"
    np.savez_compressed(out_path, **ensemble_weights)
    print(f"\\nSaved {out_path}")
