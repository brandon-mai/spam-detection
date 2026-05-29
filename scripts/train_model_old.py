import numpy as np
import torch
import torch.nn as nn
import sys

class ShotValidator(nn.Module):
    def __init__(self, in_dim=24, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def _select_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), "cuda not available"
    try:
        _ = torch.zeros(4, device="cuda") + 1
        torch.cuda.synchronize()
        return torch.device("cuda"), "cuda OK"
    except Exception as e:
        return torch.device("cpu"), f"cuda failed ({type(e).__name__}: {str(e)[:60]}), using CPU"

if __name__ == "__main__":
    device, why = _select_device()
    print(f"Training on: {device} ({why})")

    try:
        data = np.load("shot_dataset.npz")
    except Exception as e:
        print("Failed to load shot_dataset.npz:", e)
        sys.exit(1)

    feats = data["features"]
    labels = data["labels"]
    meta_game = data["meta_game"]

    rng = np.random.default_rng(42)
    games = np.unique(meta_game)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * 0.2))
    val_games = set(games[:n_val].tolist())
    val_mask = np.array([g in val_games for g in meta_game], dtype=bool)
    Xt, yt = feats[~val_mask], labels[~val_mask]
    Xv, yv = feats[val_mask], labels[val_mask]
    print(f"  train: {len(Xt)} shots ({len(games)-n_val} games), val: {len(Xv)} shots ({n_val} games)")
    print(f"  train pos: {yt.mean()*100:.1f}%, val pos: {yv.mean()*100:.1f}%")

    pr = max(yt.mean(), 1e-6)
    pos_weight = torch.tensor([(1.0 - pr) / pr], device=device)
    print(f"  pos_weight (neg/pos): {pos_weight.item():.3f}")

    Xt_t = torch.from_numpy(Xt).to(device)
    yt_t = torch.from_numpy(yt).to(device).float()
    Xv_t = torch.from_numpy(Xv).to(device)
    yv_t = torch.from_numpy(yv).to(device).float()

    EPOCHS = 40
    BATCH = 512
    ENSEMBLE_SEEDS = [42, 100, 7]

    trained_models = []
    for mi, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\\n--- Training model {mi+1}/{len(ENSEMBLE_SEEDS)} (seed={seed}) ---")
        torch.manual_seed(seed)
        model = ShotValidator(in_dim=24, hidden=64).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_acc03 = 0
        best_sd = None
        for epoch in range(1, EPOCHS + 1):
            model.train()
            idx = torch.randperm(len(Xt_t), device=device)
            losses = []
            for i in range(0, len(idx), BATCH):
                b = idx[i:i+BATCH]
                logits = model(Xt_t[b])
                loss = crit(logits, yt_t[b])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss))
            tl = float(np.mean(losses))
            model.eval()
            with torch.no_grad():
                vlogits = model(Xv_t)
                vprob = torch.sigmoid(vlogits)
                v_acc05 = float(((vprob > 0.5).float() == yv_t).float().mean())
                v_acc03 = float(((vprob > 0.3).float() == yv_t).float().mean())
            if v_acc03 > best_acc03:
                best_acc03 = v_acc03
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 10 == 0 or epoch == EPOCHS:
                print(f"  epoch {epoch:3d} | t_loss={tl:.4f} v_acc(0.3)={v_acc03:.3f} best={best_acc03:.3f}")
        model.load_state_dict(best_sd)
        trained_models.append(model)

    weights = {"n_models": np.array([len(trained_models)], dtype=np.int32)}
    for mi, model in enumerate(trained_models):
        model.eval()
        sd = model.state_dict()
        weights[f"m{mi}_0_w"] = sd["net.0.weight"].cpu().numpy().astype(np.float32)
        weights[f"m{mi}_0_b"] = sd["net.0.bias"].cpu().numpy().astype(np.float32)
        weights[f"m{mi}_2_w"] = sd["net.2.weight"].cpu().numpy().astype(np.float32)
        weights[f"m{mi}_2_b"] = sd["net.2.bias"].cpu().numpy().astype(np.float32)
        weights[f"m{mi}_4_w"] = sd["net.4.weight"].cpu().numpy().astype(np.float32)
        weights[f"m{mi}_4_b"] = sd["net.4.bias"].cpu().numpy().astype(np.float32)

    np.savez("weights.npz", **weights)
    print("Saved weights.npz")
