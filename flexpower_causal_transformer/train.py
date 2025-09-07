import os, yaml, math, random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data.dataset import DummyFlexDataset
from models.causal_transformer import CausalTransformerFlex
from utils.metrics import binary_classification_metrics
from utils.scheduler import WarmupCosineLR

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def focal_loss_with_logits(logits, targets, gamma=2.0, pos_weight=1.0, reduction="mean"):
    # logits: [B], targets: [B] in {0,1}
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = torch.sigmoid(logits)
    pt = p*targets + (1-p)*(1-targets)
    loss = ((1-pt)**gamma) * bce
    return loss.mean() if reduction == "mean" else loss.sum()

def collate(batch):
    xs, metas, ys = zip(*batch)
    x = torch.stack(xs)                     # [B, L, C]
    y = torch.stack(ys).squeeze(-1)         # [B]
    meta = {}
    for k in metas[0].keys():
        meta[k] = torch.stack([m[k] for m in metas]).view(len(batch), 1)  # [B,1]
    return x, meta, y

def main(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    # Data
    ds = DummyFlexDataset(
        size=cfg["train_size"] + cfg["val_size"] + cfg["test_size"],
        seq_len=cfg["seq_len"],
        num_cont=cfg["num_cont"],
        num_stations=cfg["num_stations"],
        num_receivers=cfg["num_receivers"],
        num_antennas=cfg["num_antennas"],
        num_constellations=cfg["num_constellations"],
        num_prns=cfg["num_prns"],
        seed=cfg["seed"],
    )
    n_train = cfg["train_size"]
    n_val = cfg["val_size"]
    n_test = cfg["test_size"]
    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [n_train, n_val, n_test],
                                                              generator=torch.Generator().manual_seed(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)
    test_loader  = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)

    # Model
    model = CausalTransformerFlex(
        num_cont=cfg["num_cont"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        use_film=cfg["use_film"],
        film_dim=cfg["film_dim"],
        num_stations=cfg["num_stations"],
        num_receivers=cfg["num_receivers"],
        num_antennas=cfg["num_antennas"],
        num_constellations=cfg["num_constellations"],
        num_prns=cfg["num_prns"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    total_steps = cfg["epochs"] * math.ceil(n_train / cfg["batch_size"])
    sched = WarmupCosineLR(opt, warmup_steps=cfg["warmup_steps"], total_steps=total_steps, min_lr=1e-6)

    best_val_f1, best_path = -1, "best_model.pt"

    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        total_loss = 0.0
        for step, (x, meta, y) in enumerate(train_loader, 1):
            x = x.to(device)
            y = y.to(device)
            for k in meta:
                meta[k] = meta[k].to(device)

            logits = model(x, meta)  # [B]
            if cfg["use_focal"]:
                loss = focal_loss_with_logits(logits, y, gamma=cfg["focal_gamma"], pos_weight=cfg["pos_weight"])
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for x, meta, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                for k in meta:
                    meta[k] = meta[k].to(device)
                logits = model(x, meta)
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)
        m = binary_classification_metrics(all_logits, all_targets, threshold=cfg["threshold"])
        print(f"Epoch {epoch:02d} | TrainLoss {total_loss/step:.4f} | Val F1 {m['f1']:.4f} Acc {m['acc']:.4f} P {m['precision']:.4f} R {m['recall']:.4f}")

        if m['f1'] > best_val_f1:
            best_val_f1 = m['f1']
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)

    print(f"Best Val F1: {best_val_f1:.4f}, saved to {best_path}")

    # Test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, meta, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            for k in meta:
                meta[k] = meta[k].to(device)
            logits = model(x, meta)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    m = binary_classification_metrics(all_logits, all_targets, threshold=cfg["threshold"])
    print(f"[TEST] F1 {m['f1']:.4f} Acc {m['acc']:.4f} P {m['precision']:.4f} R {m['recall']:.4f}")

if __name__ == "__main__":
    main()
