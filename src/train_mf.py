"""
Phase 2 — PyTorch Multi-Domain Matrix Factorization
Joint training on all 3 domains simultaneously.

Usage (single domain):
    python train_mf.py --data data/processed/movies.csv --domain 0

Usage (joint — all 3 domains):
    python train_mf.py \
        --data data/processed/movies.csv data/processed/food.csv data/processed/books.csv \
        --domain 0 1 2

Domain IDs:
    0 = movies (MovieLens)
    1 = food   (Food Recipes)
    2 = books  (Amazon Books)
"""

import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',      nargs='+', default=['data/processed/movies.csv'],
                   help='One or more processed CSV files (unified schema)')
    p.add_argument('--domain',    nargs='+', type=int, default=[0],
                   help='Domain ID for each file (same order as --data)')
    p.add_argument('--out',       default='checkpoints/')
    p.add_argument('--n_factors', type=int, default=128, help='Embedding size')
    p.add_argument('--n_domains', type=int, default=4,   help='Total domains reserved')
    p.add_argument('--epochs',    type=int, default=20)
    p.add_argument('--batch',     type=int, default=4096)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--wd',        type=float, default=1e-5, help='Weight decay (L2 reg)')
    p.add_argument('--workers',   type=int, default=4)
    return p.parse_args()

# ── Data ──────────────────────────────────────────────────────────────────────

def load_domain(path, domain_id):
    """
    Load one processed CSV (unified schema: userId, itemId, rating, timestamp, domain_idx).
    Returns df with contiguous per-domain user/item indices and lookup maps.
    """
    print(f'  Loading {path} (domain={domain_id}) ...')
    df = pd.read_csv(path)

    all_users = df['userId'].unique()
    all_items = df['itemId'].unique()
    user2idx  = {u: i for i, u in enumerate(all_users)}
    item2idx  = {v: i for i, v in enumerate(all_items)}

    df['user_idx']   = df['userId'].map(user2idx)
    df['item_idx']   = df['itemId'].map(item2idx)
    df['domain_idx'] = domain_id

    print(f'    Users: {len(all_users):,}  Items: {len(all_items):,}  Ratings: {len(df):,}')
    return df, len(all_users), len(all_items), user2idx, item2idx


def load_and_split_all(data_paths, domain_ids):
    """
    Load all domains, apply global index offsets so user/item IDs don't collide,
    time-split each domain 80/20, then concatenate into joint train/test sets.
    """
    assert len(data_paths) == len(domain_ids), '--data and --domain must have same number of entries'

    print('Loading datasets ...')
    all_train, all_test = [], []
    domain_meta = {}
    user_offset = 0
    item_offset = 0

    for path, domain_id in zip(data_paths, domain_ids):
        df, n_users, n_items, user2idx, item2idx = load_domain(path, domain_id)

        # Shift indices so each domain occupies a unique range
        df['user_idx'] += user_offset
        df['item_idx'] += item_offset

        # Time-based 80/20 split per domain
        df = df.sort_values('timestamp').reset_index(drop=True)
        split = int(len(df) * 0.8)
        all_train.append(df.iloc[:split].copy())
        all_test.append(df.iloc[split:].copy())

        domain_meta[domain_id] = {
            'n_users':     n_users,
            'n_items':     n_items,
            'user_offset': user_offset,
            'item_offset': item_offset,
            'user2idx':    user2idx,
            'item2idx':    item2idx,
        }
        user_offset += n_users
        item_offset += n_items

    train_df = pd.concat(all_train, ignore_index=True)
    test_df  = pd.concat(all_test,  ignore_index=True)

    print(f'\nJoint dataset:')
    print(f'  Total users : {user_offset:,}')
    print(f'  Total items : {item_offset:,}')
    print(f'  Train rows  : {len(train_df):,}')
    print(f'  Test rows   : {len(test_df):,}')

    return train_df, test_df, user_offset, item_offset, domain_meta


class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users   = torch.tensor(df['user_idx'].values,   dtype=torch.long)
        self.items   = torch.tensor(df['item_idx'].values,   dtype=torch.long)
        self.domains = torch.tensor(df['domain_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values,     dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.domains[idx], self.ratings[idx]

# ── Model ─────────────────────────────────────────────────────────────────────

class MultiDomainMF(nn.Module):
    """
    Matrix Factorization with domain-aware user representations.

    For each interaction (user, item, domain):
        user_ctx  = user_emb + domain_emb   ← domain shifts user taste
        score     = dot(user_ctx, item_emb) + user_bias + item_bias + global_bias
    """
    def __init__(self, n_users, n_items, n_domains, n_factors):
        super().__init__()
        self.user_emb    = nn.Embedding(n_users,   n_factors)
        self.item_emb    = nn.Embedding(n_items,   n_factors)
        self.domain_emb  = nn.Embedding(n_domains, n_factors)

        self.user_bias   = nn.Embedding(n_users, 1)
        self.item_bias   = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        for emb in [self.user_emb, self.item_emb, self.domain_emb]:
            nn.init.normal_(emb.weight, mean=0, std=0.01)
        for bias in [self.user_bias, self.item_bias]:
            nn.init.zeros_(bias.weight)

    def forward(self, user, item, domain):
        u = self.user_emb(user)
        i = self.item_emb(item)
        d = self.domain_emb(domain)

        user_ctx = u + d
        dot      = (user_ctx * i).sum(dim=1)
        bias     = (self.user_bias(user).squeeze(1)
                  + self.item_bias(item).squeeze(1)
                  + self.global_bias)
        return dot + bias

# ── Training ──────────────────────────────────────────────────────────────────

def rmse(preds, targets):
    return torch.sqrt(((preds - targets) ** 2).mean()).item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for users, items, domains, ratings in loader:
        users, items, domains, ratings = (
            users.to(device), items.to(device),
            domains.to(device), ratings.to(device)
        )
        optimizer.zero_grad()
        preds = model(users, items, domains)
        loss  = criterion(preds, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(ratings)
    return (total_loss / len(loader.dataset)) ** 0.5


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    for users, items, domains, ratings in loader:
        users, items, domains = users.to(device), items.to(device), domains.to(device)
        preds = model(users, items, domains).cpu()
        all_preds.append(preds)
        all_targets.append(ratings)
    return rmse(torch.cat(all_preds), torch.cat(all_targets))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU:    {torch.cuda.get_device_name(0)}')

    os.makedirs(args.out, exist_ok=True)

    # Data
    train_df, test_df, n_users, n_items, domain_meta = load_and_split_all(
        args.data, args.domain
    )

    train_loader = DataLoader(
        RatingsDataset(train_df), batch_size=args.batch,
        shuffle=True, num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0)
    )
    test_loader = DataLoader(
        RatingsDataset(test_df), batch_size=args.batch * 2,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0)
    )

    # Model
    model = MultiDomainMF(n_users, n_items, args.n_domains, args.n_factors).to(device)
    print(f'\nModel params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.MSELoss()

    # Train
    print(f'\nTraining {args.epochs} epochs on {len(args.data)} domain(s) ...\n')
    best_rmse = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_rmse = train_epoch(model, train_loader, optimizer, criterion, device)
        test_rmse  = eval_epoch(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f'Epoch {epoch:>3}/{args.epochs}  '
              f'train RMSE: {train_rmse:.4f}  '
              f'test RMSE: {test_rmse:.4f}  '
              f'({elapsed:.1f}s)')

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            ckpt_path = os.path.join(args.out, 'mf_best.pt')
            torch.save({
                'epoch':       epoch,
                'model':       model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'test_rmse':   best_rmse,
                'n_users':     n_users,
                'n_items':     n_items,
                'n_factors':   args.n_factors,
                'n_domains':   args.n_domains,
                'domain_meta': domain_meta,
            }, ckpt_path)
            print(f'  ✓ Best model saved (RMSE={best_rmse:.4f})')

    print(f'\nDone. Best test RMSE: {best_rmse:.4f}')
    print(f'Checkpoint: {args.out}mf_best.pt')


if __name__ == '__main__':
    main()
