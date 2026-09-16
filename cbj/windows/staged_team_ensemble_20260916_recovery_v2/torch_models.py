"""Torch architectures for AE64, class-specific FM and token MLP candidates."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Autoencoder64(nn.Module):
    def __init__(self, input_width: int):
        super().__init__()
        self.input_width = int(input_width)
        self.encoder1 = nn.Linear(self.input_width, 256)
        self.encoder2 = nn.Linear(256, 64)
        self.decoder1 = nn.Linear(64, 256)
        self.decoder2 = nn.Linear(256, self.input_width)

    def encode(self, x):
        return torch.relu(self.encoder2(torch.relu(self.encoder1(x))))

    def forward(self, x):
        return self.decoder2(torch.relu(self.decoder1(self.encode(x))))

    def architecture(self):
        return [self.input_width, 256, 64, 256, self.input_width]


class ClassSpecificFM(nn.Module):
    def __init__(self, symbolic_width: int, missing_width: int, numeric_width: int, classes: int, rank: int):
        super().__init__()
        self.symbolic_width = int(symbolic_width)
        self.missing_width = int(missing_width)
        self.numeric_width = int(numeric_width)
        self.classes = int(classes)
        self.rank = int(rank)
        self.linear = nn.Linear(self.symbolic_width + self.missing_width + self.numeric_width, self.classes)
        self.factors = nn.Parameter(torch.empty(self.classes, self.symbolic_width, self.rank))
        nn.init.normal_(self.factors, mean=0.0, std=0.01)

    def forward(self, symbolic, missing, numeric):
        linear_logits = self.linear(torch.cat([symbolic, missing, numeric], dim=1))
        projected = torch.einsum("bf,kfr->bkr", symbolic, self.factors)
        squared_projected = projected.square()
        correction = torch.einsum("bf,kfr->bkr", symbolic.square(), self.factors.square())
        interaction = 0.5 * (squared_projected - correction).sum(dim=2)
        return linear_logits + interaction


class TokenMLP(nn.Module):
    def __init__(self, vocab_sizes, embedding_dim, pool, dropout, numeric_width, classes):
        super().__init__()
        if pool not in {"mean_max", "mean_sqrt_normalized_sum"}:
            raise ValueError("unsupported token pooling")
        self.pool = pool
        self.embedding_dim = int(embedding_dim)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(int(size) + 1, self.embedding_dim, padding_idx=0)
            for name, size in vocab_sizes.items()
        })
        self.position_projection = nn.Linear(2, self.embedding_dim, bias=False)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(2 * self.embedding_dim + int(numeric_width), int(classes))

    def _pool(self, token_state, mask):
        mask_float = mask.unsqueeze(2).to(token_state.dtype)
        token_state = token_state * mask_float
        count = mask_float.sum(dim=1)
        mean = token_state.sum(dim=1) / count.clamp_min(1.0)
        if self.pool == "mean_max":
            masked = token_state.masked_fill(~mask.unsqueeze(2), float("-inf"))
            maximum = masked.max(dim=1).values
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            return torch.cat([mean, maximum], dim=1)
        sqrt_sum = token_state.sum(dim=1) / count.clamp_min(1.0).sqrt()
        return torch.cat([mean, sqrt_sum], dim=1)

    def forward(self, categorical, position, mask, numeric, return_pooled=False):
        state = self.position_projection(position)
        for name, embedding in self.embeddings.items():
            state = state + embedding(categorical[name])
        state = self.token_mlp(state)
        pooled = self._pool(state, mask)
        logits = self.classifier(torch.cat([pooled, numeric], dim=1))
        if return_pooled:
            return logits, pooled
        return logits


def _epoch_batches(rows, batch_rows, generator):
    order = torch.randperm(rows, generator=generator)
    for start in range(0, rows, batch_rows):
        yield order[start:start + batch_rows]


def train_autoencoder(X, *, device, epochs, batch_rows, learning_rate, weight_decay, seed):
    torch.manual_seed(int(seed))
    model = Autoencoder64(X.shape[1]).to(device)
    data = torch.as_tensor(X, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    generator = torch.Generator().manual_seed(int(seed))
    model.train()
    for _ in range(int(epochs)):
        for indices in _epoch_batches(len(data), int(batch_rows), generator):
            batch = data[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise RuntimeError("autoencoder loss is non-finite")
            loss.backward()
            optimizer.step()
    return model


def train_fm(payload, y, weights, *, classes, rank, device, epochs, batch_rows, learning_rate, weight_decay, seed):
    torch.manual_seed(int(seed))
    arrays = {name: torch.as_tensor(payload[name], dtype=torch.float32) for name in ("symbolic", "missing", "numeric41")}
    labels = torch.as_tensor(y, dtype=torch.long)
    sample_weights = torch.as_tensor(weights, dtype=torch.float32)
    model = ClassSpecificFM(
        arrays["symbolic"].shape[1], arrays["missing"].shape[1], arrays["numeric41"].shape[1],
        int(classes), int(rank),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    generator = torch.Generator().manual_seed(int(seed))
    model.train()
    for _ in range(int(epochs)):
        for indices in _epoch_batches(len(labels), int(batch_rows), generator):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(arrays[name][indices].to(device) for name in ("symbolic", "missing", "numeric41")))
            losses = F.cross_entropy(logits, labels[indices].to(device), reduction="none")
            loss = (losses * sample_weights[indices].to(device)).sum() / sample_weights[indices].sum()
            if not torch.isfinite(loss):
                raise RuntimeError("FM loss is non-finite")
            loss.backward()
            optimizer.step()
    return model


def pack_token_rows(rows):
    max_tokens = max(1, max((len(row) for row in rows), default=0))
    batch = len(rows)
    result = {
        name: torch.zeros((batch, max_tokens), dtype=torch.long)
        for name in ("gene", "kind", "aa_from", "aa_to")
    }
    result["position"] = torch.zeros((batch, max_tokens, 2), dtype=torch.float32)
    result["mask"] = torch.zeros((batch, max_tokens), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        for token_index, token in enumerate(row):
            for name in ("gene", "kind", "aa_from", "aa_to"):
                result[name][row_index, token_index] = int(token[name])
            result["position"][row_index, token_index, 0] = float(token["position"])
            result["position"][row_index, token_index, 1] = float(token["unknown_position"])
            result["mask"][row_index, token_index] = True
    return result


def train_token_mlp(rows, numeric, y, weights, *, vocab_sizes, embedding_dim, pool, dropout,
                    numeric_width, classes, device, epochs, batch_rows, learning_rate, weight_decay, seed):
    torch.manual_seed(int(seed))
    packed = pack_token_rows(rows)
    numeric_tensor = torch.as_tensor(numeric, dtype=torch.float32)
    labels = torch.as_tensor(y, dtype=torch.long)
    sample_weights = torch.as_tensor(weights, dtype=torch.float32)
    model = TokenMLP(vocab_sizes, embedding_dim, pool, dropout, numeric_width, classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    generator = torch.Generator().manual_seed(int(seed))
    model.train()
    for _ in range(int(epochs)):
        for indices in _epoch_batches(len(labels), int(batch_rows), generator):
            categorical = {name: packed[name][indices].to(device) for name in ("gene", "kind", "aa_from", "aa_to")}
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                categorical, packed["position"][indices].to(device), packed["mask"][indices].to(device),
                numeric_tensor[indices].to(device),
            )
            losses = F.cross_entropy(logits, labels[indices].to(device), reduction="none")
            loss = (losses * sample_weights[indices].to(device)).sum() / sample_weights[indices].sum()
            if not torch.isfinite(loss):
                raise RuntimeError("token MLP loss is non-finite")
            loss.backward()
            optimizer.step()
    return model
