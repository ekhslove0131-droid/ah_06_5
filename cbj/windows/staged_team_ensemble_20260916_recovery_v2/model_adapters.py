"""Pinned estimator recipes, CPU adapters and fail-closed GPU dispatch."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


class GPURequiredError(RuntimeError):
    pass


class ModelAdapterError(RuntimeError):
    pass


@dataclass
class ModelArtifact:
    model: object
    recipe: dict
    class_order: list[str]
    feature_names: list[str]
    train_feature_sha256: str
    device: str


class IndexedProbabilityModel:
    def __init__(self, model, class_order):
        self.model = model
        self.classes_ = np.asarray(class_order)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


def _torch_device():
    import torch
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TorchFMPredictor:
    def __init__(self, model, class_order, batch_rows=512):
        self.model = model.cpu().eval()
        self.classes_ = np.asarray(class_order)
        self.batch_rows = int(batch_rows)

    def predict_proba(self, payload):
        import torch
        device = _torch_device(); self.model.to(device)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(payload["numeric41"]), self.batch_rows):
                stop = start + self.batch_rows
                logits = self.model(*(
                    torch.as_tensor(payload[name][start:stop], dtype=torch.float32, device=device)
                    for name in ("symbolic", "missing", "numeric41")
                ))
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        self.model.cpu()
        return np.row_stack(outputs)


class TorchTokenPredictor:
    def __init__(self, model, class_order, batch_rows=256):
        self.model = model.cpu().eval()
        self.classes_ = np.asarray(class_order)
        self.batch_rows = int(batch_rows)

    def predict_proba(self, payload):
        import torch
        from torch_models import pack_token_rows
        device = _torch_device(); self.model.to(device)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(payload["tokens"]), self.batch_rows):
                stop = start + self.batch_rows
                packed = pack_token_rows(payload["tokens"][start:stop])
                categorical = {name: packed[name].to(device) for name in ("gene", "kind", "aa_from", "aa_to")}
                logits = self.model(
                    categorical, packed["position"].to(device), packed["mask"].to(device),
                    torch.as_tensor(payload["numeric41"][start:stop], dtype=torch.float32, device=device),
                )
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        self.model.cpu()
        return np.row_stack(outputs)


class AEHeadPredictor:
    def __init__(self, autoencoder, head, class_order, batch_rows=512):
        self.autoencoder = autoencoder.cpu().eval()
        self.head = head
        # The sklearn head can emit a different column order from the frozen
        # competition order.  Preserve the head's real order so
        # ``predict_checked`` can align it explicitly.
        self.classes_ = np.asarray(getattr(head, "classes_", class_order))
        self.batch_rows = int(batch_rows)

    def predict_proba(self, payload):
        import torch
        device = _torch_device(); self.autoencoder.to(device)
        latent_parts = []
        with torch.no_grad():
            for start in range(0, len(payload["ae_input"]), self.batch_rows):
                stop = start + self.batch_rows
                latent_parts.append(self.autoencoder.encode(
                    torch.as_tensor(payload["ae_input"][start:stop], dtype=torch.float32, device=device)
                ).cpu().numpy())
        self.autoencoder.cpu()
        latent = np.row_stack(latent_parts)
        matrix = np.column_stack([latent / 8.0, np.asarray(payload["numeric41"], dtype=np.float32)])
        return self.head.predict_proba(matrix)


class OVRLogisticModel:
    def __init__(self, models, class_reweights, symbolic_width, class_order):
        self.models = models
        self.class_reweights = np.asarray(class_reweights, dtype=np.float32)
        self.symbolic_width = int(symbolic_width)
        self.class_order = list(class_order)

    def predict_proba(self, X):
        outputs = []
        for index, model in enumerate(self.models):
            transformed = _apply_class_reweight(X, self.class_reweights[index], self.symbolic_width)
            outputs.append(model.predict_proba(transformed)[:, 1])
        probabilities = np.column_stack(outputs)
        total = probabilities.sum(axis=1, keepdims=True)
        return np.divide(probabilities, total, out=np.full_like(probabilities, 1 / len(outputs)), where=total > 0)


def _logistic_params(C):
    return {
        "C": float(C), "penalty": "l2", "solver": "lbfgs", "max_iter": 3000,
        "tol": 1e-5, "random_state": 42,
    }


def effective_recipe(declaration):
    family = declaration["family"]
    p = declaration["parameters"]
    recipe = {"family": family, "candidate_id": declaration["candidate_id"], "config_hash": declaration["config_hash"]}
    if family == "T1":
        recipe.update(kind="xgboost", device_kind="GPU_REQUIRED", weighting=p["class_weight"], params={
            "n_estimators": 200, "learning_rate": 0.1, "max_depth": 3,
            "random_state": 42, "n_jobs": 4, "tree_method": "hist", "device": "cuda:0",
            "objective": "multi:softprob", "subsample": 1.0, "colsample_bytree": 1.0,
            "reg_lambda": 1.0, "reg_alpha": 0.0, "min_child_weight": 1.0,
        })
    elif family == "T2":
        recipe.update(kind="extratrees", device_kind="CPU_INTENTIONAL", weighting="estimator_balanced", params={
            "n_estimators": 500, "criterion": "entropy", "max_features": p["max_features"],
            "min_samples_leaf": int(p["min_samples_leaf"]), "class_weight": "balanced",
            "bootstrap": False, "random_state": 42, "n_jobs": 4,
        })
    elif family == "T3":
        if p["model"] == "XGB200_depth3_lr0.1":
            return effective_recipe({**declaration, "family": "T1", "parameters": {"class_weight": "sqrt_inverse"}}) | {"family": family}
        recipe.update(kind="extratrees", device_kind="CPU_INTENTIONAL", weighting="sqrt_inverse", params={
            "n_estimators": 500, "criterion": "entropy", "max_features": "sqrt",
            "min_samples_leaf": 2, "class_weight": None, "bootstrap": False,
            "random_state": 42, "n_jobs": 4,
        })
    elif family == "T4":
        transformer = None
        if p["compressor"] == "AE64":
            transformer = {"kind": "AE64", "hidden": [256, 64, 256], "epochs": 80, "batch_rows": 128, "learning_rate": 0.001, "weight_decay": 0.0001}
        if p["head"] == "logreg_C1":
            recipe.update(kind="logistic", device_kind="GPU_REQUIRED" if transformer else "CPU_INTENTIONAL", weighting="sqrt_inverse", params=_logistic_params(1), transformer=transformer)
        else:
            recipe.update(kind="lightgbm", device_kind="GPU_REQUIRED", weighting="sqrt_inverse", params={
                "objective": "multiclass", "n_estimators": 300, "learning_rate": 0.05,
                "num_leaves": 15, "min_child_samples": 30, "reg_lambda": 5.0,
                "subsample": 1.0, "colsample_bytree": 1.0, "random_state": 42,
                "n_jobs": 4, "device_type": "gpu", "gpu_device_id": 0,
            }, transformer=transformer)
    elif family == "T5":
        if p["model"] == "logreg_C1":
            recipe.update(kind="logistic", device_kind="CPU_INTENTIONAL", weighting="sqrt_inverse", params=_logistic_params(1))
        else:
            recipe.update(kind="extratrees", device_kind="CPU_INTENTIONAL", weighting="sqrt_inverse", params={
                "n_estimators": 500, "criterion": "entropy", "max_features": "sqrt",
                "min_samples_leaf": 5, "class_weight": None, "bootstrap": False,
                "random_state": 42, "n_jobs": 4,
            })
    elif family == "T6":
        if p["model"] == "XGB200_depth3_lr0.1":
            return effective_recipe({**declaration, "family": "T1", "parameters": {"class_weight": "sqrt_inverse"}}) | {"family": family}
        recipe.update(kind="extratrees", device_kind="CPU_INTENTIONAL", weighting="sqrt_inverse", params={
            "n_estimators": 500, "criterion": "entropy", "max_features": "sqrt",
            "min_samples_leaf": 5, "class_weight": None, "bootstrap": False,
            "random_state": 42, "n_jobs": 4,
        })
    elif family in {"G1", "G2", "G4"}:
        recipe.update(kind="logistic", device_kind="CPU_INTENTIONAL", weighting="sqrt_inverse", params=_logistic_params(p["C"]))
    elif family == "G3":
        recipe.update(kind="ovr_logistic", device_kind="CPU_INTENTIONAL", weighting="sqrt_inverse", params=_logistic_params(p["C"]))
    elif family == "G5":
        recipe.update(kind="fm_torch", device_kind="GPU_REQUIRED", weighting="sqrt_inverse", params={
            "interaction_rank": int(p["interaction_rank"]), "weight_decay": float(p["weight_decay"]),
            "learning_rate": 0.001, "epochs": 80, "batch_rows": 128, "random_seed": 42,
            "optimizer": "AdamW",
        })
    elif family == "G6":
        recipe.update(kind="token_mlp_torch", device_kind="GPU_REQUIRED", weighting="sqrt_inverse", params={
            "token_embedding_dim": int(p["token_embedding_dim"]), "pool": p["pool"],
            "dropout": float(p["dropout"]), "weight_decay": 0.01, "learning_rate": 0.001,
            "epochs": 60, "batch_rows": 32, "random_seed": 42, "optimizer": "AdamW",
        })
    else:
        raise ModelAdapterError("unsupported model family")
    return recipe


def _sqrt_inverse_weights(y, class_order):
    y = np.asarray(y)
    counts = {label: int(np.sum(y == label)) for label in class_order}
    if any(value == 0 for value in counts.values()):
        raise ModelAdapterError("training partition is missing a class")
    weights = np.asarray([math.sqrt(len(y) / counts[label]) for label in y], dtype=np.float64)
    return weights / weights.mean()


def _combined_weights(recipe, features, y, weights, class_order):
    result = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).copy()
    if result.shape != (len(y),) or not np.isfinite(result).all() or (result <= 0).any():
        raise ModelAdapterError("sample weights are invalid")
    if recipe["weighting"] == "sqrt_inverse":
        result *= _sqrt_inverse_weights(y, class_order)
    multiplier = getattr(features, "sample_weight_multiplier", np.ones(len(y), dtype=np.float64))
    result *= np.asarray(multiplier, dtype=np.float64)
    return result / result.mean()


def _apply_class_reweight(X, ratio, symbolic_width):
    left = X[:, :symbolic_width]
    right = X[:, symbolic_width:]
    if sparse.issparse(left):
        left = left.multiply(ratio)
        return sparse.hstack([left, right], format="csr")
    return np.column_stack([np.asarray(left) * ratio, np.asarray(right)])


def _require_cuda(recipe, capabilities):
    if recipe["device_kind"] != "GPU_REQUIRED":
        return
    if capabilities is not None and capabilities.get("cuda") is not True:
        raise GPURequiredError("CUDA is required; CPU fallback is forbidden")
    if capabilities is None:
        try:
            import torch
            available = torch.cuda.is_available()
        except Exception:
            available = False
        if not available:
            raise GPURequiredError("CUDA is required; CPU fallback is forbidden")


def _fit_lightgbm_gpu(params, X, y, sample_weight):
    from lightgbm import LGBMClassifier
    from lightgbm.basic import LightGBMError
    base = LGBMClassifier(**params)
    try:
        base.fit(X, y, sample_weight=sample_weight)
    except LightGBMError as exc:
        if "GPU Tree Learner" in str(exc):
            raise GPURequiredError(str(exc)) from exc
        raise
    if base.booster_.params.get("device_type") != "gpu":
        raise GPURequiredError("LightGBM did not retain GPU device_type")
    return base


def fit_model(recipe, features, y, weights, class_order, *, device="cuda:0", capabilities=None):
    _require_cuda(recipe, capabilities)
    if recipe["device_kind"] == "GPU_REQUIRED" and device != "cuda:0":
        raise GPURequiredError("CUDA device must be exactly cuda:0")
    y = np.asarray(y)
    if set(y.tolist()) != set(class_order):
        raise ModelAdapterError("training labels differ from class order")
    sample_weight = _combined_weights(recipe, features, y, weights, class_order)
    kind = recipe["kind"]
    artifact_device = "cpu"
    if (recipe.get("transformer") or {}).get("kind") == "AE64":
        import torch
        from torch_models import train_autoencoder
        if not torch.cuda.is_available():
            raise GPURequiredError("CUDA is required for AE64")
        transformer = recipe["transformer"]
        autoencoder = train_autoencoder(
            np.asarray(features.train, dtype=np.float32), device=device,
            epochs=transformer["epochs"], batch_rows=transformer["batch_rows"],
            learning_rate=transformer["learning_rate"], weight_decay=transformer["weight_decay"], seed=42,
        )
        autoencoder.eval()
        with torch.no_grad():
            latent = autoencoder.encode(torch.as_tensor(features.train, dtype=torch.float32, device=device)).cpu().numpy() / 8.0
        if features.deferred_numeric_train is None:
            raise ModelAdapterError("AE64 requires deferred numeric41")
        transformed_train = np.column_stack([latent, features.deferred_numeric_train]).astype(np.float32)
        if kind == "logistic":
            head = LogisticRegression(**recipe["params"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                head.fit(transformed_train, y, sample_weight=sample_weight)
        elif kind == "lightgbm":
            label_index = np.asarray([class_order.index(label) for label in y], dtype=np.int32)
            head_base = _fit_lightgbm_gpu(recipe["params"], transformed_train, label_index, sample_weight)
            head = IndexedProbabilityModel(head_base, class_order)
        else:
            raise ModelAdapterError("AE64 head is unsupported")
        model = AEHeadPredictor(autoencoder, head, class_order, batch_rows=transformer["batch_rows"])
        artifact_device = "cuda:0"
    elif kind == "logistic":
        model = LogisticRegression(**recipe["params"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(features.train, y, sample_weight=sample_weight)
    elif kind == "extratrees":
        model = ExtraTreesClassifier(**recipe["params"])
        model.fit(features.train, y, sample_weight=sample_weight if recipe["weighting"] != "estimator_balanced" else None)
    elif kind == "ovr_logistic":
        if features.class_reweights is None or features.symbolic_width <= 0:
            raise ModelAdapterError("G3 requires class-specific NB ratios")
        models = []
        for class_index, label in enumerate(class_order):
            transformed = _apply_class_reweight(features.train, features.class_reweights[class_index], features.symbolic_width)
            binary_y = (y == label).astype(np.uint8)
            model = LogisticRegression(**recipe["params"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(transformed, binary_y, sample_weight=sample_weight)
            models.append(model)
        model = OVRLogisticModel(models, features.class_reweights, features.symbolic_width, class_order)
    elif kind == "xgboost":
        from xgboost import XGBClassifier
        label_index = np.asarray([class_order.index(label) for label in y], dtype=np.int32)
        base = XGBClassifier(**recipe["params"])
        base.fit(features.train, label_index, sample_weight=sample_weight)
        config = base.get_booster().save_config()
        if '"device":"cuda:0"' not in config.replace(" ", ""):
            raise GPURequiredError("XGBoost did not execute with cuda:0")
        model = IndexedProbabilityModel(base, class_order)
        artifact_device = "cuda:0"
    elif kind == "lightgbm":
        label_index = np.asarray([class_order.index(label) for label in y], dtype=np.int32)
        base = _fit_lightgbm_gpu(recipe["params"], features.train, label_index, sample_weight)
        model = IndexedProbabilityModel(base, class_order)
        artifact_device = "cuda:0"
    elif kind == "fm_torch":
        import torch
        from torch_models import train_fm
        if not torch.cuda.is_available():
            raise GPURequiredError("CUDA is required for FM")
        label_index = np.asarray([class_order.index(label) for label in y], dtype=np.int64)
        p = recipe["params"]
        base = train_fm(
            features.train, label_index, sample_weight, classes=len(class_order), rank=p["interaction_rank"],
            device=device, epochs=p["epochs"], batch_rows=p["batch_rows"],
            learning_rate=p["learning_rate"], weight_decay=p["weight_decay"], seed=p["random_seed"],
        )
        if str(next(base.parameters()).device) != "cuda:0":
            raise GPURequiredError("FM parameters are not on cuda:0 after fit")
        model = TorchFMPredictor(base, class_order, batch_rows=p["batch_rows"])
        artifact_device = "cuda:0"
    elif kind == "token_mlp_torch":
        import torch
        from torch_models import train_token_mlp
        if not torch.cuda.is_available():
            raise GPURequiredError("CUDA is required for token MLP")
        label_index = np.asarray([class_order.index(label) for label in y], dtype=np.int64)
        p = recipe["params"]
        fitted = features.state["fitted"]
        vocab_sizes = {
            "gene": len(fitted["gene_vocabulary"]), "kind": len(fitted["kind_vocabulary"]),
            "aa_from": len(fitted["aa_from_vocabulary"]), "aa_to": len(fitted["aa_to_vocabulary"]),
        }
        base = train_token_mlp(
            features.train["tokens"], features.train["numeric41"], label_index, sample_weight,
            vocab_sizes=vocab_sizes, embedding_dim=p["token_embedding_dim"], pool=p["pool"],
            dropout=p["dropout"], numeric_width=features.train["numeric41"].shape[1],
            classes=len(class_order), device=device, epochs=p["epochs"], batch_rows=p["batch_rows"],
            learning_rate=p["learning_rate"], weight_decay=p["weight_decay"], seed=p["random_seed"],
        )
        if str(next(base.parameters()).device) != "cuda:0":
            raise GPURequiredError("token MLP parameters are not on cuda:0 after fit")
        model = TorchTokenPredictor(base, class_order, batch_rows=p["batch_rows"])
        artifact_device = "cuda:0"
    else:
        raise ModelAdapterError(f"model implementation is not admitted: {kind}")
    feature_names = list(getattr(features, "feature_names", []))
    train_hash = getattr(features, "train_sha256", hashlib.sha256(repr(features.state).encode()).hexdigest())
    return ModelArtifact(model, recipe, list(class_order), feature_names, train_hash, artifact_device)


def predict_checked(artifact, X, class_order):
    if list(class_order) != artifact.class_order:
        raise ModelAdapterError("prediction class order differs from saved artifact")
    raw = np.asarray(artifact.model.predict_proba(X), dtype=np.float64)
    model_classes = getattr(artifact.model, "classes_", None)
    if model_classes is not None:
        indices = []
        classes = model_classes.tolist()
        for label in class_order:
            if label not in classes:
                raise ModelAdapterError("saved model is missing a class")
            indices.append(classes.index(label))
        raw = raw[:, indices]
    if raw.ndim != 2 or raw.shape[1] != len(class_order) or not np.isfinite(raw).all() or (raw < 0).any():
        raise ModelAdapterError("model probabilities are invalid")
    total = raw.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ModelAdapterError("model probability row has zero mass")
    output = raw / total
    if not np.allclose(output.sum(axis=1), 1.0, atol=1e-8):
        raise ModelAdapterError("model probabilities do not normalize")
    return output


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_artifact(artifact, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return {"schema_version": "MODEL_ARTIFACT_RECEIPT_V1", "sha256": _sha(path), "class_order": artifact.class_order}


def load_artifact(path, receipt):
    if receipt.get("schema_version") != "MODEL_ARTIFACT_RECEIPT_V1" or receipt.get("sha256") != _sha(path):
        raise ModelAdapterError("saved model receipt mismatch")
    artifact = joblib.load(path)
    if artifact.class_order != receipt.get("class_order"):
        raise ModelAdapterError("saved model class order mismatch")
    return artifact
