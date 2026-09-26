"""Standalone tabular GRACE--Wachter calibration and evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.optimize import linear_sum_assignment
from core import MixedDomain, prototypes, solve_wachter, grace, score, future_bank, exact_w2

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_raw_frame(dataset_config: dict, data_home: str) -> tuple[pd.DataFrame, pd.Series]:
    source = dataset_config.get("source", "openml")
    target_column = dataset_config.get("target_column")
    if source == "openml":
        openml_kwargs = {
            "as_frame": True,
            "data_home": data_home,
        }
        if dataset_config.get("openml_id") is not None:
            openml_kwargs["data_id"] = int(dataset_config["openml_id"])
        else:
            openml_kwargs["name"] = dataset_config.get("openml_name", "heloc")
            if dataset_config.get("openml_version") is not None:
                openml_kwargs["version"] = int(dataset_config["openml_version"])
        dataset = fetch_openml(**openml_kwargs)
        frame = dataset.data.copy()
        labels = dataset.target.copy()
        if target_column and target_column in frame.columns:
            labels = frame[target_column].copy()
            frame = frame.drop(columns=[target_column])
        return frame, pd.Series(labels)
    if source == "csv":
        csv_path = Path(dataset_config["path"]).expanduser()
        frame = pd.read_csv(csv_path)
        if not target_column:
            raise ValueError("CSV datasets require data.dataset.target_column")
        labels = frame[target_column].copy()
        frame = frame.drop(columns=[target_column])
        return frame, pd.Series(labels)
    raise ValueError(f"Unsupported tabular data source: {source}")

def _binarize_labels(labels: pd.Series, positive_label: object | None) -> np.ndarray:
    clean = labels.copy()
    if positive_label is not None:
        return (clean.astype(str) == str(positive_label)).astype(float).to_numpy()
    numeric = pd.to_numeric(clean, errors="coerce")
    if not numeric.isna().any():
        unique = sorted(numeric.dropna().unique().tolist())
        if len(unique) != 2:
            raise ValueError(
                "Tabular experiments require binary labels; set positive_label "
                "for non-binary or non-standard targets."
            )
        return (numeric == unique[-1]).astype(float).to_numpy()
    unique = sorted(clean.astype(str).dropna().unique().tolist())
    if len(unique) != 2:
        raise ValueError(
            "Tabular experiments require binary labels; set positive_label "
            "for non-binary or non-standard targets."
        )
    return (clean.astype(str) == unique[-1]).astype(float).to_numpy()

def _preprocess_tabular(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    dataset_config: dict,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    drop_columns = [
        column for column in dataset_config.get("drop_columns", [])
        if column in train_frame.columns
    ]
    if drop_columns:
        train_frame = train_frame.drop(columns=drop_columns)
        test_frame = test_frame.drop(columns=drop_columns)
    missing_values = dataset_config.get("missing_values", [])
    if missing_values:
        train_frame = train_frame.mask(train_frame.isin(missing_values), np.nan)
        test_frame = test_frame.mask(test_frame.isin(missing_values), np.nan)

    configured_numeric = list(dataset_config.get("numeric_columns", []))
    unknown_numeric = [
        column for column in configured_numeric if column not in train_frame.columns
    ]
    if unknown_numeric:
        raise ValueError(f"Unknown numeric columns: {unknown_numeric}")
    categorical_columns = list(dataset_config.get("categorical_columns", []))
    unknown_categorical = [
        column for column in categorical_columns if column not in train_frame.columns
    ]
    if unknown_categorical:
        raise ValueError(f"Unknown categorical columns: {unknown_categorical}")
    inferred_categorical = train_frame.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    categorical_columns = sorted(
        (set(categorical_columns) | set(inferred_categorical)) - set(configured_numeric)
    )
    numeric_columns = [
        column for column in train_frame.columns if column not in categorical_columns
    ]

    parts_train: list[np.ndarray] = []
    parts_test: list[np.ndarray] = []
    feature_names: list[str] = []
    if numeric_columns:
        train_numeric_frame = train_frame[numeric_columns].replace(
            {r"[\$,]": "", r"%": ""}, regex=True
        )
        test_numeric_frame = test_frame[numeric_columns].replace(
            {r"[\$,]": "", r"%": ""}, regex=True
        )
        train_numeric_frame = train_numeric_frame.apply(pd.to_numeric, errors="coerce")
        test_numeric_frame = test_numeric_frame.apply(pd.to_numeric, errors="coerce")
        numeric_imputer = SimpleImputer(strategy="median")
        train_numeric = numeric_imputer.fit_transform(train_numeric_frame)
        test_numeric = numeric_imputer.transform(test_numeric_frame)
        parts_train.append(train_numeric)
        parts_test.append(test_numeric)
        feature_names.extend(numeric_columns)
    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        train_categorical = categorical_imputer.fit_transform(
            train_frame[categorical_columns]
        )
        test_categorical = categorical_imputer.transform(
            test_frame[categorical_columns]
        )
        train_categorical = train_categorical.astype(str)
        test_categorical = test_categorical.astype(str)
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            max_categories=dataset_config.get("max_categories"),
        )
        train_encoded = encoder.fit_transform(train_categorical)
        test_encoded = encoder.transform(test_categorical)
        parts_train.append(train_encoded)
        parts_test.append(test_encoded)
        feature_names.extend(
            encoder.get_feature_names_out(categorical_columns).tolist()
        )
    if not parts_train:
        raise ValueError("No usable feature columns found for tabular dataset")
    train_values = np.concatenate(parts_train, axis=1)
    test_values = np.concatenate(parts_test, axis=1)
    scaler = StandardScaler()
    train_values = scaler.fit_transform(train_values)
    test_values = scaler.transform(test_values)
    return train_values, test_values, feature_names

def load_tabular_three_way(
    dataset_config: dict,
    data_home: str,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    seed: int = 0,
):
    """Deterministic split with preprocessing fitted on train only."""
    frame, labels = _load_raw_frame(dataset_config, data_home)
    binary = _binarize_labels(labels, dataset_config.get("positive_label"))
    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1-train_fraction:
        raise ValueError("train/calibration fractions must leave a nonempty test split")
    ids = np.arange(len(frame))
    train_ids, remainder = train_test_split(
        ids, train_size=train_fraction, random_state=seed, stratify=binary)
    cal_share = calibration_fraction / (1-train_fraction)
    cal_ids, test_ids = train_test_split(
        remainder, train_size=cal_share, random_state=seed+1,
        stratify=binary[remainder])
    combined = pd.concat((frame.iloc[cal_ids], frame.iloc[test_ids]), axis=0)
    train_values, combined_values, feature_names = _preprocess_tabular(
        frame.iloc[train_ids], combined, dataset_config)
    n_cal=len(cal_ids); cal_values=combined_values[:n_cal]; test_values=combined_values[n_cal:]
    def tensor(x): return torch.tensor(x,dtype=torch.float32)
    return {
        "train_x":tensor(train_values),"train_y":tensor(binary[train_ids]),
        "cal_x":tensor(cal_values),"cal_y":tensor(binary[cal_ids]),
        "test_x":tensor(test_values),"test_y":tensor(binary[test_ids]),
        "train_ids":train_ids,"cal_ids":cal_ids,"test_ids":test_ids,
        "feature_names":feature_names,
    }

class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(previous, 1)

    def forward(
        self, values: torch.Tensor, return_embedding: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.features(values)
        logits = self.classifier(embedding).squeeze(-1)
        if return_embedding:
            return logits, embedding
        return logits

def train_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    hidden_dims: list[int],
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[TabularMLP, float]:
    seed_everything(seed)
    model = TabularMLP(train_x.shape[1], hidden_dims)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        model.train()
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        prediction = (torch.sigmoid(model(test_x)) >= 0.5).float()
        accuracy = float((prediction == test_y).float().mean())
    return model, accuracy

def category_groups(names):
    numeric={"duration","credit_amount","installment_commitment","residence_since","age","existing_credits","num_dependents"}
    roots=[]
    for name in names:
        if name in numeric: continue
        root=name.rsplit("_",1)[0]
        if root not in roots: roots.append(root)
    # Robustly recover roots by matching original German field prefixes.
    fields=("checking_status","credit_history","purpose","savings_status","employment","personal_status","other_parties","property_magnitude","other_payment_plans","housing","job","own_telephone","foreign_worker")
    return {f:[i for i,n in enumerate(names) if n.startswith(f+"_")] for f in fields}

def exact_w2_support(x,y):
    c=torch.cdist(x,y).square().numpy();r,k=linear_sum_assignment(c)
    return float(np.sqrt(c[r,k].mean()))

def bootstrap_budget(cal_x,B,support,seed):
    g=torch.Generator().manual_seed(seed);x=cal_x[torch.randperm(len(cal_x),generator=g)[:support]]
    rng=np.random.default_rng(seed+1); vals=[]
    for _ in range(B):
        idx=torch.as_tensor(rng.integers(0,len(x),len(x)));vals.append(exact_w2_support(x[idx],x))
    return float(np.quantile(vals,.95)),vals

DATASETS = {
    'heloc': {'source': 'openml', 'openml_name': 'heloc', 'openml_version': 1,
              'missing_values': [-7, -8, -9]},
    'diabetes': {'source': 'openml', 'openml_id': 37, 'positive_label': 'tested_negative'},
    'german': {'source': 'openml', 'openml_id': 31, 'positive_label': 'good'},
    'fico': {'source': 'csv', 'target_column': 'RiskPerformance', 'positive_label': 'Good',
             'categorical_columns': ['MaxDelq2PublicRecLast12M', 'MaxDelqEver'],
             'missing_values': [-7, -8, -9]},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--model', choices=['mlp', 'logistic'], default='mlp')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data-home', type=Path, required=True)
    parser.add_argument('--csv', type=Path, help='Local CSV; required for mixed FICO')
    parser.add_argument('--target-column', help='Target column when supplying a local CSV')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths are refused')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    cfg = json.loads(args.config.read_text())
    if cfg['seed'] != 0:
        raise ValueError('This extraction preserves the seed0 protocol and derived seeds.')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(0)
    dataset = dict(DATASETS[args.dataset])
    if args.csv:
        if not args.csv.is_file():
            raise FileNotFoundError(args.csv)
        dataset.update(source='csv', path=str(args.csv.resolve()))
        if args.target_column:
            dataset['target_column'] = args.target_column
    if args.dataset == 'fico' and not args.csv:
        parser.error('FICO requires --csv /path/to/heloc_dataset_v1.csv')
    args.output.mkdir(parents=True)
    out = args.output
    write_json(out/'config.json', cfg)
    d = load_tabular_three_way(dataset, str(args.data_home), cfg['split'][0], cfg['split'][1], 0)
    names = d['feature_names']
    groups = category_groups(names) if args.dataset == 'german' else {
        field: [i for i, name in enumerate(names) if name.startswith(field+'_')]
        for field in dataset.get('categorical_columns', [])}
    domain = MixedDomain(d['train_x'], groups)
    hidden = cfg['training']['hidden'] if args.model == 'mlp' else []
    epochs = cfg['training']['epochs'] if hidden else max(
        cfg['training']['epochs'], math.ceil(1000/math.ceil(len(d['train_x'])/256)))
    model, accuracy = train_model(d['train_x'], d['train_y'], d['cal_x'], d['cal_y'],
                                 hidden, epochs, cfg['training']['lr'], 0)
    torch.save(model.state_dict(), out/'current.pt')
    sources, source_ids = {}, {}
    for split, seed in [('cal', 3), ('test', 2)]:
        with torch.no_grad():
            ids = (model(d[split+'_x']).sigmoid() < .5).nonzero().flatten()
        permutation = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed))
        ids = ids[permutation[:cfg['max_factuals']]]
        if not len(ids):
            raise RuntimeError(f'No unfavorable {split} factuals')
        sources[split] = d[split+'_x'][ids]
        source_ids[split] = d[split+'_ids'][ids.numpy()].tolist()
    with torch.no_grad():
        p1 = d['train_x'][model(d['train_x']).sigmoid() >= .5]
    if not len(p1):
        raise RuntimeError('No favorable training references')
    bref, bootstrap = bootstrap_budget(d['cal_x'], 20, min(128, len(d['cal_x'])), 4)
    torch.save(dict(data=d, sources=sources, source_ids=source_ids, p1=p1,
                    groups=groups, bref=bref, hidden=hidden), out/'prepared.pt')
    write_json(out/'metadata.json', dict(dataset=dataset, classifier=args.model, seed=0,
        calibration_accuracy=accuracy, epochs=epochs, b_ref=bref, bootstrap=bootstrap,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), threads=torch.get_num_threads(),
        command=sys.argv, torch_version=torch.__version__,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), Path(__file__).with_name('core.py')]},
        csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest() if args.csv else None))
    bases = {}
    for split in ['cal', 'test']:
        z, records = solve_wachter(model, sources[split], .5, groups,
                                   prototypes(d['train_x'], groups), 5, 500, 90000)
        with torch.no_grad():
            failed = (model(z).sigmoid() < .5).tolist()
        bases[split] = z
        torch.save(dict(cf=z, records=records, native_failure=failed), out/f'{split}_wachter.pt')
    model = model.to(device)
    x, y = d['train_x'].to(device), d['train_y'].to(device)
    bank, states, records = future_bank(model, x, y, domain, bref, cfg, cfg['future']['cal_seed'])
    torch.save(dict(states=states, records=records), out/'future_cal.pt')
    rows, candidates = [], []
    for beta in cfg['beta_grid']:
        for ratio in cfg['r_ratios']:
            print(f'Calibration: r/b_ref={ratio}, beta={beta}', flush=True)
            result = grace(model, x, y, bases['cal'], p1, domain, ratio*bref, beta, cfg)
            torch.save(result, out/f'cal_grace_b{beta:g}_r{ratio:g}.pt')
            metrics = score(model, result['cf'], bank, sources['cal'])
            rows.append(dict(r_ratio=ratio, r=ratio*bref, beta=beta,
                feasible=result['feasible'], FV=metrics['FV'], proximity=metrics['Cost']))
            if result['feasible']:
                candidates.append((metrics['FV'], -metrics['Cost'], -ratio, -beta, ratio, beta))
    pd.DataFrame(rows).to_csv(out/'calibration.csv', index=False)
    if not candidates:
        write_json(out/'selection.json', {'status': 'no_feasible_calibration'})
        raise RuntimeError('No feasible calibration configuration; retained outputs, no test selection')
    *_, ratio, beta = max(candidates)
    write_json(out/'selection.json', dict(r_ratio=ratio, r=ratio*bref, beta=beta))
    selected = grace(model, x, y, bases['test'], p1, domain, ratio*bref, beta, cfg)
    torch.save(selected, out/'test_grace_wachter.pt')
    write_json(out/'frozen.json', dict(time=time.time(), selection=dict(r_ratio=ratio, beta=beta)))
    bank, states, records = future_bank(model, x, y, domain, bref, cfg, cfg['future']['test_seed'])
    torch.save(dict(states=states, records=records), out/'future_test.pt')
    rows = []
    variants = [('Wachter', 'native', bases['test']),
                ('GRACE-Wachter', 'relaxed' if groups else 'native', selected['cf'])]
    if groups:
        variants.append(('GRACE-Wachter', 'hard', domain.project(selected['cf'], True)))
    for method, realization, z in variants:
        metrics = score(model, z, bank, sources['test'])
        geometry = None
        if method == 'GRACE-Wachter':
            detour = exact_w2(bases['test'], z).distance + exact_w2(z, p1).distance
            geometry = detour <= selected['budget'] + cfg['solver']['feas_tol']*max(1., selected['D'])
        rows.append(dict(method=method, realization=realization, n=len(z), FV=metrics['FV'],
            proximity=metrics['Cost'], current_validity=metrics['current_validity'],
            geometry_feasible=geometry, status='infeasible' if geometry is False else 'evaluated',
            invalid_current_count=int((~metrics['current']).sum())))
        torch.save(metrics, out/f'scores_{method}_{realization}.pt')
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'results.csv', index=False)
    write_json(out/'complete.json', dict(completed=time.time(), rows=len(rows)))
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
