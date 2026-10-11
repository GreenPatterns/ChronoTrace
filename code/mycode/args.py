from easydict import EasyDict

import os
def _data_root():
    """Same resolution as chronotrace.core: $CHRONOTRACE_DATA_DIR, else <repo>/data, else ../data/cyberdata."""
    if os.environ.get("CHRONOTRACE_DATA_DIR"):
        return os.environ["CHRONOTRACE_DATA_DIR"]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for candidate in (os.path.join(repo, "data"), os.path.join(os.path.dirname(repo), "data", "cyberdata")):
        if os.path.exists(os.path.join(candidate, "bitcoinotc.csv")):
            return candidate
    return os.path.join(repo, "data")


DATA_ROOT = _data_root()

def parameter_parser(dataset="otc", pred_snap='single', attack=True,
                     atype='bad_mouthing', victim_percentage=0.1,
                     homogeneous_edges=True, model='s3r',
                     epochs=50, lambda_temp=0.01):

    base = {
        "single_prediction": pred_snap == 'single',
        "time_slots": 10,
        "train_time_slots": 2,
        "seed": 42,
        "layers": [32, 64, 32],
        "learning_rate": 0.005,
        "weight_decay": 1e-5,
        "epochs": epochs,
        "dropout": 0.0,
        "attack": attack,
        "attack_type": atype,
        "attack_percentage": 1.0,
        "attack_target_future": True,
        "enable_task3": False,
        "victim_percentage": victim_percentage,
        "homogeneous_edges": homogeneous_edges,
        "model": model,
        "lambda_temp": lambda_temp,
        "adv_flip_ratio": 0.02,
        "robust_aggr": False,
    }

    if dataset.lower() in ["otc", "bitcoinotc"]:
        dataset_args = {
            "dataset": "BitcoinOTC",
            "edge_path": os.path.join(DATA_ROOT, "bitcoinotc-rating.txt"),
            "data_path": os.path.join(DATA_ROOT, "bitcoinotc.csv"),
            "attention_head": 16,
            "attention_dropout": 0.3,
        }
    elif dataset.lower() in ["epinions", "epinion", "epn"]:
        dataset_args = {
            "dataset": "Epinions",
            "edge_path": os.path.join(DATA_ROOT, "epinions-rating.txt"),
            "data_path": os.path.join(DATA_ROOT, "epinions.csv"),
            "num_nodes": 131828,
            "attention_head": 8,
            "attention_dropout": 0.2,
        }
    elif dataset.lower() in ["alpha", "bitcoinalpha"]:
        dataset_args = {
            "dataset": "BitcoinAlpha",
            "edge_path": os.path.join(DATA_ROOT, "bitcoinalpha-rating.txt"),
            "data_path": os.path.join(DATA_ROOT, "bitcoinalpha.csv"),
            "attention_head": 16,
            "attention_dropout": 0.2,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'otc', 'epn', or 'alpha'.")

    base.update(dataset_args)
    return EasyDict(base)
