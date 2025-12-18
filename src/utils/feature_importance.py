import numpy as np
import torch
from typing import List, Optional, Dict, Union


def get_important_features(
    model,
    *,
    which: str = "synergistic",   # "synergistic" | "non_synergistic"
    threshold: float = 0.7,
    feature_names: Optional[List[str]] = None,
    return_dict: bool = False,
    verbose: bool = False,
):
    """
    Extract important features based on SynFS gates.

    Args
    ----
    model : SynFSModel
        Trained SynFS model.
    which : str
        "synergistic" or "non_synergistic"
    threshold : float
        Gate threshold.
    feature_names : list[str], optional
        If provided, map indices -> feature names.
    return_dict : bool
        If True, return detailed dict instead of list.
    verbose : bool
        Print summary.

    Returns
    -------
    indices OR names OR dict
    """

    assert which in {"synergistic", "non_synergistic"}

    fs_model = (
        model.s_model if which == "synergistic"
        else model.ns_model
    )

    # ---- collect gates ----
    gates = model.get_gates(fs_model)  # list of tensors
    gates_np = np.concatenate([
        g.detach().cpu().numpy() for g in gates
    ])

    important_idx = np.where(gates_np > threshold)[0]

    if feature_names is not None:
        assert len(feature_names) == len(gates_np), \
            "feature_names length must match total gate dimension"
        important_names = [feature_names[i] for i in important_idx]
    else:
        important_names = None

    if verbose:
        print(f"[{which}] threshold={threshold}")
        print(f"  total features     : {len(gates_np)}")
        print(f"  selected features  : {len(important_idx)}")

        if important_names is not None:
            for i, n in zip(important_idx, important_names):
                print(f"  [{i:4d}] {n}")
        else:
            print("  indices:", important_idx)

    if return_dict:
        return {
            "which": which,
            "threshold": threshold,
            "indices": important_idx,
            "names": important_names,
            "gates": gates_np[important_idx],
        }

    return important_names if feature_names is not None else important_idx
