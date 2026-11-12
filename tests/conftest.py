import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_functions(script, names):
    """Import only the named top-level functions from a script without running its pipeline."""
    tree = ast.parse((ROOT / "src" / script).read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n.name in names)]
    keep = [n for n in keep if not (isinstance(n, ast.ImportFrom) and n.module == "features")]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), script, "exec"), ns)
    return [ns[n] for n in names]


@pytest.fixture(scope="session")
def results():
    pd_model = json.loads((ROOT / "results/02_pd_model.json").read_text())
    cox = json.loads((ROOT / "results/03_cox.json").read_text())
    return pd_model, cox
