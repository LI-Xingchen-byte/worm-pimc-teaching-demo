from __future__ import annotations

import ast
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "wormpimc"
FORBIDDEN_IMPORT_ROOTS = {"IPython", "jupyter", "matplotlib"}


def test_core_package_has_no_notebook_or_plotting_imports() -> None:
    violations: list[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", maxsplit=1)[0]
                    if root in FORBIDDEN_IMPORT_ROOTS:
                        violations.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", maxsplit=1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(f"{path.name}: from {node.module}")

    assert violations == []


def test_derivation_formula_anchors_are_present_and_unique() -> None:
    text = (PROJECT_ROOT / "derivations.md").read_text(encoding="utf-8")
    anchors = re.findall(r'<a id="(eq-[^"]+)"></a>', text)
    required = {
        "eq-wrap",
        "eq-centered-displacement",
        "eq-image-resolved-displacement",
        "eq-free-density",
        "eq-z-log-weight",
        "eq-bridge-recursion",
        "eq-metropolis-log-ratio",
        "eq-wiggle-log-ratio",
        "eq-displace-log-ratio",
        "eq-periodic-free-density",
        "eq-periodic-bridge-mixture",
        "eq-worm-extended-measure",
        "eq-worm-sector-coefficient",
        "eq-worm-primitive-action",
        "eq-worm-log-weight",
        "eq-green-residence-normalization",
        "eq-open-log-ratio",
        "eq-close-log-ratio",
        "eq-insert-log-ratio",
        "eq-remove-log-ratio",
        "eq-advance-log-ratio",
        "eq-recede-log-ratio",
        "eq-swap-log-ratio",
        "eq-thermodynamic-energy-estimator",
        "eq-green-histogram-estimator",
        "eq-g1-beta-minus-estimator",
        "eq-blocking-ratio-standard-error",
    }

    assert required <= set(anchors)
    assert len(anchors) == len(set(anchors))
