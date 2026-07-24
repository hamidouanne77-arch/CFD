#!/usr/bin/env python3
"""Convert formatted 2-D multi-grid PLOT3D NASA hump grids to SU2.

The TMR files contain one block, with i varying fastest. The first j-line is
the NASA hump wall; the last j-line is the contoured inviscid upper boundary.
Coordinates may be scaled to the physical hump chord (0.4200 m).
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import numpy as np


def read_tokens(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="ascii", errors="strict") as f:
            return f.read().split()
    return path.read_text(encoding="ascii").split()


def read_plot3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    tok = read_tokens(path)
    pos = 0
    nblk = int(tok[pos]); pos += 1
    if nblk != 1:
        raise ValueError(f"Expected one block, got {nblk}")
    ni = int(tok[pos]); nj = int(tok[pos + 1]); pos += 2
    n = ni * nj
    if len(tok) < pos + 2*n:
        raise ValueError("Truncated PLOT3D file")
    x = np.asarray(tok[pos:pos+n], dtype=float).reshape((nj, ni)); pos += n
    y = np.asarray(tok[pos:pos+n], dtype=float).reshape((nj, ni)); pos += n
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite coordinate")
    return x, y


def node_id(i: int, j: int, ni: int) -> int:
    return j*ni + i


def write_su2(x: np.ndarray, y: np.ndarray, out: Path) -> None:
    nj, ni = x.shape
    lines: list[str] = ["NDIME= 2", f"NELEM= {(ni-1)*(nj-1)}"]
    eid = 0
    for j in range(nj - 1):
        for i in range(ni - 1):
            n0 = node_id(i, j, ni)
            n1 = node_id(i + 1, j, ni)
            n2 = node_id(i + 1, j + 1, ni)
            n3 = node_id(i, j + 1, ni)
            lines.append(f"9 {n0} {n1} {n2} {n3} {eid}")
            eid += 1

    lines.append(f"NPOIN= {ni*nj}")
    for j in range(nj):
        for i in range(ni):
            nid = node_id(i, j, ni)
            lines.append(f"{x[j,i]:.17e} {y[j,i]:.17e} {nid}")

    markers: dict[str, list[tuple[int, int]]] = {
        "wall": [(node_id(i+1, 0, ni), node_id(i, 0, ni)) for i in range(ni-1)],
        "upper": [(node_id(i, nj-1, ni), node_id(i+1, nj-1, ni)) for i in range(ni-1)],
        "inlet": [(node_id(0, j, ni), node_id(0, j+1, ni)) for j in range(nj-1)],
        "outlet": [(node_id(ni-1, j+1, ni), node_id(ni-1, j, ni)) for j in range(nj-1)],
    }
    lines.append(f"NMARK= {len(markers)}")
    for name, edges in markers.items():
        lines.append(f"MARKER_TAG= {name}")
        lines.append(f"MARKER_ELEMS= {len(edges)}")
        lines.extend(f"3 {a} {b}" for a, b in edges)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.scale) or args.scale <= 0.0:
        raise ValueError("--scale must be positive and finite")
    x, y = read_plot3d(args.input)
    x = x * args.scale
    y = y * args.scale
    write_su2(x, y, args.output)
    print(f"converted {x.shape[1]}x{x.shape[0]} at scale={args.scale} -> {args.output}")


if __name__ == "__main__":
    main()
