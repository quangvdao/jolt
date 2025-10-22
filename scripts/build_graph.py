#!/usr/bin/env python3
"""
Jolt DAG Graph Builder

Builds a graph of sumchecks and claims from a lightweight YAML spec plus static scans of the codebase.

Outputs:
- DOT (Graphviz) → SVG/PNG for docs
- GraphML → import into yEd/Gephi
- Cytoscape HTML → interactive single-file view

Quick start
  python scripts/build_graph.py --repo /path/to/jolt \
    --out jolt-dag

Using uv (recommended)
  # Full: YAML + GraphML (installs deps ephemerally)
  uv run --python 3.11 --with pyyaml --with networkx -- python3 scripts/build_graph.py \
    --repo /path/to/jolt \
    --spec /path/to/jolt/path/to/graph.yaml \
    --out /path/to/jolt/jolt-dag

  # Minimal: YAML only (no GraphML)
  uv run --python 3.11 --with pyyaml -- python3 scripts/build_graph.py \
    --repo /path/to/jolt \
    --spec /path/to/jolt/path/to/graph.yaml \
    --out /path/to/jolt/jolt-dag

  # Pure static scan (no YAML merge)
  uv run --python 3.11 -- python3 scripts/build_graph.py \
    --repo /path/to/jolt \
    --out /path/to/jolt/jolt-dag

Optional dependencies (if not using uv)
  pip install pyyaml networkx

Spec
- A YAML file defines nodes and edges (human-editable). Pass with --spec.
- Static scan augments it by finding:
  - consumes: get_virtual_polynomial_opening(VirtualPolynomial::<X>, SumcheckId::<Y>)
  - produces: append_virtual(..., VirtualPolynomial::<X>, SumcheckId::<Y>, ...)

Viewing
- Graphviz: dot -Tsvg jolt-dag/jolt_dag.dot > jolt_dag.svg
- Cytoscape: open jolt-dag/jolt_dag.html in a browser
- yEd/Gephi: import jolt-dag/jolt_dag.graphml

Runtime logging (optional)
- Build jolt-core with feature `graphlog` and set env var JOLT_GRAPH_LOG to a file path.
- The logger records produces/consumes of VirtualPolynomial with a SumcheckId.
- This script can be extended to merge JSONL logs into the graph if desired.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # pyright: ignore[reportAssignmentType]

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # pyright: ignore[reportAssignmentType]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_spec(path: Optional[Path]) -> Dict:
    if path is None:
        return {"nodes": [], "edges": []}
    if not path.exists():
        print(f"[warn] spec not found: {path}")
        return {"nodes": [], "edges": []}
    if path.suffix in (".json",):
        return json.loads(read_text(path))
    if yaml is None:
        print(f"[warn] PyYAML not installed; ignoring spec {path}")
        return {"nodes": [], "edges": []}
    return yaml.safe_load(read_text(path)) or {"nodes": [], "edges": []}


def parse_sumcheck_ids(repo: Path) -> List[str]:
    # Parse enum SumcheckId variants from opening_proof.rs
    target = repo / "jolt-core/src/poly/opening_proof.rs"
    if not target.exists():
        return []
    text = read_text(target)
    m = re.search(r"enum\s+SumcheckId\s*\{([^}]+)\}", text)
    if not m:
        return []
    body = m.group(1)
    ids: List[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        name = line.split(",")[0].strip()
        if name:
            ids.append(name)
    return ids


def scan_code_for_edges(repo: Path) -> Tuple[List[Dict], List[Dict]]:
    """Return (nodes, edges) discovered by static scan.
    Nodes returned here are only of kind 'sumcheck' (from usage) and 'claim'.
    """
    consumes_edges: List[Dict] = []  # claim -> sumcheck
    produces_edges: List[Dict] = []  # sumcheck -> claim
    sumchecks_seen = set()
    claims_seen = set()

    get_opening_re = re.compile(
        r"get_virtual_polynomial_opening\s*\(\s*VirtualPolynomial::(\w+)\s*,\s*SumcheckId::(\w+)\s*\)"
    )
    append_virtual_re = re.compile(
        r"append_virtual\s*\(.*?VirtualPolynomial::(\w+).*?SumcheckId::(\w+).*?\)",
        re.DOTALL,
    )

    for path in (repo / "jolt-core/src").rglob("*.rs"):
        text = read_text(path)
        for m in get_opening_re.finditer(text):
            vp, sc = m.group(1), m.group(2)
            claims_seen.add(f"{vp}@{sc}")
            sumchecks_seen.add(sc)
            consumes_edges.append({"from": f"{vp}@{sc}", "to": sc, "type": "consumes"})
        for m in append_virtual_re.finditer(text):
            vp, sc = m.group(1), m.group(2)
            claims_seen.add(f"{vp}@{sc}")
            sumchecks_seen.add(sc)
            produces_edges.append({"from": sc, "to": f"{vp}@{sc}", "type": "produces"})

    nodes: List[Dict] = (
        [{"id": sc, "kind": "sumcheck"} for sc in sorted(sumchecks_seen)]
        + [{"id": c, "kind": "claim"} for c in sorted(claims_seen)]
    )
    edges: List[Dict] = consumes_edges + produces_edges
    return nodes, edges


def merge_graph(spec_nodes: List[Dict], spec_edges: List[Dict], scan_nodes: List[Dict], scan_edges: List[Dict]):
    def key(n: Dict) -> str:
        return n.get("id")

    nodes_map: Dict[str, Dict] = {key(n): n for n in spec_nodes}
    for n in scan_nodes:
        nid = key(n)
        if nid not in nodes_map:
            nodes_map[nid] = n

    # merge edges by tuple key
    def ekey(e: Dict) -> Tuple[str, str, str]:
        return (e.get("from", ""), e.get("to", ""), e.get("type", ""))

    edge_set = {ekey(e) for e in spec_edges}
    merged_edges = list(spec_edges)
    for e in scan_edges:
        if ekey(e) not in edge_set:
            merged_edges.append(e)
            edge_set.add(ekey(e))

    return list(nodes_map.values()), merged_edges


def write_dot(nodes: List[Dict], edges: List[Dict], out_path: Path):
    lines = ["digraph JoltDAG {"]
    lines.append("  rankdir=LR;")
    for n in nodes:
        nid = n["id"].replace('"', "\"")
        kind = n.get("kind", "")
        shape = "box" if kind == "sumcheck" else "ellipse"
        lines.append(f"  \"{nid}\" [shape={shape}, style=filled, fillcolor={'#dff' if kind=='sumcheck' else '#ffd'}];")
    for e in edges:
        src = e["from"].replace('"', "\"")
        dst = e["to"].replace('"', "\"")
        et = e.get("type", "")
        color = "#55f" if et == "consumes" else ("#5a5" if et == "produces" else "#888")
        label = et
        lines.append(f"  \"{src}\" -> \"{dst}\" [color=\"{color}\", label=\"{label}\"];")
    lines.append("}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_graphml(nodes: List[Dict], edges: List[Dict], out_path: Path):
    if nx is None:
        print("[warn] networkx not installed; skipping GraphML")
        return
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in edges:
        G.add_edge(e["from"], e["to"], **{k: v for k, v in e.items() if k not in ("from", "to")})
    nx.write_graphml(G, out_path)


def write_cytoscape_html(nodes: List[Dict], edges: List[Dict], out_path: Path):
    data = {
        "elements": {
            "nodes": [{"data": {"id": n["id"], **{k: v for k, v in n.items() if k != "id"}}} for n in nodes],
            "edges": [
                {"data": {"id": f"{e['from']}->{e['to']}", "source": e["from"], "target": e["to"], **{k: v for k, v in e.items() if k not in ("from", "to")}}}
                for e in edges
            ],
        }
    }
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Jolt DAG</title>
  <script src=\"https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js\"></script>
  <style>
    html, body, #cy {{ width: 100%; height: 100%; margin: 0; padding: 0; }}
  </style>
  </head>
  <body>
    <div id=\"cy\"></div>
    <script>
      const graphData = {json.dumps(data)};
      const cy = cytoscape({{
        container: document.getElementById('cy'),
        elements: graphData.elements,
        style: [
          {{ selector: 'node[kind = \"sumcheck\"]', style: {{ 'shape': 'round-rectangle', 'background-color': '#cde', 'label': 'data(id)' }} }},
          {{ selector: 'node[kind = \"claim\"]', style: {{ 'shape': 'ellipse', 'background-color': '#fec', 'label': 'data(id)' }} }},
          {{ selector: 'edge[type = \"consumes\"]', style: {{ 'line-color': '#55f', 'target-arrow-color': '#55f', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'label': 'data(type)' }} }},
          {{ selector: 'edge[type = \"produces\"]', style: {{ 'line-color': '#5a5', 'target-arrow-color': '#5a5', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier', 'label': 'data(type)' }} }},
        ],
        layout: {{ name: 'breadthfirst', directed: true, spacingFactor: 1.2 }}
      }});
    </script>
  </body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True, help="Absolute path to repository root")
    parser.add_argument("--spec", type=Path, default=None, help="YAML/JSON spec path")
    parser.add_argument("--out", type=Path, required=False, default=None, help="Output directory (default: <repo>/jolt-dag)")
    args = parser.parse_args()

    repo = args.repo.resolve()
    out_dir = (args.out if args.out else (repo / "jolt-dag")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_spec(args.spec.resolve() if args.spec else None)
    spec_nodes: List[Dict] = list(spec.get("nodes", []))
    spec_edges: List[Dict] = list(spec.get("edges", []))

    scan_nodes, scan_edges = scan_code_for_edges(repo)

    # ensure all SumcheckId variants are present as nodes
    sumchecks = parse_sumcheck_ids(repo)
    for sc in sumchecks:
        if sc not in {n.get("id") for n in spec_nodes} and sc not in {n.get("id") for n in scan_nodes}:
            scan_nodes.append({"id": sc, "kind": "sumcheck"})

    nodes, edges = merge_graph(spec_nodes, spec_edges, scan_nodes, scan_edges)

    write_dot(nodes, edges, out_dir / "jolt_dag.dot")
    write_graphml(nodes, edges, out_dir / "jolt_dag.graphml")
    write_cytoscape_html(nodes, edges, out_dir / "jolt_dag.html")

    print(f"Wrote: {out_dir / 'jolt_dag.dot'}")
    print(f"Wrote: {out_dir / 'jolt_dag.html'}")
    if nx is not None:
        print(f"Wrote: {out_dir / 'jolt_dag.graphml'}")


if __name__ == "__main__":
    main()


