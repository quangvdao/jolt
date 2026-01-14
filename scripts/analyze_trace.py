#!/usr/bin/env python3
"""
Analyze Chrome trace files to extract sumcheck timing breakdown.

Usage:
    python3 scripts/analyze_trace.py benchmark-runs/traces/sha2_chain_scale22_*.json

This script parses Chrome trace JSON files and aggregates time spent in
different sumcheck implementations, providing a breakdown of:
- Total proving time
- Time per sumcheck type
- Percentage contribution of each sumcheck
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Span:
    name: str
    cat: str
    file: Optional[str]
    pid: int
    tid: int
    dur_ms: float


def _iter_events(trace_obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield dict trace events from either [..] or {traceEvents:[..]} format."""
    if isinstance(trace_obj, dict) and 'traceEvents' in trace_obj:
        trace_obj = trace_obj['traceEvents']
    if isinstance(trace_obj, list):
        for e in trace_obj:
            if isinstance(e, dict):
                yield e


def parse_trace(trace_path: str) -> List[Span]:
    """Parse a Chrome trace JSON file and return completed spans (durations)."""
    with open(trace_path, 'r') as f:
        trace_obj = json.load(f)

    # Match B/E pairs by thread and (name, cat) so nested spans with the same
    # name from different crates (e.g. jolt_core::...::prove vs dory_pcs::prove)
    # don't collide.
    stacks: DefaultDict[Tuple[int, int], List[Tuple[Tuple[str, str], float, Optional[str]]]] = defaultdict(list)
    spans: List[Span] = []

    for e in _iter_events(trace_obj):
        ph = e.get('ph')
        name = e.get('name', '')
        cat = e.get('cat', '') or ''
        file = e.get('.file')
        pid = int(e.get('pid', 0) or 0)
        tid = int(e.get('tid', 0) or 0)
        key = (name, cat)
        thread = (pid, tid)

        # Chrome trace timestamps are microseconds.
        ts = float(e.get('ts', 0) or 0)

        if ph == 'B':
            stacks[thread].append((key, ts, file))
        elif ph == 'E':
            if stacks[thread] and stacks[thread][-1][0] == key:
                (_, start_ts, start_file) = stacks[thread].pop()
                dur_ms = (ts - start_ts) / 1000.0
                spans.append(Span(name=name, cat=cat, file=start_file or file, pid=pid, tid=tid, dur_ms=dur_ms))
        elif ph == 'X':
            # Complete event (has explicit duration in microseconds).
            dur_us = e.get('dur')
            if dur_us is None:
                continue
            dur_ms = float(dur_us) / 1000.0
            spans.append(Span(name=name, cat=cat, file=file, pid=pid, tid=tid, dur_ms=dur_ms))

    return spans


def categorize_span(name: str) -> str:
    """Categorize a span name into a sumcheck type."""
    
    # Order matters - more specific patterns first
    patterns = [
        # RA Virtual sumcheck
        ('InstructionRaSumcheckProver', 'RA Virtual'),
        
        # Instruction Read RAF sumcheck
        ('InstructionReadRafSumcheckProver', 'Instruction Read RAF'),
        ('InstructionReadRafProver', 'Instruction Read RAF'),
        
        # Booleanity sumcheck
        ('BooleanitySumcheckProver', 'Booleanity'),
        
        # Registers read/write checking
        ('RegistersReadWriteCheckingProver', 'Registers RW'),
        
        # RAM read/write checking  
        ('RamReadWriteCheckingProver', 'RAM RW'),
        
        # Memory checking
        ('MemoryCheckingProver', 'Memory Checking'),
        
        # Spartan outer sumcheck
        ('SpartanOuterSumcheckProver', 'Spartan Outer'),
        ('OuterStreamingWindow', 'Spartan Outer'),
        ('OuterSharedState', 'Spartan Outer'),
        
        # Spartan inner sumcheck
        ('SpartanInnerSumcheckProver', 'Spartan Inner'),
        
        # Generic polynomial operations (not sumcheck-specific)
        ('MultilinearPolynomial::bind', 'Polynomial Binding'),
        ('ReadWriteMatrixCycleMajor::bind', 'Matrix Binding'),
    ]
    
    for pattern, category in patterns:
        if pattern in name:
            return category
    
    return None


def is_jolt_span(span: Span) -> bool:
    """Heuristic filter: keep only Jolt-internal spans for accounting."""
    if span.cat.startswith("jolt_core"):
        return True
    if span.file and span.file.startswith("jolt-core/"):
        return True
    return False


def analyze_sumchecks(spans: List[Span]) -> dict:
    """Extract sumcheck-related timings from span list."""
    # Aggregate by sumcheck type
    sumcheck_times: DefaultDict[str, float] = defaultdict(float)
    sumcheck_counts: DefaultDict[str, int] = defaultdict(int)
    by_category_and_name: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for s in spans:
        if not is_jolt_span(s):
            continue
        category = categorize_span(s.name)
        if category:
            sumcheck_times[category] += s.dur_ms
            sumcheck_counts[category] += 1
            by_category_and_name[category][s.name].append(s.dur_ms)

    sumcheck_spans: Dict[str, List[Tuple[str, float, int]]] = {}
    for category, name_map in by_category_and_name.items():
        sumcheck_spans[category] = [
            (name, sum(times), len(times)) for name, times in name_map.items()
        ]

    return {
        'sumcheck_times': dict(sumcheck_times),
        'sumcheck_counts': dict(sumcheck_counts),
        'sumcheck_spans': dict(sumcheck_spans),
    }


def get_total_proving_time(spans: List[Span]) -> float:
    """
    Get total proving time from the zkVM prover's 'prove' span only.

    Important: other crates (e.g. dory_pcs) also emit spans named 'prove'.
    Summing all of them by name would double-count nested work and skew percentages.
    """
    total = 0.0
    for s in spans:
        if s.name != "prove":
            continue
        if s.cat == "jolt_core::zkvm::prover" or (s.file == "jolt-core/src/zkvm/prover.rs"):
            total += s.dur_ms
    return total


def _durations_by_name(spans: List[Span]) -> Dict[str, List[float]]:
    d: DefaultDict[str, List[float]] = defaultdict(list)
    for s in spans:
        if not is_jolt_span(s):
            continue
        d[s.name].append(s.dur_ms)
    return dict(d)


def print_report(trace_path: str, spans: List[Span], analysis: dict):
    """Print a formatted timing report."""
    total_prove_ms = get_total_proving_time(spans)
    durations = _durations_by_name(spans)
    
    print(f"\n{'='*70}")
    print(f"Trace Analysis: {Path(trace_path).name}")
    print(f"{'='*70}")
    
    if total_prove_ms > 0:
        print(f"\nTotal Proving Time: {total_prove_ms:.1f} ms ({total_prove_ms/1000:.3f} s)")
    
    # Sumcheck breakdown
    print(f"\n{'='*70}")
    print("Sumcheck / Prover Component Breakdown")
    print(f"{'='*70}")
    
    sumcheck_times = analysis['sumcheck_times']
    sumcheck_counts = analysis['sumcheck_counts']
    
    if sumcheck_times:
        # Sort by time descending
        sorted_sumchecks = sorted(sumcheck_times.items(), key=lambda x: -x[1])
        
        total_sumcheck_ms = sum(sumcheck_times.values())
        
        print(f"\n{'Component':<30} {'Time (ms)':>12} {'Count':>8} {'% of Total':>12} {'% of Prove':>12}")
        print("-" * 78)
        
        for name, time_ms in sorted_sumchecks:
            count = sumcheck_counts.get(name, 0)
            pct_of_sc = (time_ms / total_sumcheck_ms * 100) if total_sumcheck_ms > 0 else 0
            pct_of_prove = (time_ms / total_prove_ms * 100) if total_prove_ms > 0 else 0
            print(f"{name:<30} {time_ms:>12.1f} {count:>8} {pct_of_sc:>11.1f}% {pct_of_prove:>11.1f}%")
        
        print("-" * 78)
        pct_of_prove = (total_sumcheck_ms / total_prove_ms * 100) if total_prove_ms > 0 else 0
        print(f"{'TOTAL':<30} {total_sumcheck_ms:>12.1f} {'':<8} {'100.0':>11}% {pct_of_prove:>11.1f}%")
    else:
        print("No sumcheck spans found.")
    
    # Detailed RA Virtual breakdown
    print(f"\n{'='*70}")
    print("RA Virtual Sumcheck Details")
    print(f"{'='*70}")
    
    ra_spans = [
        'InstructionRaSumcheckProver::initialize',
        'InstructionRaSumcheckProver::compute_message',
        'InstructionRaSumcheckProver::ingest_challenge',
    ]
    
    ra_total = 0.0
    print(f"\n{'Span':<55} {'Total (ms)':>10} {'Count':>7} {'Avg (ms)':>10}")
    print("-" * 85)
    
    for span_name in ra_spans:
        if span_name in durations:
            times = durations[span_name]
            total = sum(times)
            count = len(times)
            avg = total / count if count else 0
            ra_total += total
            print(f"{span_name:<55} {total:>10.1f} {count:>7} {avg:>10.2f}")
    
    print("-" * 85)
    if total_prove_ms > 0:
        pct = ra_total / total_prove_ms * 100
        print(f"{'RA Virtual Total':<55} {ra_total:>10.1f} {'':>7} {pct:>9.1f}% of prove")
    
    # Stage breakdown
    print(f"\n{'='*70}")
    print("Proving Stage Breakdown")
    print(f"{'='*70}")
    
    stage_spans = ['prove_stage1', 'prove_stage2', 'prove_stage3', 'prove_stage4', 
                   'prove_stage5', 'prove_stage6', 'prove_stage7', 'prove_stage8']
    
    print(f"\n{'Stage':<20} {'Time (ms)':>12} {'% of Prove':>12} {'Description':<30}")
    print("-" * 78)
    
    stage_descriptions = {
        'prove_stage1': 'Witness commitment',
        'prove_stage2': 'R1CS commitments', 
        'prove_stage3': 'Instruction lookups',
        'prove_stage4': 'Memory checking',
        'prove_stage5': 'Register checking',
        'prove_stage6': 'Spartan outer sumcheck',
        'prove_stage7': 'Spartan inner sumcheck',
        'prove_stage8': 'Dory batch opening',
    }
    
    for name in stage_spans:
        if name in durations:
            times = durations[name]
            total = sum(times)
            pct = (total / total_prove_ms * 100) if total_prove_ms > 0 else 0
            desc = stage_descriptions.get(name, '')
            print(f"{name:<20} {total:>12.1f} {pct:>11.1f}% {desc:<30}")
    
    # Commitment time
    print(f"\n{'='*70}")
    print("Commitment Time Breakdown")
    print(f"{'='*70}")
    
    commitment_spans = [
        ('generate_and_commit_witness_polynomials', 'Witness polynomials'),
        ('DoryCommitmentScheme::compute_tier1_commitment', 'Dory tier1 (1024 MSMs)'),
        ('DoryCommitmentScheme::compute_tier1_commitment_onehot', 'Dory tier1 onehot'),
        ('DoryCommitmentScheme::compute_tier2_commitment', 'Dory tier2'),
        ('DoryCommitmentScheme::prove', 'Dory prove'),
    ]
    
    print(f"\n{'Operation':<50} {'Time (ms)':>12} {'% of Prove':>12}")
    print("-" * 78)
    
    for span_name, desc in commitment_spans:
        if span_name in durations:
            times = durations[span_name]
            total = sum(times)
            pct = (total / total_prove_ms * 100) if total_prove_ms > 0 else 0
            print(f"{desc:<50} {total:>12.1f} {pct:>11.1f}%")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_trace.py <trace.json> [trace2.json ...]")
        print("\nAnalyzes Chrome trace files to extract sumcheck timing breakdown.")
        sys.exit(1)
    
    for trace_path in sys.argv[1:]:
        try:
            spans = parse_trace(trace_path)
            analysis = analyze_sumchecks(spans)
            print_report(trace_path, spans, analysis)
        except FileNotFoundError:
            print(f"Error: File not found: {trace_path}")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {trace_path}: {e}")
        except Exception as e:
            print(f"Error processing {trace_path}: {e}")
            raise


if __name__ == '__main__':
    main()
