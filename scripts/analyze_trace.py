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
from pathlib import Path


def parse_trace(trace_path: str) -> dict:
    """Parse a Chrome trace JSON file and return timing statistics."""
    
    with open(trace_path, 'r') as f:
        events = json.load(f)
    
    # Handle both array format and object format with traceEvents key
    if isinstance(events, dict) and 'traceEvents' in events:
        events = events['traceEvents']
    
    # Match B/E pairs by thread and name
    stacks = defaultdict(list)  # (pid, tid) -> stack of (name, start_ts)
    durations = defaultdict(list)  # name -> [duration_ms, ...]
    
    for e in events:
        if not isinstance(e, dict):
            continue
        
        ph = e.get('ph')
        name = e.get('name', '')
        tid = (e.get('pid', 0), e.get('tid', 0))
        ts = e.get('ts', 0)  # microseconds
        
        if ph == 'B':
            stacks[tid].append((name, ts))
        elif ph == 'E':
            if stacks[tid] and stacks[tid][-1][0] == name:
                start_name, start_ts = stacks[tid].pop()
                dur_ms = (ts - start_ts) / 1000
                durations[name].append(dur_ms)
    
    return durations


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


def analyze_sumchecks(durations: dict) -> dict:
    """Extract sumcheck-related timings from duration data."""
    
    # Aggregate by sumcheck type
    sumcheck_times = defaultdict(float)
    sumcheck_counts = defaultdict(int)
    sumcheck_spans = defaultdict(list)  # category -> [(span_name, total_ms, count)]
    
    for name, times in durations.items():
        category = categorize_span(name)
        if category:
            total_ms = sum(times)
            count = len(times)
            sumcheck_times[category] += total_ms
            sumcheck_counts[category] += count
            sumcheck_spans[category].append((name, total_ms, count))
    
    return {
        'sumcheck_times': dict(sumcheck_times),
        'sumcheck_counts': dict(sumcheck_counts),
        'sumcheck_spans': dict(sumcheck_spans),
    }


def get_total_proving_time(durations: dict) -> float:
    """Get total proving time from the 'prove' span."""
    if 'prove' in durations:
        return sum(durations['prove'])
    return 0.0


def print_report(trace_path: str, durations: dict, analysis: dict):
    """Print a formatted timing report."""
    
    total_prove_ms = get_total_proving_time(durations)
    
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
            durations = parse_trace(trace_path)
            analysis = analyze_sumchecks(durations)
            print_report(trace_path, durations, analysis)
        except FileNotFoundError:
            print(f"Error: File not found: {trace_path}")
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {trace_path}: {e}")
        except Exception as e:
            print(f"Error processing {trace_path}: {e}")
            raise


if __name__ == '__main__':
    main()
