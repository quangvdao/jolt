from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent
runs = {}
for path in sorted((root / 'raw').glob('*.txt')):
    bench, version, pair, _ = path.name.split('.')
    values = {}
    bounds = {}
    for line in path.read_text().splitlines():
        if not line or '=' not in line:
            continue
        record = line.split()[0]
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        for qualifier in ('phase', 'layout'):
            if qualifier in fields:
                record += ' ' + qualifier + '=' + fields[qualifier]
        for key, value in fields.items():
            if not (key == 'ns' or key.endswith('_ns')) or key.startswith(('model', 'target', 'threshold', 'spec_threshold')):
                continue
            if key in ('min_ns', 'max_ns') or '_min_ns' in key or '_max_ns' in key:
                continue
            if key in ('gain_over_default_ns', 'combined_spread_ns', 'median_gain_ns', 'sum_spreads_ns'):
                continue
            try:
                value = float(value)
            except ValueError:
                continue
            canonical = key.replace('median_ns', 'ns')
            values.setdefault((record, canonical), []).append(value)
            prefix = '' if key in ('ns', 'median_ns') else canonical[:-2]
            low = fields.get(prefix + 'min_ns')
            high = fields.get(prefix + 'max_ns')
            try:
                bound = (float(low), float(high))
            except (ValueError, TypeError):
                bound = (value, value)
            bounds.setdefault((record, canonical), []).append(bound)
    runs[(bench, version, pair)] = {
        key: (statistics.median(samples), min(x[0] for x in bounds[key]), max(x[1] for x in bounds[key]))
        for key, samples in values.items()
    }

benches = ['tail', 'routers', 'reduction', 'chunk_product', 'column_pass', 'outer_f2', 'fold']
keys = set(key for run in runs.values() for key in run)
lines = [
    '# Loaded-machine benchmark comparison', '',
    'Every figure below is loaded-machine evidence. Each process used native target code, `--log-t 20 --threads 1 --samples 3`.',
    'For each bench, the session ran before, after, before, after, before, after. Before is integration head `a398d5f0a`; after is the completed runner port.',
    'The `before 1` / `after 1` through `before 3` / `after 3` columns retain each invocation’s reported median. Summary columns take the median of those three reported medians; they are not a claim about nine raw samples.',
    'Repeated fold subphase lines are reduced to a median within each invocation. Baseline chunk split/gather reporting alone was changed from minimum to median, outside timed/allocation intervals; all baseline kernel work and other records are unchanged.',
    'Bracketed ranges cover all reported min/max bounds across the three invocations, or the spread of emitted medians where the baseline did not expose bounds. New or retired diagnostic fields are shown as —; raw logs retain all original records, allocation maxima, model values, thresholds and metadata.',
    'Baseline cold-fixture statistics mix cold and cached work; once-only setup records and cold maxima retain their original meanings rather than being treated as steady-state kernel phases.',
    'Timing units: ns/cycle. See `session.txt` for timestamps and load averages and `raw/` for every native process output.', '',
    '| Record / field | Before (loaded) | After (loaded) | Before 1 (loaded) | After 1 (loaded) | Before 2 (loaded) | After 2 (loaded) | Before 3 (loaded) | After 3 (loaded) | Beyond reported ranges |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
]
summary = []
flags = []
for bench in benches:
    for key in sorted(k for k in keys if any(k in runs.get((bench, v, str(i)), {}) for v in ('before', 'after') for i in range(1, 4))):
        columns = []
        aggregate = {}
        for version in ('before', 'after'):
            points = [runs.get((bench, version, str(i)), {}).get(key) for i in range(1, 4)]
            existing = [point for point in points if point]
            if existing:
                aggregate[version] = (statistics.median(p[0] for p in existing), min(p[1] for p in existing), max(p[2] for p in existing))
                m, low, high = aggregate[version]
                columns.append(f'{m:.6f} [{low:.6f}, {high:.6f}]')
            else:
                columns.append('—')
        beyond = ''
        if len(aggregate) == 2:
            b, a = aggregate['before'], aggregate['after']
            if abs(a[0] - b[0]) > (a[2] - a[1]) + (b[2] - b[1]):
                beyond = 'inspect'
                flags.append((bench, key, aggregate))
        cells = []
        for pair in range(1, 4):
            for version in ('before', 'after'):
                point = runs.get((bench, version, str(pair)), {}).get(key)
                cells.append(f'{point[0]:.6f}' if point else '—')
        lines.append('| ' + ' | '.join([f'`{key[0]}` / `{key[1]}`', *columns, *cells, beyond]) + ' |')
        if key[1] == 'total_ns' and key[0] in [
            'tail/local/20/1', 'routers/default/all_rows/20/1', 'reduction/local/20/1',
            'chunk_product/uniform_digits/20/1', 'column_pass/local/20/1',
            'outer_f2/local/20/1', 'fold/default/all_rows/20/1']:
            summary.append((bench, key, aggregate))
(root / 'comparison.md').write_text('\n'.join(lines) + '\n')
(root / 'flags.txt').write_text('\n'.join(map(str, flags)) + '\n')
print('Primary loaded-machine totals:')
for entry in summary:
    print(entry)
print('Flagged fields:', len(flags))
