"""Self-contained SVC-Q0 inputs; never import earlier campaign modules."""
import json
from pathlib import Path
import numpy as np

MASK = (1 << 64) - 1
THRESHOLDS = ((7, 10), (4, 5))
LUT = np.asarray([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def read_fps(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fp, key = line.split('\t')
        value = bytes.fromhex(fp)
        assert len(value) == 32 and int(key) not in result
        result[int(key)] = int.from_bytes(value, 'little')
    return result


def datasets(root):
    root = Path(root)
    base = read_fps(root / 'base_a.fps')
    other = read_fps(root / 'base_b.fps')
    delta = read_fps(root / 'delta.fps')
    assert len(base) == len(other) == 32768 and len(delta) == 4096
    assert not (base.keys() & other.keys() or base.keys() & delta.keys() or other.keys() & delta.keys())
    base.update(other)
    queries = read_fps(root / 'queries.fps')
    for rows in (base, delta):
        queries.update({i: rows[i] for i in [j for j in sorted(rows) if rows[j]][:16]})
    assert len(queries) == 36 and all(queries.values())
    fixture_base = read_fps(root / 'synthetic_targets.fps')
    fixture_delta = {(1 << 40) + 1: 127, (1 << 40) + 2: 127,
                     (1 << 40) + 3: 0, (1 << 40) + 4: (1 << 256) - 1,
                     (1 << 40) + 5: 1}
    fixture_q = read_fps(root / 'synthetic_queries.fps')
    fixture_q.update({(1 << 40) + 101: 1, (1 << 40) + 102: (1 << 256) - 1})
    return {'real': (base, delta, queries), 'fixture': (fixture_base, fixture_delta, fixture_q)}


def packed(key, fp):
    return np.asarray([key & MASK] + [(fp >> (64 * i)) & MASK for i in range(4)]
                      + [bin(fp).count('1')], dtype=np.uint64)


def arrays(rows):
    keys = np.asarray(list(rows), dtype='<i8')
    byte_rows = np.frombuffer(b''.join(fp.to_bytes(32, 'little') for fp in rows.values()),
                             dtype=np.uint8).reshape(-1, 32).copy()
    pc = LUT[byte_rows].sum(axis=1).astype('<u2')
    order = np.lexsort((keys, pc))
    return keys[order], byte_rows[order], pc[order]


def dump_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
