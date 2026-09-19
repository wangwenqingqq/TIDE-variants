#!/usr/bin/env python3
"""Export native HDF5 rows for a bytewise independent local round-trip audit."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import numpy as np
import tables as tb

root, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=False)
manifest = {}
for path in sorted(root.glob('*.h5')):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with tb.open_file(str(path), mode='r') as handle:
        rows = handle.root.fps.read()
        bins = handle.root.config[4]
    target = output / (path.stem + '.npy')
    np.save(target, rows, allow_pickle=False)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    manifest[path.stem] = dict(h5_sha256=digest, rows=len(rows), bins=bins,
                              npy_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
(output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': 'EXPORTED_READONLY', 'files': len(manifest),
                  'rows': sum(x['rows'] for x in manifest.values())}, indent=2))
