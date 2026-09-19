"""Read-only validation of committed branch payload hashes and Git modes."""
from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent

def git(*args):
    return subprocess.check_output(["git","-C",str(R),*args])

catalog=json.loads(git("show","main:VARIANTS.json"))
checked=0
for row in catalog["variants"]:
    ref=row["branch"]
    manifest=json.loads(git("show",ref+":SNAPSHOT.json"))
    entries={line.split(b"\t",1)[1].decode():line.split(b" ",1)[0].decode()
             for line in git("ls-tree","-r",ref).splitlines()}
    files=manifest["files"]
    assert len(files)==row["file_count"]
    for path,meta in files.items():
        assert hashlib.sha256(git("show",ref+":"+path)).hexdigest()==meta["sha256"],(ref,path)
        assert entries[path]==meta["mode"],(ref,path)
        checked+=1
print(json.dumps({"branches_verified":len(catalog["variants"]),"file_entries_verified":checked}))
