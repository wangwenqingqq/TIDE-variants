#!/usr/bin/env python3
"""Source-only D0 local-ext4 control plane v5-postexpiry-m1: prepare a ledger or issue one claimed auth.

This tool is intentionally separate from selector v7-m1 and runner v12-m1.  It has no plan,
trace, index, FVEC, CUDA, or GPU command.  The normal issuer has no CLI seed input:
it obtains the 256-bit seed only from Linux getrandom(2), writes it only to the fixed
private ext4 authorization leaf only after a successfully persisted nonsecret issuer claim, and never prints it.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import pathlib
import pwd
import stat
import subprocess
import sys
import time
from typing import Any


class ControlPlaneError(RuntimeError):
    """Public failures carry fixed tokens only; never serialize secret-bearing state."""


# Identity and private-ledger namespace are deliberately fixed, never CLI-configurable.
REQUIRED_UID = 1001
REQUIRED_USERNAME = "zhuxiaomao"
PRIVATE_PARENT = "/workspace"
LEDGER_LEAF = ".tide_d0_auth_ledger_v6"
AUTHORIZATION_LEDGER_ROOT = PRIVATE_PARENT + "/" + LEDGER_LEAF
AUTHORIZATION_DIR = "authorizations"
RESERVATIONS_DIR = "reservations"
OUTCOMES_DIR = "outcomes"
LEDGER_CHILDREN = (AUTHORIZATION_DIR, RESERVATIONS_DIR, OUTCOMES_DIR)
# A fixed nonsecret sibling reserves the global fixed-D0 issuance slot before this control plane's
# D0 seed and authorization-ID _kernel_random_exact draws. It is deliberately not a 32-hex authorization filename, so selector v7-m1
# and runner v12-m1 only ever open the separately generated <authorization_id>.json leaf.
ISSUER_CLAIM_LEAF = "d0_v4_m1_postexpiry_issuer_claim.json"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
EXT4_SUPER_MAGIC = 0xEF53
FIXED_TTL_SECONDS = 3600

# This source verifies its fixed Markdown contract before either command.  The source
# itself is frozen by the source-only SHA256SUMS manifest, avoiding a self-hash cycle.
CONTROL_PLANE_ROOT = "/workspace/tide_d0_control_plane_v5_postexpiry_m1_sourceonly_20260809T104443CST"
CONTROL_PLANE_CONTRACT_PATH = CONTROL_PLANE_ROOT + "/metadata/CONTROL_PLANE_CONTRACT.md"
CONTROL_PLANE_CONTRACT_SHA256 = "351a753b5d11ad52cae0cfc7fff76a78ed3417e2cc189ff3ce38c16fe9cc1d5a"

# Frozen selector-v7-m1 artifact paths and raw byte pins.
SELECTOR_V7_M1_ROOT = "/workspace/sift_d0_selection_plan_generator_v7_m1_postexpiry_migration_sourceonly_20260809T101845CST"
SELECTOR_V7_M1_SOURCE_PATH = SELECTOR_V7_M1_ROOT + "/src/sift_d0_selection_plan_generator_v7_m1.py"
SELECTOR_V7_M1_CONTRACT_PATH = SELECTOR_V7_M1_ROOT + "/metadata/SELECTION_PLAN_CONTRACT.md"
SELECTOR_V7_M1_SOURCE_SHA256 = "6985fe43b1a2c6a792118c7ed60307d4ce0c790d3352ea107f544ff580ba252c"
SELECTOR_V7_M1_CONTRACT_DOCUMENT_SHA256 = "76f407e94c857e46d9acad7b0a9f53477427ef27da5b7ed283d65858eab9dcd4"
SELECTOR_V7_M1_FUNCTIONAL_CONTRACT_SHA256 = "abd0411c485b1e07989d98fe56aed448412226833a2f985d149ddd9d7115b50b"


# Frozen published runner-v12-m1 artifact paths and raw byte pins.
RUNNER_V12_ROOT = "/archive-network-storage/archive_user/gts_concept_exact_squared_real_d0_v7_m1_postexpiry_runner_v12_sourceonly_20260809T103249CST"
RUNNER_V12_SOURCE_PATH = RUNNER_V12_ROOT + "/src/tide_real_d0_v4_terminalfinal_cpu_runner_v12_m1.cpp"
RUNNER_V12_CONTRACT_PATH = RUNNER_V12_ROOT + "/contract/RUNNER_CONTRACT_V12_M1.md"
RUNNER_V12_PREDICATE_PATH = RUNNER_V12_ROOT + "/contract/PREDICATE_BUNDLE_V3.json"
RUNNER_V12_SOURCE_SHA256 = "140b42bcd184fdf9ec0201ee887cf8348f2f04e8777b0d976c9b0ddd789103dd"
RUNNER_V12_CONTRACT_SHA256 = "01786b1cc37451d884adbaf32a778a66e0bac363627d6ff68a325ff0b96831ac"
RUNNER_V12_PREDICATE_SHA256 = "b55299f3d5fbb0aee3dbb0b2483e197170892f5366e69afd804b7e56dc268d64"

# Selector-v7-m1 literal contract values.  The issuer verifies them again after obtaining
# the functional selector contract from the raw SHA-pinned selector source bytes.
SEED_AUTHORIZATION_SCHEMA = "tide-d0-selection-plan-seed-authorization-v7-m1-private-local-ledger-half-open-validity"
SEED_AUTHORIZATION_STATUS = "LOCALLY_GOVERNED_ONE_SHOT_DEVELOPER_ONLY_UNSEALED_PRIVATE_LEDGER_V7_M1_HALF_OPEN_VALIDITY"
SEED_AUTHORIZATION_SCOPE = "developer_only_known_history_limited_unsealed_not_heldout_not_final"
COVERAGE_BINDING_PROTOCOL = "tide-d0-rank-first-match-coverage-v2-prebound"
FIXED_RESERVATION_LEAF = "d0_v4_m1_pinned_preflight_selection_plan.json"
COVERAGE_OUTCOME_LEAF = "d0_v4_pinned_preflight_selection_plan_coverage.json"

COVERAGE_OUTCOME_PATH = AUTHORIZATION_LEDGER_ROOT + "/outcomes/" + COVERAGE_OUTCOME_LEAF
PLAN_OUTPUT_PARENT = "/archive-network-storage/archive_user/sift_d0_selection_plan_outputs_v4_m1"
LOCAL_GOVERNANCE_LIMIT = (
    "Developer-only cooperative workflow: a local-ext4 private ledger requires opened-FD fstatfs="
    "EXT4_SUPER_MAGIC plus regular non-symlink same-EUID 0700 directories and a regular non-symlink "
    "same-EUID 0600 seed authorization, so seed_hex is never written to NAS; its permanent local O_EXCL "
    "reservation prevents accidental reuse but does not prove an adversarial one-shot seed ceremony or a sealed evaluation."
)
ALLOCATION = {
    "arrival_coverage_reservoir": 1024,
    "ordinary_arrivals": 128,
    "query_coverage_reservoir": 1024,
    "independent_natural_queries": 32,
}
FORBIDDEN_CAPABILITIES = [
    "gpu_or_cuda",
    "heldout_or_final_claim",
    "index_build_or_query",
    "raw_fvec_access",
    "route_or_receipt_candidate_selection",
    "seed_generation",
    "seed_resample_after_missing_coverage_condition",
    "trace_materialization",
]
PINNED_PREFLIGHT_ROOT = "/archive-network-storage/archive_user/sift_d0_v4_preflight_output_20260809T031633CST"
PINNED_PREFLIGHT_ANCHOR: dict[str, Any] = {
    "root": PINNED_PREFLIGHT_ROOT,
    "schema": "tide-sift-d0-payload-universe-preflight-v4",
    "status": "DEVELOPER_ONLY_KNOWN_HISTORY_LIMITED_UNSEALED",
    "committed_status": "COMMITTED_PREFLIGHT_ONLY",
    "manifest_sha256": "dc096b1429b63ae5e17e62d67b64bd812cd47379669b79b441e41a19add04076",
    "committed_sha256": "7f7bc61ad901d19bb8b558388cd828ebccf66790e2aedbeba1b6c5bca3461e0e",
    "artifact_sha256": {
        "base_role_multiset": "08861ef9299f47b9da04a2d89a6d23ee12dc6de709fe2837c2028f31cda75bfb",
        "eligible_arrival_universe": "e71e6aec14d00a9f7ec9b652ebd94810cf62d38115c535403b3d5299c6651752",
        "eligible_query_universe": "cf1c4a4b71b980062319b09af0588d5e6b6f5d261880268d6ab9ddcde7fcc898",
    },
    "source_sha256": {
        "base": "21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816",
        "learn": "331bc82b6a0e89465776a3ba0c2113e0bd0cceaa014ec3ed639bc8b981af72ea",
        "query": "f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc",
    },
    "exposure_registry_sha256": "eed8d4ef7a474ab42c6eb1806d62a2da10e25a4033681a38ad746524844db45e",
    "base_role_rows": 4096,
    "eligible_arrival_rows": 80613,
    "eligible_query_rows": 8726,
}
AUTHORIZATION_FIELDS = {
    "schema", "status", "scope", "authorization_id", "external_approval_id",
    "issued_unix_seconds", "not_before_unix_seconds", "expires_unix_seconds",
    "seed_hex", "seed_sha256", "selector_source_sha256",
    "selector_contract_sha256", "selector_contract_document_sha256",
    "preflight_anchor", "intended_preflight_root", "intended_output_root",
    "coverage_binding_protocol", "coverage_predicate_bundle_sha256",
    "coverage_runner_source_sha256", "coverage_runner_contract_sha256", "coverage_outcome_path",
    "allocation", "ledger_root", "forbidden_capabilities", "local_governance_limit",
}
ISSUER_CLAIM_SCHEMA = "tide-d0-local-ext4-issuer-claim-v5-postexpiry-m1-before-d0-draws-permanent-burn"
ISSUER_CLAIM_STATUS = "PERMANENT_POSTEXPIRY_M1_ISSUER_CLAIM_BEFORE_D0_DRAWS"
ISSUER_CLAIM_SCOPE = "developer_only_known_history_limited_unsealed_not_heldout_not_final_no_seed_material"
ISSUER_CLAIM_PATH = AUTHORIZATION_LEDGER_ROOT + "/authorizations/" + ISSUER_CLAIM_LEAF
ISSUER_CLAIM_FIELDS = {
    "schema", "status", "scope", "issuer_claim_path", "external_approval_id",
    "issued_unix_seconds", "not_before_unix_seconds", "expires_unix_seconds", "ttl_seconds",
    "selector_source_sha256", "selector_contract_sha256", "selector_contract_document_sha256",
    "preflight_anchor", "intended_preflight_root", "intended_output_root",
    "coverage_binding_protocol", "coverage_predicate_bundle_sha256",
    "coverage_runner_source_sha256", "coverage_runner_contract_sha256", "coverage_outcome_path",
    "allocation", "ledger_root", "forbidden_capabilities", "local_governance_limit",
}


class _LinuxStatFs(ctypes.Structure):
    _fields_ = [
        ("f_type", ctypes.c_long),
        ("f_bsize", ctypes.c_long),
        ("f_blocks", ctypes.c_ulong),
        ("f_bfree", ctypes.c_ulong),
        ("f_bavail", ctypes.c_ulong),
        ("f_files", ctypes.c_ulong),
        ("f_ffree", ctypes.c_ulong),
        ("f_fsid", ctypes.c_int * 2),
        ("f_namelen", ctypes.c_long),
        ("f_frsize", ctypes.c_long),
        ("f_flags", ctypes.c_long),
        ("f_spare", ctypes.c_long * 4),
    ]


def _fail(token: str) -> None:
    raise ControlPlaneError(token)


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _nofollow_flags(directory: bool, writable: bool = False, create: bool = False, exclusive: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or (directory and directory_flag is None):
        _fail("FAIL_CLOSED_O_NOFOLLOW_OR_DIRECTORY_UNAVAILABLE")
    flags = (os.O_WRONLY if writable else os.O_RDONLY) | nofollow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= directory_flag
    if create:
        flags |= os.O_CREAT
    if exclusive:
        flags |= os.O_EXCL
    return flags


def _identity(st: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode), st.st_mode & 0o7777, st.st_uid, st.st_gid, st.st_size)


def _stable_object_identity(st: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Object identity for a write transition; deliberately excludes mutable st_size."""
    return (st.st_dev, st.st_ino, stat.S_IFMT(st.st_mode), st.st_mode & 0o7777, st.st_uid, st.st_gid)



def _checked_close(fd: int, token: str) -> None:
    try:
        os.close(fd)
    except OSError as exc:
        raise ControlPlaneError(token) from exc


def _best_effort_close(fds: list[int]) -> None:
    for fd in reversed(fds):
        try:
            os.close(fd)
        except OSError:
            pass


def _require_linux_uid_1001() -> None:
    if sys.platform != "linux" or not hasattr(os, "getresuid"):
        _fail("FAIL_CLOSED_REQUIRED_LINUX_UID_API")
    try:
        real_uid, effective_uid, saved_uid = os.getresuid()
        account = pwd.getpwuid(REQUIRED_UID)
    except (OSError, KeyError) as exc:
        raise ControlPlaneError("FAIL_CLOSED_REQUIRED_IDENTITY") from exc
    if (
        (real_uid, effective_uid, saved_uid) != (REQUIRED_UID, REQUIRED_UID, REQUIRED_UID)
        or account.pw_name != REQUIRED_USERNAME
        or os.geteuid() != REQUIRED_UID
    ):
        _fail("FAIL_CLOSED_REQUIRED_UID_1001_ZHUXIAOMAO")


def _require_ext4_fd(fd: int, token: str) -> None:
    if sys.platform != "linux":
        _fail("FAIL_CLOSED_PRIVATE_LEDGER_NOT_LINUX")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fstatfs = libc.fstatfs
    except (AttributeError, OSError) as exc:
        raise ControlPlaneError("FAIL_CLOSED_PRIVATE_LEDGER_FSTATFS_UNAVAILABLE") from exc
    fstatfs.argtypes = [ctypes.c_int, ctypes.POINTER(_LinuxStatFs)]
    fstatfs.restype = ctypes.c_int
    observed = _LinuxStatFs()
    ctypes.set_errno(0)
    if fstatfs(fd, ctypes.byref(observed)) != 0:
        code = ctypes.get_errno()
        raise ControlPlaneError(token) from OSError(code, os.strerror(code))
    if int(observed.f_type) != EXT4_SUPER_MAGIC:
        _fail(token)


def _require_private_dir_fd(fd: int, token: str) -> os.stat_result:
    try:
        observed = os.fstat(fd)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    if (
        not stat.S_ISDIR(observed.st_mode)
        or (observed.st_mode & 0o7777) != PRIVATE_DIR_MODE
        or observed.st_uid != REQUIRED_UID
        or observed.st_uid != os.geteuid()
    ):
        _fail(token)
    _require_ext4_fd(fd, token)
    return observed


def _require_private_file_fd(fd: int, token: str) -> os.stat_result:
    try:
        observed = os.fstat(fd)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or (observed.st_mode & 0o7777) != PRIVATE_FILE_MODE
        or observed.st_uid != REQUIRED_UID
        or observed.st_uid != os.geteuid()
    ):
        _fail(token)
    _require_ext4_fd(fd, token)
    return observed


def _open_fixed_private_parent() -> int:
    """Open /workspace component-by-component without following a component symlink."""
    fds: list[int] = []
    try:
        root_fd = os.open("/", os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
        fds.append(root_fd)
        current = root_fd
        for component in ("home", "data", "archive_user"):
            try:
                next_fd = os.open(component, _nofollow_flags(directory=True), dir_fd=current)
            except OSError as exc:
                raise ControlPlaneError("FAIL_CLOSED_PRIVATE_PARENT_OPEN") from exc
            fds.append(next_fd)
            current = next_fd
        _require_private_dir_fd(current, "FAIL_CLOSED_PRIVATE_PARENT_REQUIREMENTS")
        for fd in fds[:-1]:
            _checked_close(fd, "FAIL_CLOSED_PRIVATE_PARENT_CLOSE")
        return current
    except Exception:
        _best_effort_close(fds)
        raise


def _open_private_child_dir(parent_fd: int, leaf: str, token: str) -> int:
    if leaf not in {LEDGER_LEAF, *LEDGER_CHILDREN}:
        _fail(token)
    try:
        fd = os.open(leaf, _nofollow_flags(directory=True), dir_fd=parent_fd)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    try:
        _require_private_dir_fd(fd, token)
        return fd
    except Exception:
        _best_effort_close([fd])
        raise


def _stat_child_nofollow(parent_fd: int, leaf: str, token: str) -> os.stat_result:
    try:
        observed = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    if stat.S_ISLNK(observed.st_mode):
        _fail(token)
    return observed


def _verify_child_readback(parent_fd: int, leaf: str, expected: os.stat_result, token: str) -> None:
    after_path = _stat_child_nofollow(parent_fd, leaf, token)
    if _identity(after_path) != _identity(expected):
        _fail(token)
    fd = _open_private_child_dir(parent_fd, leaf, token)
    try:
        after_fd = _require_private_dir_fd(fd, token)
        if _identity(after_fd) != _identity(expected):
            _fail(token)
    finally:
        _checked_close(fd, token)


def _precreate_umask_gate() -> None:
    """Reject an owner-bit-masking umask before any mkdir can leave an inaccessible 000 dir."""
    try:
        observed = os.umask(0)
        os.umask(observed)
    except OSError as exc:
        raise ControlPlaneError("FAIL_CLOSED_PRECREATE_UMASK") from exc
    if observed & 0o700:
        _fail("FAIL_CLOSED_PRECREATE_UMASK_OWNER_MASK")


def _create_new_private_child_dir(parent_fd: int, leaf: str, token: str) -> int:
    """Create one previously absent fixed child; EEXIST is a fail-closed race, never repair."""
    _precreate_umask_gate()
    try:
        os.mkdir(leaf, mode=PRIVATE_DIR_MODE, dir_fd=parent_fd)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    fd = _open_private_child_dir(parent_fd, leaf, token)
    try:
        # The zero-write umask gate guarantees owner bits survived mkdir. fchmod is only
        # applied to this retained newly-created FD, never to an existing ledger object.
        os.fchmod(fd, PRIVATE_DIR_MODE)
        expected = _require_private_dir_fd(fd, token)
        try:
            os.fsync(fd)
            os.fsync(parent_fd)
        except OSError as exc:
            raise ControlPlaneError(token) from exc
        _verify_child_readback(parent_fd, leaf, expected, token)
        return fd
    except Exception:
        _best_effort_close([fd])
        raise


def _probe_root_absent_or_existing(parent_fd: int) -> bool:
    """Return True only when the fixed root is absent; reject symlink/non-directory states."""
    try:
        observed = os.stat(LEDGER_LEAF, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError as exc:
        raise ControlPlaneError("FAIL_CLOSED_PREPARE_LEDGER_ROOT") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        _fail("FAIL_CLOSED_PREPARE_LEDGER_ROOT")
    return False


def _require_exact_ledger_root_children(root_fd: int, token: str) -> None:
    try:
        observed = set(os.listdir(root_fd))
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    if observed != set(LEDGER_CHILDREN):
        _fail(token)


def _require_empty_private_dir(fd: int, token: str) -> None:
    try:
        if os.listdir(fd):
            _fail(token)
    except OSError as exc:
        raise ControlPlaneError(token) from exc


def _open_existing_ledger_layout() -> tuple[int, int, int, int, int]:
    """Pin parent/workspace/admin/three child dirfds. This function never creates a ledger leaf."""
    parent_fd = _open_fixed_private_parent()
    fds = [parent_fd]
    try:
        root_fd = _open_private_child_dir(parent_fd, LEDGER_LEAF, "FAIL_CLOSED_LEDGER_ROOT")
        fds.append(root_fd)
        auth_fd = _open_private_child_dir(root_fd, AUTHORIZATION_DIR, "FAIL_CLOSED_LEDGER_AUTH_DIR")
        fds.append(auth_fd)
        reservation_fd = _open_private_child_dir(root_fd, RESERVATIONS_DIR, "FAIL_CLOSED_LEDGER_RESERVATIONS_DIR")
        fds.append(reservation_fd)
        outcomes_fd = _open_private_child_dir(root_fd, OUTCOMES_DIR, "FAIL_CLOSED_LEDGER_OUTCOMES_DIR")
        fds.append(outcomes_fd)
        _require_exact_ledger_root_children(root_fd, "FAIL_CLOSED_LEDGER_ROOT_FILESET")
        return parent_fd, root_fd, auth_fd, reservation_fd, outcomes_fd
    except Exception:
        _best_effort_close(fds)
        raise


def _read_regular_absolute(path_text: str, expected_sha: str, token: str, max_bytes: int = 32 << 20) -> bytes:
    """Read a fixed, regular, non-symlink leaf once and bind the returned bytes to SHA-256."""
    path = pathlib.PurePosixPath(path_text)
    if not path.is_absolute() or ".." in path.parts or str(path) != path_text:
        _fail(token)
    try:
        before_path = os.lstat(path_text)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
        _fail(token)
    try:
        fd = os.open(path_text, _nofollow_flags(directory=False))
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    try:
        before_fd = os.fstat(fd)
        if not stat.S_ISREG(before_fd.st_mode) or _identity(before_fd) != _identity(before_path):
            _fail(token)
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                block = os.read(fd, 1 << 20)
            except OSError as exc:
                raise ControlPlaneError(token) from exc
            if not block:
                break
            total += len(block)
            if total > max_bytes:
                _fail(token)
            chunks.append(block)
        after_fd = os.fstat(fd)
        if _identity(before_fd) != _identity(after_fd):
            _fail(token)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    finally:
        _checked_close(fd, token)
    try:
        after_path = os.lstat(path_text)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    if _identity(before_path) != _identity(after_path):
        _fail(token)
    raw = b"".join(chunks)
    if _sha256(raw) != expected_sha:
        _fail(token)
    return raw


def _validate_control_plane_contract() -> None:
    _read_regular_absolute(
        CONTROL_PLANE_CONTRACT_PATH,
        CONTROL_PLANE_CONTRACT_SHA256,
        "FAIL_CLOSED_CONTROL_PLANE_CONTRACT_BINDING",
    )


_SELECTOR_FUNCTIONAL_PROBE = r"""
import hashlib
import json
import sys
import types

source = sys.stdin.buffer.read()
module_name = "_tide_d0_selector_v7_m1_isolated_contract_probe"
module = types.ModuleType(module_name)
module.__file__ = "/workspace/sift_d0_selection_plan_generator_v7_m1_postexpiry_migration_sourceonly_20260809T101845CST/src/sift_d0_selection_plan_generator_v7_m1.py"
sys.modules[module_name] = module
exec(compile(source, module.__file__, "exec"), module.__dict__)
contract = module.selector_contract()
canonical = (json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
if module.canonical_json(contract) != canonical:
    raise RuntimeError("canonical")
payload = {"contract": contract, "functional_sha256": hashlib.sha256(canonical).hexdigest()}
sys.stdout.buffer.write((json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"))
"""


def _json_no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("FAIL_CLOSED_SELECTOR_PROBE_OUTPUT")
        result[key] = value
    return result


def _selector_functional_contract_sha256() -> str:
    """Evaluate only raw-SHA-pinned selector bytes in a separate isolated Python process.

    The parent issuer never execs selector bytes. The probe receives the bytes only on
    stdin, starts in `/`, uses `-I -S -B`, inherits no ledger descriptors, has stderr
    discarded, a short timeout, and returns only public functional-contract bytes. This
    is containment, not a hostile-code sandbox; the raw SHA pin remains the trust root.
    """
    source_raw = _read_regular_absolute(
        SELECTOR_V7_M1_SOURCE_PATH,
        SELECTOR_V7_M1_SOURCE_SHA256,
        "FAIL_CLOSED_SELECTOR_SOURCE_BINDING",
    )
    _read_regular_absolute(
        SELECTOR_V7_M1_CONTRACT_PATH,
        SELECTOR_V7_M1_CONTRACT_DOCUMENT_SHA256,
        "FAIL_CLOSED_SELECTOR_MARKDOWN_BINDING",
    )
    try:
        result = subprocess.run(
            ["/usr/bin/python3", "-I", "-S", "-B", "-c", _SELECTOR_FUNCTIONAL_PROBE],
            input=source_raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd="/",
            env={"LC_ALL": "C", "LANG": "C", "PATH": "/usr/bin:/bin"},
            close_fds=True,
            start_new_session=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ControlPlaneError("FAIL_CLOSED_SELECTOR_ISOLATED_PROBE") from exc
    if result.returncode != 0 or not result.stdout or len(result.stdout) > (2 << 20):
        _fail("FAIL_CLOSED_SELECTOR_ISOLATED_PROBE")
    try:
        payload = json.loads(result.stdout.decode("utf-8"), object_pairs_hook=_json_no_duplicate)
    except (UnicodeDecodeError, json.JSONDecodeError, ControlPlaneError) as exc:
        raise ControlPlaneError("FAIL_CLOSED_SELECTOR_PROBE_OUTPUT") from exc
    if not isinstance(payload, dict) or set(payload) != {"contract", "functional_sha256"}:
        _fail("FAIL_CLOSED_SELECTOR_PROBE_OUTPUT")
    contract = payload.get("contract")
    functional_sha = payload.get("functional_sha256")
    if not isinstance(contract, dict) or not isinstance(functional_sha, str) or len(functional_sha) != 64 or functional_sha.lower() != functional_sha or any(ch not in "0123456789abcdef" for ch in functional_sha):
        _fail("FAIL_CLOSED_SELECTOR_PROBE_OUTPUT")
    auth_contract = contract.get("seed_authorization")
    issuance_contract = auth_contract.get("private_ledger_issuance_requirements") if isinstance(auth_contract, dict) else None

    if (
        not isinstance(auth_contract, dict)
        or contract.get("plan_output_parent_root") != PLAN_OUTPUT_PARENT
        or contract.get("local_governance_limit") != LOCAL_GOVERNANCE_LIMIT
        or auth_contract.get("schema") != SEED_AUTHORIZATION_SCHEMA
        or auth_contract.get("status") != SEED_AUTHORIZATION_STATUS
        or auth_contract.get("scope") != SEED_AUTHORIZATION_SCOPE
        or auth_contract.get("authorization_document_root") != AUTHORIZATION_LEDGER_ROOT + "/authorizations"
        or auth_contract.get("reservation_root") != AUTHORIZATION_LEDGER_ROOT + "/reservations"
        or not isinstance(issuance_contract, dict)
        or issuance_contract.get("ledger_root") != AUTHORIZATION_LEDGER_ROOT
        or not isinstance(issuance_contract.get("reservation_file"), dict)
        or issuance_contract["reservation_file"].get("path_rule") != "reservations/" + FIXED_RESERVATION_LEAF
        or not isinstance(issuance_contract.get("outcome_file"), dict)
        or issuance_contract["outcome_file"].get("path_rule") != "outcomes/" + COVERAGE_OUTCOME_LEAF

        or auth_contract.get("coverage_outcome_path") != COVERAGE_OUTCOME_PATH
        or auth_contract.get("plan_output_parent_root") != PLAN_OUTPUT_PARENT
        or auth_contract.get("validity_interval") != "[not_before_unix_seconds, expires_unix_seconds)"
        or auth_contract.get("ttl_seconds") != [1, 3600]
        or auth_contract.get("required_fields") != [
            "schema", "status", "scope", "authorization_id", "external_approval_id",
            "issued_unix_seconds", "not_before_unix_seconds", "expires_unix_seconds",
            "seed_hex", "seed_sha256", "selector_source_sha256",
            "selector_contract_sha256", "selector_contract_document_sha256",
            "preflight_anchor", "intended_preflight_root", "intended_output_root",
            "coverage_binding_protocol", "coverage_predicate_bundle_sha256",
            "coverage_runner_source_sha256", "coverage_runner_contract_sha256", "coverage_outcome_path",
            "allocation", "ledger_root", "forbidden_capabilities", "local_governance_limit",
        ]
        or contract.get("preflight_input", {}).get("pinned_exact_anchor") != PINNED_PREFLIGHT_ANCHOR
        or contract.get("allocation") != ALLOCATION
        or contract.get("forbidden_capabilities") != FORBIDDEN_CAPABILITIES
    ):
        _fail("FAIL_CLOSED_SELECTOR_FUNCTIONAL_CONTRACT")
    if functional_sha != SELECTOR_V7_M1_FUNCTIONAL_CONTRACT_SHA256:
        _fail("FAIL_CLOSED_SELECTOR_FUNCTIONAL_PIN")

    if _sha256(_canonical_json(contract)) != functional_sha:
        _fail("FAIL_CLOSED_SELECTOR_FUNCTIONAL_CANONICALIZATION")
    return functional_sha


def _verify_frozen_runner_v12() -> None:
    _read_regular_absolute(RUNNER_V12_SOURCE_PATH, RUNNER_V12_SOURCE_SHA256, "FAIL_CLOSED_RUNNER_SOURCE_BINDING")
    _read_regular_absolute(RUNNER_V12_CONTRACT_PATH, RUNNER_V12_CONTRACT_SHA256, "FAIL_CLOSED_RUNNER_CONTRACT_BINDING")
    _read_regular_absolute(RUNNER_V12_PREDICATE_PATH, RUNNER_V12_PREDICATE_SHA256, "FAIL_CLOSED_RUNNER_PREDICATE_BINDING")


def _validate_approval_id(value: str) -> str:
    # Claim data is public/auditable but must be strict ASCII before permanent claim.
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or not value.isascii()
        or any(ch not in allowed for ch in value)
    ):
        _fail("FAIL_CLOSED_EXTERNAL_APPROVAL_ID")
    return value



def _validate_intended_output_root(value: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("FAIL_CLOSED_INTENDED_OUTPUT_ROOT")
    pure = pathlib.PurePosixPath(value)
    if not pure.is_absolute() or ".." in pure.parts or str(pure) != value:
        _fail("FAIL_CLOSED_INTENDED_OUTPUT_ROOT")
    if str(pure.parent) != PLAN_OUTPUT_PARENT:
        _fail("FAIL_CLOSED_INTENDED_OUTPUT_ROOT")
    name = pure.name
    if (
        not name.startswith("d0_v4_plan_")
        or len(name) == len("d0_v4_plan_")
        or len(name) > 160
        or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for ch in name)
    ):
        _fail("FAIL_CLOSED_INTENDED_OUTPUT_ROOT")
    return value


def _kernel_random_exact(nbytes: int) -> bytes:
    if sys.platform != "linux" or not hasattr(os, "getrandom") or nbytes <= 0:
        _fail("FAIL_CLOSED_KERNEL_RANDOM_UNAVAILABLE")
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining:
        try:
            chunk = os.getrandom(remaining, 0)
        except InterruptedError:
            continue
        except OSError as exc:
            raise ControlPlaneError("FAIL_CLOSED_KERNEL_RANDOM_UNAVAILABLE") from exc
        if not chunk:
            _fail("FAIL_CLOSED_KERNEL_RANDOM_UNAVAILABLE")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_authorization_id(value: str) -> str:
    if not isinstance(value, str) or len(value) != 32 or value.lower() != value or any(ch not in "0123456789abcdef" for ch in value):
        _fail("FAIL_CLOSED_AUTHORIZATION_ID")
    return value


def _validate_authorization_document(
    document: dict[str, Any],
    *,
    authorization_id: str,
    external_approval_id: str,
    issued: int,
    expires: int,
    intended_output_root: str,
    selector_contract_sha256: str,
) -> None:
    if set(document) != AUTHORIZATION_FIELDS:
        _fail("FAIL_CLOSED_AUTHORIZATION_SCHEMA")
    seed_hex = document.get("seed_hex")
    seed_sha = document.get("seed_sha256")
    if (
        document.get("schema") != SEED_AUTHORIZATION_SCHEMA
        or document.get("status") != SEED_AUTHORIZATION_STATUS
        or document.get("scope") != SEED_AUTHORIZATION_SCOPE
        or document.get("authorization_id") != authorization_id
        or document.get("external_approval_id") != external_approval_id
        or document.get("issued_unix_seconds") != issued
        or document.get("not_before_unix_seconds") != issued
        or document.get("expires_unix_seconds") != expires
        or not isinstance(seed_hex, str)
        or len(seed_hex) != 64
        or seed_hex.lower() != seed_hex
        or any(ch not in "0123456789abcdef" for ch in seed_hex)
        or not isinstance(seed_sha, str)
        or seed_sha != _sha256(bytes.fromhex(seed_hex))
        or document.get("selector_source_sha256") != SELECTOR_V7_M1_SOURCE_SHA256
        or document.get("selector_contract_sha256") != selector_contract_sha256
        or document.get("selector_contract_document_sha256") != SELECTOR_V7_M1_CONTRACT_DOCUMENT_SHA256
        or document.get("preflight_anchor") != PINNED_PREFLIGHT_ANCHOR
        or document.get("intended_preflight_root") != PINNED_PREFLIGHT_ROOT
        or document.get("intended_output_root") != intended_output_root
        or document.get("coverage_binding_protocol") != COVERAGE_BINDING_PROTOCOL
        or document.get("coverage_predicate_bundle_sha256") != RUNNER_V12_PREDICATE_SHA256
        or document.get("coverage_runner_source_sha256") != RUNNER_V12_SOURCE_SHA256
        or document.get("coverage_runner_contract_sha256") != RUNNER_V12_CONTRACT_SHA256
        or document.get("coverage_outcome_path") != COVERAGE_OUTCOME_PATH
        or document.get("allocation") != ALLOCATION
        or document.get("ledger_root") != AUTHORIZATION_LEDGER_ROOT
        or document.get("forbidden_capabilities") != FORBIDDEN_CAPABILITIES
        or document.get("local_governance_limit") != LOCAL_GOVERNANCE_LIMIT
    ):
        _fail("FAIL_CLOSED_AUTHORIZATION_SCHEMA")


def _read_private_regular_at(parent_fd: int, leaf: str, expected_identity: tuple[int, int, int, int, int, int, int], token: str) -> bytes:
    if not leaf or "/" in leaf or leaf in {".", ".."}:
        _fail(token)
    before_path = _stat_child_nofollow(parent_fd, leaf, token)
    if not stat.S_ISREG(before_path.st_mode):
        _fail(token)
    try:
        fd = os.open(leaf, _nofollow_flags(directory=False), dir_fd=parent_fd)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    try:
        before_fd = _require_private_file_fd(fd, token)
        if _identity(before_path) != _identity(before_fd) or _identity(before_fd) != expected_identity:
            _fail(token)
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after_fd = _require_private_file_fd(fd, token)
        if _identity(after_fd) != expected_identity:
            _fail(token)
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    finally:
        _checked_close(fd, token)
    after_path = _stat_child_nofollow(parent_fd, leaf, token)
    if _identity(after_path) != expected_identity:
        _fail(token)
    return b"".join(chunks)


def _write_private_json_exclusive(auth_dir_fd: int, leaf: str, raw: bytes, token_prefix: str) -> str:
    if not leaf or "/" in leaf or leaf in {".", ".."}:
        _fail("FAIL_CLOSED_" + token_prefix + "_LEAF")
    try:
        fd = os.open(
            leaf,
            _nofollow_flags(directory=False, writable=True, create=True, exclusive=True),
            PRIVATE_FILE_MODE,
            dir_fd=auth_dir_fd,
        )
    except FileExistsError as exc:
        raise ControlPlaneError("FAIL_CLOSED_" + token_prefix + "_ALREADY_EXISTS") from exc
    except OSError as exc:
        raise ControlPlaneError("FAIL_CLOSED_" + token_prefix + "_CREATE") from exc
    try:
        os.fchmod(fd, PRIVATE_FILE_MODE)
        created = _require_private_file_fd(fd, "FAIL_CLOSED_" + token_prefix + "_FILE_REQUIREMENTS")
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                _fail("FAIL_CLOSED_" + token_prefix + "_WRITE")
            offset += written
        os.fsync(fd)
        after_write = _require_private_file_fd(fd, "FAIL_CLOSED_" + token_prefix + "_FILE_REQUIREMENTS")
        if _stable_object_identity(created) != _stable_object_identity(after_write) or after_write.st_size != len(raw):
            _fail("FAIL_CLOSED_" + token_prefix + "_FILE_REQUIREMENTS")
    except OSError as exc:
        raise ControlPlaneError("FAIL_CLOSED_" + token_prefix + "_WRITE") from exc
    finally:
        _checked_close(fd, "FAIL_CLOSED_" + token_prefix + "_CLOSE")
    try:
        os.fsync(auth_dir_fd)
    except OSError as exc:
        raise ControlPlaneError("FAIL_CLOSED_" + token_prefix + "_DIRECTORY_FSYNC") from exc
    readback = _read_private_regular_at(
        auth_dir_fd,
        leaf,
        _identity(after_write),
        "FAIL_CLOSED_" + token_prefix + "_READBACK",
    )
    if readback != raw:
        _fail("FAIL_CLOSED_" + token_prefix + "_READBACK")
    return _sha256(raw)


def _write_issuer_claim_exclusive(auth_dir_fd: int, raw: bytes) -> str:
    return _write_private_json_exclusive(auth_dir_fd, ISSUER_CLAIM_LEAF, raw, "ISSUER_CLAIM")


def _write_authorization_exclusive(auth_dir_fd: int, authorization_id: str, raw: bytes) -> str:
    return _write_private_json_exclusive(auth_dir_fd, authorization_id + ".json", raw, "AUTHORIZATION")


def _require_exact_private_dir_entries(fd: int, expected: set[str], token: str) -> None:
    try:
        observed = set(os.listdir(fd))
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    if observed != expected:
        _fail(token)


def _issuer_claim_document(
    *,
    external_approval_id: str,
    issued: int,
    expires: int,
    intended_output_root: str,
    selector_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": ISSUER_CLAIM_SCHEMA,
        "status": ISSUER_CLAIM_STATUS,
        "scope": ISSUER_CLAIM_SCOPE,
        "issuer_claim_path": ISSUER_CLAIM_PATH,
        "external_approval_id": external_approval_id,
        "issued_unix_seconds": issued,
        "not_before_unix_seconds": issued,
        "expires_unix_seconds": expires,
        "ttl_seconds": FIXED_TTL_SECONDS,
        "selector_source_sha256": SELECTOR_V7_M1_SOURCE_SHA256,
        "selector_contract_sha256": selector_contract_sha256,
        "selector_contract_document_sha256": SELECTOR_V7_M1_CONTRACT_DOCUMENT_SHA256,
        "preflight_anchor": PINNED_PREFLIGHT_ANCHOR,
        "intended_preflight_root": PINNED_PREFLIGHT_ROOT,
        "intended_output_root": intended_output_root,
        "coverage_binding_protocol": COVERAGE_BINDING_PROTOCOL,
        "coverage_predicate_bundle_sha256": RUNNER_V12_PREDICATE_SHA256,
        "coverage_runner_source_sha256": RUNNER_V12_SOURCE_SHA256,
        "coverage_runner_contract_sha256": RUNNER_V12_CONTRACT_SHA256,
        "coverage_outcome_path": COVERAGE_OUTCOME_PATH,
        "allocation": ALLOCATION,
        "ledger_root": AUTHORIZATION_LEDGER_ROOT,
        "forbidden_capabilities": FORBIDDEN_CAPABILITIES,
        "local_governance_limit": LOCAL_GOVERNANCE_LIMIT,
    }


def _validate_issuer_claim_document(
    document: dict[str, Any],
    *,
    external_approval_id: str,
    issued: int,
    expires: int,
    intended_output_root: str,
    selector_contract_sha256: str,
) -> None:
    expected = _issuer_claim_document(
        external_approval_id=external_approval_id,
        issued=issued,
        expires=expires,
        intended_output_root=intended_output_root,
        selector_contract_sha256=selector_contract_sha256,
    )
    if set(document) != ISSUER_CLAIM_FIELDS or document != expected:
        _fail("FAIL_CLOSED_ISSUER_CLAIM_SCHEMA")


def prepare_ledger() -> dict[str, Any]:
    """Create only a wholly new fixed ledger; never repair an existing root."""
    _require_linux_uid_1001()
    _validate_control_plane_contract()
    fds: list[int] = []
    try:
        parent_fd = _open_fixed_private_parent()
        fds.append(parent_fd)
        root_absent = _probe_root_absent_or_existing(parent_fd)
        if root_absent:
            root_fd = _create_new_private_child_dir(parent_fd, LEDGER_LEAF, "FAIL_CLOSED_PREPARE_LEDGER_ROOT")
            fds.append(root_fd)
            # A root created by this invocation must still be empty before any child mkdir.
            _require_empty_private_dir(root_fd, "FAIL_CLOSED_PREPARE_NEW_ROOT_NOT_EMPTY")
            auth_fd = _create_new_private_child_dir(root_fd, AUTHORIZATION_DIR, "FAIL_CLOSED_PREPARE_AUTH_DIR")
            fds.append(auth_fd)
            reservation_fd = _create_new_private_child_dir(root_fd, RESERVATIONS_DIR, "FAIL_CLOSED_PREPARE_RESERVATIONS_DIR")
            fds.append(reservation_fd)
            outcomes_fd = _create_new_private_child_dir(root_fd, OUTCOMES_DIR, "FAIL_CLOSED_PREPARE_OUTCOMES_DIR")
            fds.append(outcomes_fd)
        else:
            # Critical ordering: inspect closed child set *before* any mkdir. An existing
            # empty/partial/malformed root is a permanent fail-closed state, never repaired.
            root_fd = _open_private_child_dir(parent_fd, LEDGER_LEAF, "FAIL_CLOSED_PREPARE_LEDGER_ROOT")
            fds.append(root_fd)
            _require_exact_ledger_root_children(root_fd, "FAIL_CLOSED_PREPARE_EXISTING_ROOT_FILESET")
            auth_fd = _open_private_child_dir(root_fd, AUTHORIZATION_DIR, "FAIL_CLOSED_PREPARE_AUTH_DIR")
            fds.append(auth_fd)
            reservation_fd = _open_private_child_dir(root_fd, RESERVATIONS_DIR, "FAIL_CLOSED_PREPARE_RESERVATIONS_DIR")
            fds.append(reservation_fd)
            outcomes_fd = _open_private_child_dir(root_fd, OUTCOMES_DIR, "FAIL_CLOSED_PREPARE_OUTCOMES_DIR")
            fds.append(outcomes_fd)
        _require_exact_ledger_root_children(root_fd, "FAIL_CLOSED_PREPARE_LEDGER_FILESET")
        try:
            os.fsync(root_fd)
            os.fsync(parent_fd)
        except OSError as exc:
            raise ControlPlaneError("FAIL_CLOSED_PREPARE_FINAL_FSYNC") from exc
        _verify_child_readback(parent_fd, LEDGER_LEAF, _require_private_dir_fd(root_fd, "FAIL_CLOSED_PREPARE_READBACK"), "FAIL_CLOSED_PREPARE_READBACK")
        _verify_child_readback(root_fd, AUTHORIZATION_DIR, _require_private_dir_fd(auth_fd, "FAIL_CLOSED_PREPARE_READBACK"), "FAIL_CLOSED_PREPARE_READBACK")
        _verify_child_readback(root_fd, RESERVATIONS_DIR, _require_private_dir_fd(reservation_fd, "FAIL_CLOSED_PREPARE_READBACK"), "FAIL_CLOSED_PREPARE_READBACK")
        _verify_child_readback(root_fd, OUTCOMES_DIR, _require_private_dir_fd(outcomes_fd, "FAIL_CLOSED_PREPARE_READBACK"), "FAIL_CLOSED_PREPARE_READBACK")
        for fd in reversed(fds):
            _checked_close(fd, "FAIL_CLOSED_PREPARE_CLOSE")
        fds.clear()
        return {
            "status": "PRIVATE_LOCAL_EXT4_LEDGER_DIRECTORIES_PREPARED_ONLY",
            "scope": "developer_only_unsealed_control_plane_directory_preparation_only_no_auth_no_seed_no_plan_no_fvec_no_gpu",
            "ledger_root": AUTHORIZATION_LEDGER_ROOT,
            "uid": REQUIRED_UID,
        }
    except Exception:
        _best_effort_close(fds)
        raise


def issue_authorization(external_approval_id: str, intended_output_root: str) -> dict[str, Any]:
    """Persist the fixed claim before this control plane's D0 seed/authorization-ID draws, then issue one v7-m1-compatible local authorization."""
    _require_linux_uid_1001()
    _validate_control_plane_contract()
    external_approval_id = _validate_approval_id(external_approval_id)
    intended_output_root = _validate_intended_output_root(intended_output_root)
    # All deterministic source/contract/predicate validation is complete before claim.
    selector_contract_sha256 = _selector_functional_contract_sha256()
    _verify_frozen_runner_v12()

    fds: list[int] = []
    try:
        parent_fd, root_fd, auth_fd, reservation_fd, outcomes_fd = _open_existing_ledger_layout()
        fds.extend((parent_fd, root_fd, auth_fd, reservation_fd, outcomes_fd))
        # Pre-claim acceptance is exactly the empty authorization directory and empty
        # selector/runner-owned sibling directories. Any pre-existing leaf burns the slot.
        _require_exact_private_dir_entries(auth_fd, set(), "FAIL_CLOSED_AUTHORIZATION_SLOT_ALREADY_USED")
        _require_empty_private_dir(reservation_fd, "FAIL_CLOSED_RESERVATION_SLOT_ALREADY_USED")
        _require_empty_private_dir(outcomes_fd, "FAIL_CLOSED_OUTCOME_SLOT_ALREADY_USED")

        issued = int(time.time())
        if issued < 0:
            _fail("FAIL_CLOSED_REALTIME_CLOCK")
        expires = issued + FIXED_TTL_SECONDS
        claim = _issuer_claim_document(
            external_approval_id=external_approval_id,
            issued=issued,
            expires=expires,
            intended_output_root=intended_output_root,
            selector_contract_sha256=selector_contract_sha256,
        )
        _validate_issuer_claim_document(
            claim,
            external_approval_id=external_approval_id,
            issued=issued,
            expires=expires,
            intended_output_root=intended_output_root,
            selector_contract_sha256=selector_contract_sha256,
        )
        issuer_claim_sha256 = _write_issuer_claim_exclusive(auth_fd, _canonical_json(claim))
        # A returned writer result means its exclusive write, file/dir fsync, and same-dirfd
        # readback succeeded. If it raises, this invocation cannot draw a D0 seed/ID or retry.
        # This postclaim set is checked before this control plane's D0 seed and authorization-ID
        # _kernel_random_exact draws. A concurrent loser fails at O_EXCL above first.
        _require_exact_private_dir_entries(auth_fd, {ISSUER_CLAIM_LEAF}, "FAIL_CLOSED_ISSUER_CLAIM_FILESET")

        seed = _kernel_random_exact(32)
        authorization_id = _validate_authorization_id(_kernel_random_exact(16).hex())
        seed_hex = seed.hex()
        document: dict[str, Any] = {
            "schema": SEED_AUTHORIZATION_SCHEMA,
            "status": SEED_AUTHORIZATION_STATUS,
            "scope": SEED_AUTHORIZATION_SCOPE,
            "authorization_id": authorization_id,
            "external_approval_id": external_approval_id,
            "issued_unix_seconds": issued,
            "not_before_unix_seconds": issued,
            "expires_unix_seconds": expires,
            "seed_hex": seed_hex,
            "seed_sha256": _sha256(seed),
            "selector_source_sha256": SELECTOR_V7_M1_SOURCE_SHA256,
            "selector_contract_sha256": selector_contract_sha256,
            "selector_contract_document_sha256": SELECTOR_V7_M1_CONTRACT_DOCUMENT_SHA256,
            "preflight_anchor": PINNED_PREFLIGHT_ANCHOR,
            "intended_preflight_root": PINNED_PREFLIGHT_ROOT,
            "intended_output_root": intended_output_root,
            "coverage_binding_protocol": COVERAGE_BINDING_PROTOCOL,
            "coverage_predicate_bundle_sha256": RUNNER_V12_PREDICATE_SHA256,
            "coverage_runner_source_sha256": RUNNER_V12_SOURCE_SHA256,
            "coverage_runner_contract_sha256": RUNNER_V12_CONTRACT_SHA256,
            "coverage_outcome_path": COVERAGE_OUTCOME_PATH,
            "allocation": ALLOCATION,
            "ledger_root": AUTHORIZATION_LEDGER_ROOT,
            "forbidden_capabilities": FORBIDDEN_CAPABILITIES,
            "local_governance_limit": LOCAL_GOVERNANCE_LIMIT,
        }
        _validate_authorization_document(
            document,
            authorization_id=authorization_id,
            external_approval_id=external_approval_id,
            issued=issued,
            expires=expires,
            intended_output_root=intended_output_root,
            selector_contract_sha256=selector_contract_sha256,
        )
        authorization_sha256 = _write_authorization_exclusive(auth_fd, authorization_id, _canonical_json(document))
        _require_exact_private_dir_entries(
            auth_fd,
            {ISSUER_CLAIM_LEAF, authorization_id + ".json"},
            "FAIL_CLOSED_AUTHORIZATION_POSTWRITE_FILESET",
        )

        for fd in reversed(fds):
            _checked_close(fd, "FAIL_CLOSED_ISSUER_CLOSE")
        fds.clear()
        return {
            "status": "LOCAL_EXT4_V7_M1_AUTHORIZATION_ISSUED_AFTER_PERMANENT_POSTEXPIRY_M1_ISSUER_CLAIM",
            "scope": "developer_only_known_history_limited_unsealed_not_heldout_not_final; local O_EXCL workflow protection only",
            "authorization_id": authorization_id,
            "authorization_path": AUTHORIZATION_LEDGER_ROOT + "/authorizations/" + authorization_id + ".json",
            "authorization_file_sha256": authorization_sha256,
            "issuer_claim_path": ISSUER_CLAIM_PATH,
            "issuer_claim_sha256": issuer_claim_sha256,
            "intended_output_root": intended_output_root,
            "not_before_unix_seconds": issued,
            "expires_unix_seconds": expires,
            "ttl_seconds": FIXED_TTL_SECONDS,
            "selector_source_sha256": SELECTOR_V7_M1_SOURCE_SHA256,
            "selector_contract_sha256": selector_contract_sha256,
            "selector_contract_document_sha256": SELECTOR_V7_M1_CONTRACT_DOCUMENT_SHA256,
            "runner_source_sha256": RUNNER_V12_SOURCE_SHA256,
            "runner_contract_sha256": RUNNER_V12_CONTRACT_SHA256,
            "predicate_bundle_sha256": RUNNER_V12_PREDICATE_SHA256,
        }
    except Exception:
        _best_effort_close(fds)
        raise


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-ledger", help="Create only the fixed private ext4 ledger directories.")
    issue = commands.add_parser("issue-authorization", help="Create one local selector-v7-m1 authorization; no CLI seed exists.")
    issue.add_argument("--external-approval-id", required=True)
    issue.add_argument("--intended-output-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-ledger":
            result = prepare_ledger()
        elif args.command == "issue-authorization":
            result = issue_authorization(args.external_approval_id, args.intended_output_root)
        else:
            _fail("FAIL_CLOSED_COMMAND")
        # The returned object intentionally excludes seed_hex and seed_sha256.
        print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0
    except ControlPlaneError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception:
        print("FAIL_CLOSED_UNEXPECTED", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
