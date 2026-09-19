#!/usr/bin/env python3
"""Source-only D0 local-ext4 control plane: prepare the private ledger or issue one auth.

This tool is intentionally separate from selector v6 and runner v11.  It has no plan,
trace, index, FVEC, CUDA, or GPU command.  The normal issuer has no CLI seed input:
it obtains the 256-bit seed only from Linux getrandom(2), writes it only to the fixed
private ext4 authorization leaf, and never prints it.
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
import sys
import time
import types
from typing import Any


class ControlPlaneError(RuntimeError):
    """Public failures carry fixed tokens only; never serialize secret-bearing state."""


# Identity and private-ledger namespace are deliberately fixed, never CLI-configurable.
REQUIRED_UID = 1001
REQUIRED_USERNAME = "zhuxiaomao"
PRIVATE_PARENT = "/workspace"
LEDGER_LEAF = ".tide_d0_auth_ledger_v5"
AUTHORIZATION_LEDGER_ROOT = PRIVATE_PARENT + "/" + LEDGER_LEAF
AUTHORIZATION_DIR = "authorizations"
RESERVATIONS_DIR = "reservations"
OUTCOMES_DIR = "outcomes"
LEDGER_CHILDREN = (AUTHORIZATION_DIR, RESERVATIONS_DIR, OUTCOMES_DIR)
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
EXT4_SUPER_MAGIC = 0xEF53

# This source verifies its fixed Markdown contract before either command.  The source
# itself is frozen by the source-only SHA256SUMS manifest, avoiding a self-hash cycle.
CONTROL_PLANE_ROOT = "/workspace/tide_d0_control_plane_v1_sourceonly_20260809T103000CST"
CONTROL_PLANE_CONTRACT_PATH = CONTROL_PLANE_ROOT + "/metadata/CONTROL_PLANE_CONTRACT.md"
CONTROL_PLANE_CONTRACT_SHA256 = "664a0d7261ba50ffaf4f636bbf0ddc77d63f63c050a0e5f30b38e4e04876a3d4"

# Frozen selector-v6 artifact paths and raw byte pins.
SELECTOR_V6_ROOT = "/workspace/sift_d0_selection_plan_generator_v6_halfopenauth_20260809T081000CST"
SELECTOR_V6_SOURCE_PATH = SELECTOR_V6_ROOT + "/src/sift_d0_selection_plan_generator_v6.py"
SELECTOR_V6_CONTRACT_PATH = SELECTOR_V6_ROOT + "/metadata/SELECTION_PLAN_CONTRACT.md"
SELECTOR_V6_SOURCE_SHA256 = "8ce7a0e776e9eaae8d0e11b51dfa5ee6cc860b2f9f38b00021baa321260c5acd"
SELECTOR_V6_CONTRACT_DOCUMENT_SHA256 = "486410344e63bca5a1c9cfdf5262e4d5bb1961bbaecb07f9a860bce7549d8aa5"

# Frozen published runner-v11 artifact paths and raw byte pins.
RUNNER_V11_ROOT = "/archive-network-storage/archive_user/gts_concept_exact_squared_real_d0_v6_halfopenauth_runner_v11_sourceonly_20260809T090500CST"
RUNNER_V11_SOURCE_PATH = RUNNER_V11_ROOT + "/src/tide_real_d0_v4_terminalfinal_cpu_runner_v5.cpp"
RUNNER_V11_CONTRACT_PATH = RUNNER_V11_ROOT + "/contract/RUNNER_CONTRACT_V5.md"
RUNNER_V11_PREDICATE_PATH = RUNNER_V11_ROOT + "/contract/PREDICATE_BUNDLE_V3.json"
RUNNER_V11_SOURCE_SHA256 = "0798b9050c2bcf2243bfdc025c6b0e13f0bd16adbcf81aa170df8e715bd2602b"
RUNNER_V11_CONTRACT_SHA256 = "6b796ff06da9f24ecf51ee824f7b3e9c2261eaad9202ef9e2fb756c50e3ba7db"
RUNNER_V11_PREDICATE_SHA256 = "b55299f3d5fbb0aee3dbb0b2483e197170892f5366e69afd804b7e56dc268d64"

# Selector-v6 literal contract values.  The issuer verifies them again after obtaining
# the functional selector contract from the raw SHA-pinned selector source bytes.
SEED_AUTHORIZATION_SCHEMA = "tide-d0-selection-plan-seed-authorization-v6-private-local-ledger-half-open-validity"
SEED_AUTHORIZATION_STATUS = "LOCALLY_GOVERNED_ONE_SHOT_DEVELOPER_ONLY_UNSEALED_PRIVATE_LEDGER_HALF_OPEN_VALIDITY"
SEED_AUTHORIZATION_SCOPE = "developer_only_known_history_limited_unsealed_not_heldout_not_final"
COVERAGE_BINDING_PROTOCOL = "tide-d0-rank-first-match-coverage-v2-prebound"
COVERAGE_OUTCOME_PATH = AUTHORIZATION_LEDGER_ROOT + "/outcomes/d0_v4_pinned_preflight_selection_plan_coverage.json"
PLAN_OUTPUT_PARENT = "/archive-network-storage/archive_user/sift_d0_selection_plan_outputs_v4"
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


def _ensure_private_child_dir(parent_fd: int, leaf: str, token: str) -> int:
    """Create only a fixed direct child, fsync it and its parent, then reopen/read back."""
    created = False
    try:
        before = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.mkdir(leaf, mode=PRIVATE_DIR_MODE, dir_fd=parent_fd)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise ControlPlaneError(token) from exc
    except OSError as exc:
        raise ControlPlaneError(token) from exc
    else:
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            _fail(token)

    fd = _open_private_child_dir(parent_fd, leaf, token)
    try:
        if created:
            # mkdir is never wider than 0700; fchmod only restores owner bits masked by
            # an unusual umask while the returned private FD is retained.
            try:
                os.fchmod(fd, PRIVATE_DIR_MODE)
            except OSError as exc:
                raise ControlPlaneError(token) from exc
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


def _selector_functional_contract_sha256() -> str:
    """Derive the exact canonical selector_contract() hash from raw SHA-pinned source bytes.

    Direct compile/exec of the verified bytes avoids importing by a mutable module path and
    disables bytecode generation. The source is given a non-__main__ module name, so its
    CLI entrypoint cannot run. Only selector_contract()/canonical_json() are called.
    """
    source_raw = _read_regular_absolute(
        SELECTOR_V6_SOURCE_PATH,
        SELECTOR_V6_SOURCE_SHA256,
        "FAIL_CLOSED_SELECTOR_SOURCE_BINDING",
    )
    _read_regular_absolute(
        SELECTOR_V6_CONTRACT_PATH,
        SELECTOR_V6_CONTRACT_DOCUMENT_SHA256,
        "FAIL_CLOSED_SELECTOR_MARKDOWN_BINDING",
    )
    module_name = "_tide_d0_selector_v6_control_plane_snapshot"
    old_module = sys.modules.get(module_name)
    module = types.ModuleType(module_name)
    module.__file__ = SELECTOR_V6_SOURCE_PATH
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.modules[module_name] = module
    sys.dont_write_bytecode = True
    try:
        exec(compile(source_raw, SELECTOR_V6_SOURCE_PATH, "exec"), module.__dict__)
        selector_contract = getattr(module, "selector_contract", None)
        canonical_json = getattr(module, "canonical_json", None)
        preflight_anchor = getattr(module, "d0_preflight_anchor", None)
        if not callable(selector_contract) or not callable(canonical_json) or not callable(preflight_anchor):
            _fail("FAIL_CLOSED_SELECTOR_FUNCTIONAL_CONTRACT")
        contract = selector_contract()
        auth_contract = contract.get("seed_authorization") if isinstance(contract, dict) else None
        if (
            not isinstance(auth_contract, dict)
            or contract.get("plan_output_parent_root") != PLAN_OUTPUT_PARENT
            or contract.get("local_governance_limit") != LOCAL_GOVERNANCE_LIMIT
            or preflight_anchor() != PINNED_PREFLIGHT_ANCHOR
            or auth_contract.get("schema") != SEED_AUTHORIZATION_SCHEMA
            or auth_contract.get("status") != SEED_AUTHORIZATION_STATUS
            or auth_contract.get("scope") != SEED_AUTHORIZATION_SCOPE
            or auth_contract.get("authorization_document_root") != AUTHORIZATION_LEDGER_ROOT + "/authorizations"
            or auth_contract.get("reservation_root") != AUTHORIZATION_LEDGER_ROOT + "/reservations"
            or auth_contract.get("coverage_outcome_path") != COVERAGE_OUTCOME_PATH
            or auth_contract.get("plan_output_parent_root") != PLAN_OUTPUT_PARENT
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
            or getattr(module, "ALLOC", None) != ALLOCATION
            or sorted(getattr(module, "FORBIDDEN", ())) != FORBIDDEN_CAPABILITIES
        ):
            _fail("FAIL_CLOSED_SELECTOR_FUNCTIONAL_CONTRACT")
        local_canonical = _canonical_json(contract)
        module_canonical = canonical_json(contract)
        if not isinstance(module_canonical, bytes) or module_canonical != local_canonical:
            _fail("FAIL_CLOSED_SELECTOR_FUNCTIONAL_CANONICALIZATION")
        return _sha256(local_canonical)
    except ControlPlaneError:
        raise
    except Exception as exc:
        raise ControlPlaneError("FAIL_CLOSED_SELECTOR_FUNCTIONAL_CONTRACT") from exc
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
        if old_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = old_module


def _verify_frozen_runner_v11() -> None:
    _read_regular_absolute(RUNNER_V11_SOURCE_PATH, RUNNER_V11_SOURCE_SHA256, "FAIL_CLOSED_RUNNER_SOURCE_BINDING")
    _read_regular_absolute(RUNNER_V11_CONTRACT_PATH, RUNNER_V11_CONTRACT_SHA256, "FAIL_CLOSED_RUNNER_CONTRACT_BINDING")
    _read_regular_absolute(RUNNER_V11_PREDICATE_PATH, RUNNER_V11_PREDICATE_SHA256, "FAIL_CLOSED_RUNNER_PREDICATE_BINDING")


def _validate_approval_id(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or "\n" in value or "\x00" in value:
        _fail("FAIL_CLOSED_EXTERNAL_APPROVAL_ID")
    return value


def _validate_ttl(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 3600:
        _fail("FAIL_CLOSED_TTL_SECONDS")
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
        or document.get("selector_source_sha256") != SELECTOR_V6_SOURCE_SHA256
        or document.get("selector_contract_sha256") != selector_contract_sha256
        or document.get("selector_contract_document_sha256") != SELECTOR_V6_CONTRACT_DOCUMENT_SHA256
        or document.get("preflight_anchor") != PINNED_PREFLIGHT_ANCHOR
        or document.get("intended_preflight_root") != PINNED_PREFLIGHT_ROOT
        or document.get("intended_output_root") != intended_output_root
        or document.get("coverage_binding_protocol") != COVERAGE_BINDING_PROTOCOL
        or document.get("coverage_predicate_bundle_sha256") != RUNNER_V11_PREDICATE_SHA256
        or document.get("coverage_runner_source_sha256") != RUNNER_V11_SOURCE_SHA256
        or document.get("coverage_runner_contract_sha256") != RUNNER_V11_CONTRACT_SHA256
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


def _write_authorization_exclusive(auth_dir_fd: int, authorization_id: str, raw: bytes) -> str:
    leaf = authorization_id + ".json"
    try:
        fd = os.open(
            leaf,
            _nofollow_flags(directory=False, writable=True, create=True, exclusive=True),
            PRIVATE_FILE_MODE,
            dir_fd=auth_dir_fd,
        )
    except FileExistsError as exc:
        raise ControlPlaneError("FAIL_CLOSED_AUTHORIZATION_ALREADY_EXISTS") from exc
    except OSError as exc:
        raise ControlPlaneError("FAIL_CLOSED_AUTHORIZATION_CREATE") from exc
    try:
        # The initial create mode has no group/world bits. This fchmod only restores any
        # owner bits masked by an unusual umask while the returned private fd is retained.
        os.fchmod(fd, PRIVATE_FILE_MODE)
        created = _require_private_file_fd(fd, "FAIL_CLOSED_AUTHORIZATION_FILE_REQUIREMENTS")
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                _fail("FAIL_CLOSED_AUTHORIZATION_WRITE")
            offset += written
        os.fsync(fd)
        after_write = _require_private_file_fd(fd, "FAIL_CLOSED_AUTHORIZATION_FILE_REQUIREMENTS")
        if _identity(created) != _identity(after_write):
            _fail("FAIL_CLOSED_AUTHORIZATION_FILE_REQUIREMENTS")
    except OSError as exc:
        raise ControlPlaneError("FAIL_CLOSED_AUTHORIZATION_WRITE") from exc
    finally:
        _checked_close(fd, "FAIL_CLOSED_AUTHORIZATION_CLOSE")
    try:
        os.fsync(auth_dir_fd)
    except OSError as exc:
        raise ControlPlaneError("FAIL_CLOSED_AUTHORIZATION_DIRECTORY_FSYNC") from exc
    readback = _read_private_regular_at(
        auth_dir_fd,
        leaf,
        _identity(after_write),
        "FAIL_CLOSED_AUTHORIZATION_READBACK",
    )
    if readback != raw:
        _fail("FAIL_CLOSED_AUTHORIZATION_READBACK")
    return _sha256(raw)


def prepare_ledger() -> dict[str, Any]:
    """Create only the fixed local-ext4 root and three children; no auth/seed is involved."""
    _require_linux_uid_1001()
    _validate_control_plane_contract()
    fds: list[int] = []
    try:
        parent_fd = _open_fixed_private_parent()
        fds.append(parent_fd)
        root_fd = _ensure_private_child_dir(parent_fd, LEDGER_LEAF, "FAIL_CLOSED_PREPARE_LEDGER_ROOT")
        fds.append(root_fd)
        auth_fd = _ensure_private_child_dir(root_fd, AUTHORIZATION_DIR, "FAIL_CLOSED_PREPARE_AUTH_DIR")
        fds.append(auth_fd)
        reservation_fd = _ensure_private_child_dir(root_fd, RESERVATIONS_DIR, "FAIL_CLOSED_PREPARE_RESERVATIONS_DIR")
        fds.append(reservation_fd)
        outcomes_fd = _ensure_private_child_dir(root_fd, OUTCOMES_DIR, "FAIL_CLOSED_PREPARE_OUTCOMES_DIR")
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


def issue_authorization(external_approval_id: str, ttl_seconds: int, intended_output_root: str) -> dict[str, Any]:
    """Issue exactly one v6-compatible local authorization. No secret reaches stdout/stderr/NAS."""
    _require_linux_uid_1001()
    _validate_control_plane_contract()
    external_approval_id = _validate_approval_id(external_approval_id)
    ttl_seconds = _validate_ttl(ttl_seconds)
    intended_output_root = _validate_intended_output_root(intended_output_root)
    selector_contract_sha256 = _selector_functional_contract_sha256()
    _verify_frozen_runner_v11()

    fds: list[int] = []
    try:
        parent_fd, root_fd, auth_fd, reservation_fd, outcomes_fd = _open_existing_ledger_layout()
        fds.extend((parent_fd, root_fd, auth_fd, reservation_fd, outcomes_fd))
        # A global fixed-D0 slot: a prior authorization, reservation, outcome, or unknown
        # control leaf makes this a permanent cooperative workflow NO-GO, never a retry.
        _require_empty_private_dir(auth_fd, "FAIL_CLOSED_AUTHORIZATION_SLOT_ALREADY_USED")
        _require_empty_private_dir(reservation_fd, "FAIL_CLOSED_RESERVATION_SLOT_ALREADY_USED")
        _require_empty_private_dir(outcomes_fd, "FAIL_CLOSED_OUTCOME_SLOT_ALREADY_USED")

        issued = int(time.time())
        if issued < 0:
            _fail("FAIL_CLOSED_REALTIME_CLOCK")
        expires = issued + ttl_seconds
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
            "selector_source_sha256": SELECTOR_V6_SOURCE_SHA256,
            "selector_contract_sha256": selector_contract_sha256,
            "selector_contract_document_sha256": SELECTOR_V6_CONTRACT_DOCUMENT_SHA256,
            "preflight_anchor": PINNED_PREFLIGHT_ANCHOR,
            "intended_preflight_root": PINNED_PREFLIGHT_ROOT,
            "intended_output_root": intended_output_root,
            "coverage_binding_protocol": COVERAGE_BINDING_PROTOCOL,
            "coverage_predicate_bundle_sha256": RUNNER_V11_PREDICATE_SHA256,
            "coverage_runner_source_sha256": RUNNER_V11_SOURCE_SHA256,
            "coverage_runner_contract_sha256": RUNNER_V11_CONTRACT_SHA256,
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
        raw = _canonical_json(document)
        authorization_sha256 = _write_authorization_exclusive(auth_fd, authorization_id, raw)

        # Close local descriptors before a safe public result. Any close failure yields no
        # PASS; the nonempty authorization directory prevents a silent seed replacement.
        for fd in reversed(fds):
            _checked_close(fd, "FAIL_CLOSED_ISSUER_CLOSE")
        fds.clear()
        return {
            "status": "LOCAL_EXT4_V6_AUTHORIZATION_ISSUED_ONE_SHOT_DEVELOPER_ONLY",
            "scope": "developer_only_known_history_limited_unsealed_not_heldout_not_final; local O_EXCL workflow protection only",
            "authorization_id": authorization_id,
            "authorization_path": AUTHORIZATION_LEDGER_ROOT + "/authorizations/" + authorization_id + ".json",
            "authorization_file_sha256": authorization_sha256,
            "intended_output_root": intended_output_root,
            "not_before_unix_seconds": issued,
            "expires_unix_seconds": expires,
            "selector_source_sha256": SELECTOR_V6_SOURCE_SHA256,
            "selector_contract_sha256": selector_contract_sha256,
            "selector_contract_document_sha256": SELECTOR_V6_CONTRACT_DOCUMENT_SHA256,
            "runner_source_sha256": RUNNER_V11_SOURCE_SHA256,
            "runner_contract_sha256": RUNNER_V11_CONTRACT_SHA256,
            "predicate_bundle_sha256": RUNNER_V11_PREDICATE_SHA256,
        }
    except Exception:
        _best_effort_close(fds)
        raise


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-ledger", help="Create only the fixed private ext4 ledger directories.")
    issue = commands.add_parser("issue-authorization", help="Create one local selector-v6 authorization; no CLI seed exists.")
    issue.add_argument("--external-approval-id", required=True)
    issue.add_argument("--ttl-seconds", required=True, type=int)
    issue.add_argument("--intended-output-root", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-ledger":
            result = prepare_ledger()
        elif args.command == "issue-authorization":
            result = issue_authorization(args.external_approval_id, args.ttl_seconds, args.intended_output_root)
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
