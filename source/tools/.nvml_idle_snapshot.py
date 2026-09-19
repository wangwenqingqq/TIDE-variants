#!/usr/bin/python3
# Root-private, read-only NVML snapshot helper.  It is deliberately not a
# launcher, token issuer, lease manager, or CUDA client.
import ctypes
import hashlib
import os
import re
import stat
import sys

SCHEMA = "safe-c1-g3-nvml-snapshot-v2"
PYTHON_REALPATH = "/usr/bin/python3.12"
PYTHON_SHA256 = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
NVML_LIBRARY_LINK = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
NVML_LIBRARY_REALPATH = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.590.48.01"
NVML_LIBRARY_SHA256 = "12f3bcd4ba447599a2077297e3a4ff4288205b082c2e76346718ae85c058c4b2"
NVML_SUCCESS = 0
NVML_ERROR_INSUFFICIENT_SIZE = 7
UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
PCI_RE = re.compile(r"^[0-9a-fA-F]{8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
DRIVER_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}$")

class SnapshotFailure(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)

class NvmlPciInfoV3(ctypes.Structure):
    _fields_ = [
        ("busIdLegacy", ctypes.c_char * 16),
        ("domain", ctypes.c_uint),
        ("bus", ctypes.c_uint),
        ("device", ctypes.c_uint),
        ("pciDeviceId", ctypes.c_uint),
        ("pciSubSystemId", ctypes.c_uint),
        ("busId", ctypes.c_char * 32),
    ]

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)

def require_trusted_regular(path, expected_realpath, expected_sha256, code):
    actual = os.path.realpath(path)
    if actual != expected_realpath:
        raise SnapshotFailure(code)
    try:
        info = os.stat(actual)
    except OSError:
        raise SnapshotFailure(code)
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_gid != 0 or
            (stat.S_IMODE(info.st_mode) & 0o022) != 0):
        raise SnapshotFailure(code)
    if sha256_file(actual) != expected_sha256:
        raise SnapshotFailure(code)

def c_text(buffer, code):
    raw = bytes(buffer)
    if b"\0" not in raw:
        raise SnapshotFailure(code)
    try:
        value = raw.split(b"\0", 1)[0].decode("ascii", "strict")
    except UnicodeDecodeError:
        raise SnapshotFailure(code)
    if not value or any(ch.isspace() for ch in value):
        raise SnapshotFailure(code)
    return value

def bind(lib, name, argtypes):
    try:
        function = getattr(lib, name)
    except AttributeError:
        raise SnapshotFailure("E_NVML_SYMBOL")
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function

def require_success(code, status):
    if status != NVML_SUCCESS:
        raise SnapshotFailure(code)

def parse_ordinal(argv):
    if len(argv) != 3 or argv[1] != "--gpu-ordinal" or not re.fullmatch(r"(?:0|[1-9][0-9]*)", argv[2]):
        raise SnapshotFailure("E_ARGUMENT")
    ordinal = int(argv[2])
    if ordinal > 0xffffffff:
        raise SnapshotFailure("E_ARGUMENT")
    return ordinal

def snapshot(ordinal):
    if ctypes.sizeof(NvmlPciInfoV3) != 68 or NvmlPciInfoV3.domain.offset != 16 or NvmlPciInfoV3.busId.offset != 36:
        raise SnapshotFailure("E_NVML_ABI")
    require_trusted_regular(sys.executable, PYTHON_REALPATH, PYTHON_SHA256, "E_PYTHON_FINGERPRINT")
    require_trusted_regular(NVML_LIBRARY_LINK, NVML_LIBRARY_REALPATH, NVML_LIBRARY_SHA256, "E_NVML_LIBRARY_FINGERPRINT")
    try:
        lib = ctypes.CDLL(NVML_LIBRARY_REALPATH)
    except OSError:
        raise SnapshotFailure("E_NVML_LOAD")
    init = bind(lib, "nvmlInit_v2", [])
    shutdown = bind(lib, "nvmlShutdown", [])
    get_handle = bind(lib, "nvmlDeviceGetHandleByIndex_v2", [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)])
    get_uuid = bind(lib, "nvmlDeviceGetUUID", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint])
    get_pci = bind(lib, "nvmlDeviceGetPciInfo_v3", [ctypes.c_void_p, ctypes.POINTER(NvmlPciInfoV3)])
    get_compute_processes = bind(lib, "nvmlDeviceGetComputeRunningProcesses_v3", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p])
    get_graphics_processes = bind(lib, "nvmlDeviceGetGraphicsRunningProcesses_v3", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p])
    get_driver = bind(lib, "nvmlSystemGetDriverVersion", [ctypes.POINTER(ctypes.c_char), ctypes.c_uint])
    initialized = False
    try:
        require_success("E_NVML_INIT", init())
        initialized = True
        handle = ctypes.c_void_p()
        require_success("E_NVML_ORDINAL", get_handle(ctypes.c_uint(ordinal), ctypes.byref(handle)))
        uuid_buffer = ctypes.create_string_buffer(96)
        require_success("E_NVML_UUID", get_uuid(handle, uuid_buffer, ctypes.c_uint(len(uuid_buffer))))
        uuid = c_text(uuid_buffer, "E_NVML_UUID")
        if not UUID_RE.fullmatch(uuid):
            raise SnapshotFailure("E_NVML_UUID")
        pci = NvmlPciInfoV3()
        require_success("E_NVML_PCI", get_pci(handle, ctypes.byref(pci)))
        pci_bus_id = c_text(ctypes.string_at(
            ctypes.addressof(pci) + NvmlPciInfoV3.busId.offset, 32
        ), "E_NVML_PCI")
        if not PCI_RE.fullmatch(pci_bus_id):
            raise SnapshotFailure("E_NVML_PCI")
        expected_pci_prefix = f"{pci.domain:08x}:{pci.bus:02x}:{pci.device:02x}"
        if not pci_bus_id.lower().startswith(expected_pci_prefix + "."):
            raise SnapshotFailure("E_NVML_PCI")
        def process_count_or_fail(function):
            count = ctypes.c_uint(0)
            status = function(handle, ctypes.byref(count), None)
            if status == NVML_ERROR_INSUFFICIENT_SIZE or count.value != 0:
                raise SnapshotFailure("E_GPU_BUSY")
            if status != NVML_SUCCESS:
                raise SnapshotFailure("E_NVML_PROCESS_QUERY")
            return count.value
        compute_process_count = process_count_or_fail(get_compute_processes)
        graphics_process_count = process_count_or_fail(get_graphics_processes)
        driver_buffer = ctypes.create_string_buffer(96)
        require_success("E_NVML_DRIVER", get_driver(driver_buffer, ctypes.c_uint(len(driver_buffer))))
        driver_version = c_text(driver_buffer, "E_NVML_DRIVER")
        if not DRIVER_RE.fullmatch(driver_version):
            raise SnapshotFailure("E_NVML_DRIVER")
        require_success("E_NVML_SHUTDOWN", shutdown())
        initialized = False
        return {
            "schema": SCHEMA,
            "gpu_ordinal": str(ordinal),
            "gpu_uuid": uuid,
            "gpu_pci_bus_id": pci_bus_id,
            "compute_process_count": str(compute_process_count),
            "graphics_process_count": str(graphics_process_count),
            "python_realpath": PYTHON_REALPATH,
            "python_sha256": PYTHON_SHA256,
            "nvml_library_realpath": NVML_LIBRARY_REALPATH,
            "nvml_library_sha256": NVML_LIBRARY_SHA256,
            "nvml_driver_version": driver_version,
        }
    finally:
        if initialized:
            shutdown()

def main(argv):
    try:
        ordinal = parse_ordinal(argv)
        values = snapshot(ordinal)
        order = ("schema", "gpu_ordinal", "gpu_uuid", "gpu_pci_bus_id", "compute_process_count", "graphics_process_count", "python_realpath", "python_sha256", "nvml_library_realpath", "nvml_library_sha256", "nvml_driver_version")
        sys.stdout.write("\n".join(key + "=" + values[key] for key in order) + "\n")
        return 0
    except SnapshotFailure as error:
        sys.stderr.write("safe-c1-nvml-snapshot: " + error.code + "\n")
        return 69
    except Exception:
        sys.stderr.write("safe-c1-nvml-snapshot: E_INTERNAL\n")
        return 69

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
