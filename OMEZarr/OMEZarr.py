"""OME-Zarr (OME-NGFF) support for 3D Slicer.

Registers:

* ``OMEZarrFileReader`` - scripted file reader (Add Data, drag-and-drop,
  ``slicer.util.loadNodeFromFile``), for local stores, ``.ozx`` files and URLs.
* ``OMEZarrFileWriter`` - scripted file writer for scalar and label map volumes.
* ``OMEZarrFileDialog`` - drop target for OME-Zarr directories.
* ``OMEZarrLogic`` - level selection, NGFF -> RAS geometry, chunk-limited reads,
  labels, time series, view-driven refinement.
* ``Streamer`` - shows a large store at once from its coarsest level, reads what the
  slice views show at screen resolution first and the rest of the chosen level behind it.

NGFF parsing, multiscales, store access and RFC-4 orientation come from ngff-zarr.
"""

import collections
import functools
import heapq
import itertools
import json
import logging
import os
import queue
import re
import threading
import time
import urllib.request

import numpy as np
import qt
import vtk

import slicer
import slicer.packaging
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

SPATIAL_DIMS = ("x", "y", "z")

# Slicer world coordinates are millimetres. NGFF axis units are UDUNITS-2 names.
LENGTH_UNIT_TO_MM = {
    None: 1.0,
    "": 1.0,
    "millimeter": 1.0,
    "micrometer": 1e-3,
    "micron": 1e-3,
    "nanometer": 1e-6,
    "picometer": 1e-9,
    "angstrom": 1e-7,
    "centimeter": 10.0,
    "decimeter": 100.0,
    "meter": 1000.0,
    "kilometer": 1e6,
    "inch": 25.4,
    "foot": 304.8,
    "yard": 914.4,
}

# Symbols shown in the panel, and used when the display follows the store's unit.
UNIT_SYMBOLS = {"micrometer": "µm", "micron": "µm", "nanometer": "nm", "millimeter": "mm", "centimeter": "cm"}

# Shown in the module until another store has been used: a public 36 GiB (uint16, 6.1 µm, 4 levels)
# microCT of a dry skull on JS2, for testing streaming of a large store on a new system.
SAMPLE_STORE_URL = (
    "https://js2.jetstream-cloud.org:8001/swift/v1/MorphoDepot-volumes/large-sample-data/dry_skull_4K_12.5micron.ome.zarr"
)

DEFAULT_BUDGET_FRACTION = 0.25  # of available RAM when the budget setting is "auto"
FALLBACK_MAX_BYTES = 1 << 30

# Streaming: chunks of levels other than the one being filled are kept in a bounded cache;
# a view whose visible plane needs more chunks than the limit is shown one level coarser.
STREAM_CACHE_BYTES = 512 << 20
STREAM_MAX_CHUNKS_PER_VIEW = 512
STREAM_READERS = 16  # measured on JS2: 16 fills neotoma level 0 in 18 s; 32 is barely faster and slows the tail
# Slice views come first: while one waits for a chunk, no 3D or background read starts, and those
# reads never hold more than this many readers, so a view's chunks start at once on a busy link.
STREAM_BACKGROUND_READERS = STREAM_READERS // 2
STREAM_PARTIAL_S = 0.25  # a view's plane is redrawn this often while its chunks arrive
# Remote reads: requests to one server at a time start at STREAM_READERS and halve while it answers busy.
HTTP_ATTEMPTS = 8  # per request, while the server answers busy or drops the connection
HTTP_MAX_WAIT_S = 60.0  # longest pause asked by Retry-After that is honored as such
# Requests one Slicer sends one server within a sliding window. JS2's object store answers 429 past
# 1000 requests in 10 s from one client IP, counting every GET/HEAD/PUT (2026-10-01): stay under it.
HTTP_WINDOW_REQUESTS = 900
HTTP_WINDOW_S = 10.0
DISK_CACHE_MIB = 10240  # default size of the on-disk cache of remote chunks
# Streamed volume rendering: one texture for what the 3D view shows. These are the fallbacks when
# the GPU's limits cannot be read (macOS reports no video memory; Apple GPUs allow 2048 per side).
STREAM_3D_MAX_BYTES = 2 << 30
STREAM_3D_MAX_DIM = 2048
STREAM_3D_SETTLE_MS = 300  # the texture follows the camera once it has been still this long
# The 3D view reads, in depth, from the nearest voxel its opacity shows to where the opacity
# accumulated along the view's rays reaches STREAM_3D_OPAQUE: what lies behind is not seen, so it
# is not read. An opaque rendering stops just behind the surface; a transparent one reads deeper.
# The rays (STREAM_3D_RAYS per side) are marched through the coarsest level, always in memory.
STREAM_3D_OPAQUE = 0.95
STREAM_3D_VISIBLE_OPACITY = 0.01  # opacity from which a voxel counts as seen
STREAM_3D_RAYS = 32

# Slicer core lookup tables used to colour separate microscopy channels.
CHANNEL_COLOR_NODE_IDS = {
    "red": "vtkMRMLColorTableNodeRed",
    "green": "vtkMRMLColorTableNodeGreen",
    "blue": "vtkMRMLColorTableNodeBlue",
    "yellow": "vtkMRMLColorTableNodeYellow",
    "cyan": "vtkMRMLColorTableNodeCyan",
    "magenta": "vtkMRMLColorTableNodeMagenta",
    "grey": "vtkMRMLColorTableNodeGrey",
}

MAX_COLOR_TABLE_ENTRIES = 65536


class Settings:
    """QSettings keys. Values are read on each use so the module panel takes effect immediately."""

    MAX_BYTES = "OMEZarr/MaxBytes"  # int bytes, 0 = automatic
    ORIENTATION = "OMEZarr/AssumedOrientation"  # LPS or RAS, used when a store has no RFC-4 orientation
    LOAD_LABELS = "OMEZarr/LoadLabels"
    TIME_MODE = "OMEZarr/TimeMode"  # sequence or index
    DISPLAY_UNITS = "OMEZarr/DisplayUnits"  # switch Slicer's length display unit to the store's
    AUTO_REFINE = "OMEZarr/AutoRefine"  # refine the slice views automatically after a store is loaded
    LABELS_AS_SEGMENTATION = "OMEZarr/LabelsAsSegmentation"  # load labels as Segmentation nodes
    STORAGE_OPTIONS = "OMEZarr/StorageOptions"  # JSON passed to ngff-zarr for remote stores
    DETECT_LABEL_MAPS = "OMEZarr/DetectLabelMaps"  # integer stores with few values load as label maps
    STREAM = "OMEZarr/Stream"  # show the coarsest level at once and stream the chosen level behind it
    STREAM_3D = "OMEZarr/StreamVolumeRendering"  # volume-render streamed stores in the 3D view
    STREAM_3D_MEMORY = "OMEZarr/StreamVolumeRenderingMemory"  # MiB a 3D texture may use, 0 = automatic
    DISK_CACHE = "OMEZarr/DiskCacheMiB"  # MiB of remote chunks kept on disk between sessions, 0 = off
    DISK_CACHE_DIR = "OMEZarr/DiskCacheDirectory"  # empty = <Slicer cache>/OMEZarr

    @staticmethod
    def get(key, default):
        value = qt.QSettings().value(key)
        if value is None or value == "":
            return default
        if isinstance(default, bool):
            return str(value).lower() in ("true", "1", "yes")
        if isinstance(default, int):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
        return str(value)

    @staticmethod
    def set(key, value):
        qt.QSettings().setValue(key, value)


#
# Module
#


class OMEZarr(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("OME-Zarr")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Informatics")]
        self.parent.dependencies = ["Sequences"]
        self.parent.contributors = ["Valentin Boussot (Fideus Labs)", "Matt McCormick (Fideus Labs)"]
        self.parent.helpText = _(
            "Read and write OME-Zarr (OME-NGFF) images. Drag an .ome.zarr directory into Slicer, or use "
            "File > Add Data. The finest resolution level fitting the memory budget is loaded; "
            "labels and time series are loaded alongside. Use 'Refine current view' to reload what a "
            "slice view shows at a finer level."
        )
        self.parent.acknowledgementText = _("Built on ngff-zarr (https://github.com/fideus-labs/ngff-zarr).")
        # Streaming readers must be idle before Python is finalized, or a decoder thread still
        # returning a chunk crashes the application on exit.
        slicer.app.connect("aboutToQuit()", OMEZarrLogic.onAboutToQuit)


#
# Store discovery helpers
#


def isRemoteUrl(path):
    return bool(re.match(r"^(https?|s3|gs|gcs|az|abfs)://", str(path)))


def normalizeStorePath(path):
    """Canonical form of a store path for node attributes and comparisons (URLs unchanged)."""
    path = str(path)
    if isRemoteUrl(path):
        return path.rstrip("/")
    return os.path.normpath(os.path.abspath(path))


def samePath(a, b):
    return a is not None and b is not None and normalizeStorePath(a) == normalizeStorePath(b)


def joinStorePath(root, *parts):
    root = str(root).rstrip("/")
    return "/".join([root, *parts]) if isRemoteUrl(root) else os.path.join(root, *parts)


def readStoreAttributes(root):
    """Merged root attributes of a Zarr group (v2 ``.zattrs`` or v3 ``zarr.json``), OME keys unwrapped.

    Returns None when the group has no readable metadata. Local paths and http(s) URLs only.
    """
    root = str(root).rstrip("/")
    for name in ("zarr.json", ".zattrs"):
        target = joinStorePath(root, name)
        try:
            if isRemoteUrl(target):
                if not target.startswith("http"):
                    return None
                with urllib.request.urlopen(target, timeout=20) as response:  # noqa: S310 - user-provided URL
                    attrs = json.loads(response.read().decode("utf-8"))
            else:
                if not os.path.isfile(target):
                    continue
                with open(target, encoding="utf-8") as fp:
                    attrs = json.load(fp)
        except (OSError, ValueError):
            continue
        if not isinstance(attrs, dict):
            continue
        if name == "zarr.json":
            attrs = attrs.get("attributes", {})
        merged = dict(attrs)
        if isinstance(attrs.get("ome"), dict):
            merged.update(attrs["ome"])
        return merged
    return None


def omeZarrRootFromPath(path):
    """Return the OME-Zarr multiscales root for a path, or None.

    Accepts the store directory itself, one of its metadata files
    (``zarr.json``/``.zattrs``, which is what ``Add Data`` lists when a directory
    is added), a ``.ozx``/``.zip`` file, or a remote URL.
    """
    if not path:
        return None
    path = str(path)
    if isRemoteUrl(path):
        return path.rstrip("/") if re.search(r"\.zarr/?$", path) else None
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if os.path.basename(path) in ("zarr.json", ".zattrs", ".zgroup"):
            path = os.path.dirname(path)
        elif path.lower().endswith((".ozx", ".zarr.zip")):
            return path
        else:
            return None
    if not os.path.isdir(path):
        return None
    attrs = readStoreAttributes(path)
    return path if attrs and ("multiscales" in attrs or "bioformats2raw.layout" in attrs) else None


def isBioformats2rawRoot(root):
    attrs = readStoreAttributes(root)
    return bool(attrs and "bioformats2raw.layout" in attrs and "multiscales" not in attrs)


def bioformats2rawSeries(root, limit=1000):
    """Paths of the image series of a bioformats2raw container: ``<root>/0``, ``<root>/1``, ..."""
    series = []
    for index in range(limit):
        candidate = joinStorePath(root, str(index))
        attrs = readStoreAttributes(candidate)
        if not attrs or "multiscales" not in attrs:
            break
        series.append(candidate)
    return series


def isLabelStore(root):
    attrs = readStoreAttributes(root)
    return bool(attrs and "image-label" in attrs)


def labelGroupNames(root):
    """Names listed in the ``labels`` group of an image store (empty when absent)."""
    attrs = readStoreAttributes(joinStorePath(root, "labels"))
    names = attrs.get("labels") if attrs else None
    return [str(n) for n in names] if isinstance(names, list) else []


def scalarVolumes(nodes):
    return [n for n in nodes if n.IsA("vtkMRMLScalarVolumeNode") and not n.IsA("vtkMRMLLabelMapVolumeNode")]


def labelMaps(nodes):
    return [n for n in nodes if n.IsA("vtkMRMLLabelMapVolumeNode")]


def runResponsive(work, onTick=None):
    """Run ``work()`` in a worker thread while this (GUI) thread keeps processing Qt events.

    ``onTick`` is called from the GUI thread about a hundred times a second. An exception
    raised by ``work`` is re-raised here; its return value is returned. Called from another
    thread, it just runs ``work``.
    """
    if threading.current_thread() is not threading.main_thread():
        return work()
    outcome = {}

    def target():
        try:
            outcome["result"] = work()
        except BaseException as e:  # noqa: BLE001 - re-raised in the calling thread
            outcome["error"] = e

    thread = threading.Thread(target=target, name="OMEZarr", daemon=True)
    thread.start()
    while thread.is_alive():
        slicer.app.processEvents()
        if onTick:
            onTick()
        time.sleep(0.01)
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def removesNodesWhenCancelled(function):
    """Remove the nodes ``function`` added to the scene when it is cancelled (InterruptedError)."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        before = {node.GetID() for node in slicer.util.getNodesByClass("vtkMRMLNode")}
        try:
            return function(*args, **kwargs)
        except InterruptedError:
            for node in slicer.util.getNodesByClass("vtkMRMLNode"):
                if node.GetID() not in before and node.GetScene() is not None:  # display nodes go with their volume
                    slicer.mrmlScene.RemoveNode(node)
            raise

    return wrapper


#
# Progress reporting
#


class Progress:
    """Progress dialog with a cancel button when the GUI is up; silent otherwise.

    Used as ``progress(done, total, text)``; returns False once the user cancelled. While Slicer's
    IO manager loads ``fileName``, its own dialog is driven instead of opening a second one.
    """

    def __init__(self, label, fileName=None):
        self.label = label
        self.fileName = fileName
        self.dialog = None
        self.owned = False
        self.cancelled = False

    def __enter__(self):
        if slicer.util.mainWindow() and not slicer.app.testingEnabled():
            self.dialog = self.ioManagerDialog()
            if self.dialog is None:
                self.dialog = slicer.util.createProgressDialog(
                    labelText=self.label, windowTitle=_("OME-Zarr"), maximum=100
                )
                self.owned = True
            else:
                self.dialog.setCancelButtonText(_("Cancel"))  # Slicer leaves it out for a single file
            # Only a click on Cancel cancels. QProgressDialog.wasCanceled also turns true without one:
            # the IO manager's parentless, window-modal dialog sometimes emits canceled() as it forces
            # itself on screen after its minimum duration (seen on macOS), which aborted good loads.
            for button in self.dialog.findChildren(qt.QPushButton):
                button.connect("clicked()", self.onCancelClicked)
        return self

    def onCancelClicked(self):
        self.cancelled = True

    def ioManagerDialog(self):
        """The dialog the IO manager opens around a load, labelled with the file name."""
        if not self.fileName:
            return None
        return next(
            (
                widget
                for widget in slicer.app.topLevelWidgets()
                if isinstance(widget, qt.QProgressDialog) and self.fileName in widget.labelText
            ),
            None,
        )

    def __call__(self, done, total, text=None):
        if self.dialog is None:
            return True
        if self.dialog.maximum == 100:  # with several files, the IO manager's bar counts files
            self.dialog.value = min(99, int(100 * done / max(1, total)))  # 100 resets and hides the dialog
        if text:
            self.dialog.labelText = text
        return not self.cancelled

    def __exit__(self, *args):
        if self.owned:
            self.dialog.close()
        return False


#
# Logic
#


class OMEZarrLogic(ScriptedLoadableModuleLogic):
    """NGFF <-> MRML mapping. Stateless apart from a small multiscales cache."""

    _multiscalesCache = {}

    @staticmethod
    def ensureNgffZarr():
        # From 0.46.1 on, ngff-zarr fills NgffImage.axes_orientations from the RFC-4 metadata on read.
        # 0.47.0 writes OME-Zarr 0.6 (for oblique volumes) tagged "0.6"; 0.46.1 tags it "0.6rc0".
        requirement = "ngff-zarr[remote]>=0.47.0"
        if not slicer.packaging.pip_check(requirement):
            interactive = slicer.util.mainWindow() and not slicer.app.testingEnabled()
            if interactive and not slicer.util.confirmOkCancelDisplay(
                _("ngff-zarr 0.47.0 or newer is required to read OME-Zarr images. Install it now?")
            ):
                raise RuntimeError("ngff-zarr 0.47.0 or newer is not installed")
            with slicer.util.tryWithErrorDisplay(_("Failed to install ngff-zarr"), waitCursor=True):
                slicer.util.pip_install(requirement)
        import ngff_zarr

        return ngff_zarr

    @staticmethod
    def storageOptions(path):
        """Options for a remote store: the configured JSON, and anonymous S3 access unless
        credentials are given in the environment or in that JSON.

        obstore only reads credentials from the environment, not from ``~/.aws``; without
        ``anon`` it asks the EC2 metadata service for half a minute and then fails on a
        public bucket."""
        if not isRemoteUrl(path):
            return None
        options = {}
        configured = Settings.get(Settings.STORAGE_OPTIONS, "")
        if configured:
            try:
                options.update(json.loads(configured))
            except ValueError:
                logging.warning("Ignoring invalid JSON in the OME-Zarr storage options setting")
        if str(path).startswith("s3://") and "anon" not in options and "skip_signature" not in options:
            variables = ("AWS_ACCESS_KEY_ID", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            keys = ("access_key_id", "aws_access_key_id", "secret_access_key", "token")
            hasCredentials = any(os.environ.get(name) for name in variables) or any(key in options for key in keys)
            if not hasCredentials:
                options["anon"] = True
        return options or None

    @classmethod
    def openMultiscales(cls, path, useCache=True):
        """Open a store with ngff-zarr; arrays stay lazy (dask), nothing is read yet."""
        ngff_zarr = cls.ensureNgffZarr()
        key = str(path)
        if useCache and key in cls._multiscalesCache:
            return cls._multiscalesCache[key]
        source = key
        if isBioformats2rawRoot(key):
            series = bioformats2rawSeries(key)
            if not series:
                raise ValueError(f"No image series found in the bioformats2raw container {key}")
            source = series[0]
        options = cls.storageOptions(source)
        multiscales = (
            ngff_zarr.from_ome_zarr(source, storage_options=options) if options else ngff_zarr.from_ome_zarr(source)
        )
        cls.attachAffine(multiscales)
        multiscales.omezarrSource = source  # where its levels' chunk readers read from
        if useCache:
            cls._multiscalesCache[key] = multiscales
        return multiscales

    @classmethod
    def clearCache(cls):
        cls._multiscalesCache.clear()
        cls._chunkReaders.clear()

    _chunkReaders = {}

    @classmethod
    def chunkReader(cls, multiscales, level):
        """The ChunkReader of a level of an http(s) store, or None (other stores, other layouts).
        Kept for the session, so each shard's index is read once whoever reads the level. Asks
        the server for the level's metadata the first time: call it off the GUI thread first."""
        source = getattr(multiscales, "omezarrSource", None)
        datasets = getattr(multiscales.metadata, "datasets", None) or []
        if not source or not str(source).startswith(("http://", "https://")) or not 0 <= level < len(datasets):
            return None
        if list(multiscales.images[level].dims) != ["z", "y", "x"]:
            return None
        key = (normalizeStorePath(source), level)
        if key not in cls._chunkReaders:
            cls._chunkReaders[key] = ChunkReader.forArray(source, datasets[level].path, DiskCache.instance())
        return cls._chunkReaders[key]

    @staticmethod
    def readChunks(grid, region, output, progress=None, label=""):
        """Read ``region`` ((start, stop) per z, y, x) of a LevelChunks into ``output``, chunk by chunk
        with STREAM_READERS parallel reads. From the GUI thread it keeps the application responsive
        (see computeArray); cancelling raises InterruptedError."""
        from concurrent.futures import ThreadPoolExecutor

        keys = grid.keys(region)
        state = {"done": 0, "cancel": False}
        lock = threading.Lock()

        def readOne(key):
            if state["cancel"]:
                return
            block = grid.read(key)
            target, source = [], []
            for (regionStart, regionStop), (chunkStart, chunkStop) in zip(region, grid.bounds(key)):
                start, stop = max(regionStart, chunkStart), min(regionStop, chunkStop)
                target.append(slice(start - regionStart, stop - regionStart))
                source.append(slice(start - chunkStart, stop - chunkStart))
            output[tuple(target)] = block[tuple(source)]
            with lock:
                state["done"] += 1

        def work():
            with ThreadPoolExecutor(STREAM_READERS, thread_name_prefix="OMEZarr") as pool:
                futures = [pool.submit(readOne, key) for key in keys]
                try:
                    for future in futures:
                        future.result()
                except BaseException:
                    state["cancel"] = True  # the other reads stop where they are
                    raise

        def onTick():
            if progress and not progress(state["done"], len(keys), label):
                state["cancel"] = True

        runResponsive(work, onTick)
        if state["cancel"]:
            raise InterruptedError("Loading cancelled")

    @classmethod
    def readLevel(cls, multiscales, level, dtype=None):
        """A whole level of a (z, y, x) store as a numpy array (by its chunk reader when it has one)."""
        level %= len(multiscales.images)
        image = multiscales.images[level]
        reader = cls.chunkReader(multiscales, level)
        if reader is not None:
            grid = LevelChunks(image, 0, 0, reader)
            output = np.empty(grid.shape, dtype or reader.fill.dtype)
            cls.readChunks(grid, tuple((0, n) for n in grid.shape), output)
            return output
        sub, _addZ = cls.spatialDaskArray(image)
        if dtype:
            sub = sub.astype(dtype)
        output = np.empty(sub.shape, sub.dtype)
        cls.computeArray(sub, output)
        return output

    # ---- metadata ----

    @staticmethod
    def spatialShape(image):
        dims = list(image.dims)
        return {d: image.data.shape[dims.index(d)] for d in SPATIAL_DIMS if d in dims}

    @staticmethod
    def axisLength(image, dim):
        dims = list(image.dims)
        return int(image.data.shape[dims.index(dim)]) if dim in dims else 1

    @classmethod
    def volumeBytes(cls, image, region=None):
        """Bytes of one spatial volume (single channel, single time point), or of ``region`` within it."""
        size = 1
        for d, n in cls.spatialShape(image).items():
            if region and d in region:
                n = region[d][1] - region[d][0]
            size *= int(n)
        return size * int(np.dtype(image.data.dtype).itemsize)

    @classmethod
    def levelInfo(cls, multiscales):
        info = []
        base = multiscales.images[0]
        for index, image in enumerate(multiscales.images):
            downsample = {d: image.scale.get(d, 1.0) / base.scale.get(d, 1.0) for d in SPATIAL_DIMS if d in image.dims}
            info.append(
                {
                    "level": index,
                    "dims": tuple(image.dims),
                    "shape": tuple(int(n) for n in image.data.shape),
                    "chunks": tuple(int(n) for n in image.data.chunksize),
                    "dtype": str(image.data.dtype),
                    "bytes": cls.volumeBytes(image),
                    "scale": dict(image.scale),
                    "units": dict(image.axes_units or {}),
                    "downsample": downsample,
                }
            )
        return info

    @classmethod
    def selectLevel(cls, multiscales, maxBytes, copies=1):
        """Finest level whose volume (times ``copies``: channels, time points) fits the budget."""
        for index, image in enumerate(multiscales.images):
            if cls.volumeBytes(image) * copies <= maxBytes:
                return index
        return len(multiscales.images) - 1

    @staticmethod
    def availableMemory():
        """Bytes of RAM available to load data, or None when it cannot be read.

        On macOS this is the kernel's own free-memory level (what Activity Monitor and
        ``memory_pressure`` report) times the installed RAM. psutil multiplies the kernel's page
        counts by the page size the process sees, which is 4 KiB in an x86_64 Slicer running under
        Rosetta while Apple silicon pages are 16 KiB: there it reported 4 GiB of 36 available when
        macOS reported 76% free."""
        import sys

        if sys.platform == "darwin":
            try:
                import ctypes
                import ctypes.util

                libc = ctypes.CDLL(ctypes.util.find_library("c"))

                def sysctl(name, ctype):
                    value = ctype(0)
                    size = ctypes.c_size_t(ctypes.sizeof(value))
                    if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
                        return None
                    return value.value

                level, total = sysctl("kern.memorystatus_level", ctypes.c_int), sysctl("hw.memsize", ctypes.c_uint64)
                if level and total:
                    return int(total * min(100, max(0, level)) / 100)
            except Exception:  # noqa: BLE001 - fall back to psutil
                logging.debug("OME-Zarr: could not read the macOS memory level", exc_info=True)
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001 - psutil is optional
            return None

    @classmethod
    def maxBytesFromSettings(cls):
        """Configured budget, or a fraction of the available RAM when set to automatic."""
        value = Settings.get(Settings.MAX_BYTES, 0)
        if value > 0:
            return value
        available = cls.availableMemory()
        return int(available * DEFAULT_BUDGET_FRACTION) if available else FALLBACK_MAX_BYTES

    @staticmethod
    def timeUnit(image):
        return str((image.axes_units or {}).get("t") or "")

    @staticmethod
    def lengthUnit(image):
        units = image.axes_units or {}
        return next((str(units[d]) for d in SPATIAL_DIMS if units.get(d)), "")

    # ---- geometry ----

    @staticmethod
    def unitScaleToMm(image, userMessages=None):
        factors = {}
        for d in SPATIAL_DIMS:
            unit = (image.axes_units or {}).get(d)
            unit = str(unit) if unit is not None else None
            if unit in LENGTH_UNIT_TO_MM:
                factors[d] = LENGTH_UNIT_TO_MM[unit]
            else:
                factors[d] = 1.0
                message = f"Unknown length unit '{unit}' for axis '{d}', values are used as millimetres."
                logging.warning(message)
                if userMessages:
                    userMessages.AddMessage(vtk.vtkCommand.WarningEvent, message)
        return factors

    @classmethod
    def ijkToRasMatrix(cls, image, userMessages=None):
        """4x4 IJK->RAS (mm) and the orientation source ("rfc4", "assumed-LPS", "assumed-RAS").

        NGFF x/y/z are ITK/LPS physical axes (the ngff-zarr convention) and ``translation``
        is the origin. RFC-4 orientation of every spatial axis gives the direction cosines.
        Without it, the axes are assumed LPS or RAS per the module setting.
        """
        from ngff_zarr.rfc4 import anatomical_orientation_to_itk_direction

        dims = list(image.dims)
        factors = cls.unitScaleToMm(image, userMessages)
        spatialDims = [d for d in SPATIAL_DIMS if d in dims]
        orientations = image.axes_orientations or {}
        columns = {}
        if spatialDims and all(d in orientations for d in spatialDims):
            for d in spatialDims:
                column = anatomical_orientation_to_itk_direction(orientations[d].value)
                if column is None:
                    columns = {}
                    break
                columns[d] = column
        spacing = [image.scale.get(d, 1.0) * factors[d] if d in dims else 1.0 for d in SPATIAL_DIMS]
        origin = [image.translation.get(d, 0.0) * factors[d] if d in dims else 0.0 for d in SPATIAL_DIMS]
        direction = np.eye(3)
        if columns:
            source = "rfc4"
            for columnIndex, d in enumerate(SPATIAL_DIMS):
                if d in columns:
                    direction[:, columnIndex] = columns[d]
            toRas = np.diag([-1.0, -1.0, 1.0, 1.0])
        else:
            assumed = Settings.get(Settings.ORIENTATION, "LPS").upper()
            source = f"assumed-{assumed}"
            toRas = np.eye(4) if assumed == "RAS" else np.diag([-1.0, -1.0, 1.0, 1.0])
            if userMessages:
                userMessages.AddMessage(
                    vtk.vtkCommand.MessageEvent,
                    f"No anatomical orientation (RFC-4) in the store; x/y/z axes assumed {assumed} "
                    "(OME-Zarr module settings).",
                )
        ijkToPhysical = np.eye(4)
        ijkToPhysical[:3, :3] = direction @ np.diag(spacing)
        ijkToPhysical[:3, 3] = origin
        affine = getattr(image, "_omezarrAffine", None)
        if affine is not None:
            affine = affine.copy()
            affine[:3, 3] *= factors.get("x", 1.0)
            ijkToPhysical = affine @ ijkToPhysical
            source += "+affine"
        return toRas @ ijkToPhysical, source

    @staticmethod
    def attachAffine(multiscales):
        """Attach an OME-Zarr 0.6 (RFC-5) intrinsic -> physical affine, as LPS (x,y,z) 4x4, to every level.

        Only a single affine (or rotation) whose input is the intrinsic coordinate system is used;
        other transform chains are ignored, as before.
        """
        metadata = multiscales.metadata
        transforms = getattr(metadata, "coordinateTransformations", None) or []
        try:
            intrinsic = metadata.intrinsic_coordinate_system.name
        except Exception:  # noqa: BLE001 - not 0.6 metadata
            return
        for transform in transforms:
            if getattr(transform.input, "name", None) != intrinsic:
                continue
            if transform.type == "affine" and transform.affine:
                rows = np.asarray(transform.affine, dtype=float)
            elif transform.type == "rotation" and transform.rotation:
                rows = np.hstack([np.asarray(transform.rotation, dtype=float), np.zeros((3, 1))])
            else:
                continue
            if rows.shape != (3, 4):
                logging.warning(f"Ignoring OME-Zarr affine of shape {rows.shape}; only 3D is supported")
                return
            # Metadata arrays are (z, y, x); reorder to LPS (x, y, z).
            order = [2, 1, 0]
            affine = np.eye(4)
            affine[:3, :3] = rows[:, :3][np.ix_(order, order)]
            affine[:3, 3] = rows[order, 3]
            for image in multiscales.images:
                image._omezarrAffine = affine
            return

    @classmethod
    def regionIjkToRas(cls, image, region=None, userMessages=None):
        """IJK->RAS of ``region`` within the image (of the whole image without a region), and its source."""
        ijkToRas, source = cls.ijkToRasMatrix(image, userMessages)
        if region:
            start = np.array([region.get(d, (0, None))[0] for d in SPATIAL_DIMS], dtype=float)
            ijkToRas[:3, 3] = (ijkToRas @ np.append(start, 1.0))[:3]
        return ijkToRas, source

    # ---- array access ----

    @staticmethod
    def computeArray(darray, output, progress=None, label=""):
        """Compute a dask array into ``output`` without freezing the application.

        The array is read slab by slab along its first axis (groups of chunks of about
        64 MiB, at most 32), each slab a parallel dask compute, in a worker thread.
        ``output`` is preallocated, typically a view on a vtkImageData buffer, so a volume is
        never held twice in memory. Cancelling stops after the slab being read and raises
        InterruptedError.
        """
        chunks = list(darray.chunks[0])
        steps = max(1, min(32, int(np.ceil(darray.nbytes / (64 << 20)))))
        groupSize = max(1, int(np.ceil(len(chunks) / steps)))
        total = int(np.ceil(len(chunks) / groupSize))
        state = {"done": 0, "cancel": False}

        def work():
            start = 0
            for step in range(total):
                if state["cancel"]:
                    return
                size = sum(chunks[step * groupSize : (step + 1) * groupSize])
                output[start : start + size] = darray[start : start + size].compute()
                start += size
                state["done"] = step + 1

        def onTick():
            if progress and not progress(state["done"], total, label):
                state["cancel"] = True

        runResponsive(work, onTick)
        if state["cancel"]:
            raise InterruptedError("Loading cancelled")

    @classmethod
    def spatialDaskArray(cls, image, timeIndex=0, channelIndex=0, region=None, userMessages=None):
        """The lazy (z, y, x) sub-array of one time point and channel, and whether a z axis must be added.

        ``region`` maps spatial dim -> (start, stop) in this level's index space; only the
        chunks it intersects are read when the array is computed.
        """
        dims = list(image.dims)
        index = []
        for d in dims:
            if d == "t":
                index.append(int(timeIndex))
            elif d == "c":
                index.append(int(channelIndex))
            elif d in SPATIAL_DIMS:
                index.append(slice(*region[d]) if region and d in region else slice(None))
            else:
                message = f"Axis '{d}' is not supported, using its first element."
                logging.warning(message)
                if userMessages:
                    userMessages.AddMessage(vtk.vtkCommand.WarningEvent, message)
                index.append(0)
        sub = image.data[tuple(index)]
        remaining = [d for d in dims if d in SPATIAL_DIMS]
        if "x" not in remaining or "y" not in remaining:
            raise ValueError("OME-Zarr image must have x and y axes")
        order = [remaining.index(d) for d in ("z", "y", "x") if d in remaining]
        sub = sub.transpose(order)
        return sub, "z" not in remaining

    @staticmethod
    def vtkCompatibleDtype(dtype):
        dtype = np.dtype(dtype)
        if dtype == np.bool_:
            return np.dtype(np.uint8)
        if dtype == np.float16:
            return np.dtype(np.float32)
        return dtype

    @classmethod
    def fillVolumeNode(
        cls, node, image, timeIndex, channelIndex, region, ijkToRas, userMessages, progress, label, dtype=None, reader=None
    ):
        """Read a (z, y, x) volume straight into the node's image buffer.

        The vtkImageData is allocated first and the dask array is computed into a numpy
        view of its scalars, so peak memory is one copy of the volume, not two. ``dtype``
        overrides the stored type (label maps must be integer). ``reader``: the level's
        ChunkReader, which reads remote chunks instead of dask.
        """
        from vtk.util import numpy_support

        sub, addZ = cls.spatialDaskArray(image, timeIndex, channelIndex, region, userMessages)
        shape = (1, *sub.shape) if addZ else tuple(sub.shape)
        dtype = np.dtype(dtype) if dtype else cls.vtkCompatibleDtype(sub.dtype)
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(int(shape[2]), int(shape[1]), int(shape[0]))
        imageData.AllocateScalars(numpy_support.get_vtk_array_type(dtype), 1)
        view = numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars()).reshape(shape)
        if reader is not None and not addZ:
            grid = LevelChunks(image, timeIndex, channelIndex, reader)
            zyx = tuple(tuple(region[d]) if region and d in region else (0, n) for d, n in zip(("z", "y", "x"), grid.shape))
            cls.readChunks(grid, zyx, view, progress, label)
        else:
            cls.computeArray(sub.astype(dtype), view[0] if addZ else view, progress, label)
        node.SetAndObserveImageData(imageData)
        node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(ijkToRas))
        return node

    # ---- channels and display ----

    @staticmethod
    def channelDescriptions(multiscales, image):
        """List of {label, color, window} per channel, from OMERO metadata when present."""
        dims = list(image.dims)
        count = int(image.data.shape[dims.index("c")]) if "c" in dims else 1
        descriptions = [{"label": None, "color": None, "window": None} for _ in range(count)]
        names = list(image.channel_names or [])
        colors = list(image.channel_colors or [])
        for i in range(count):
            if i < len(names) and names[i]:
                descriptions[i]["label"] = str(names[i])
            if i < len(colors) and colors[i]:
                descriptions[i]["color"] = str(colors[i])
        omero = getattr(multiscales.metadata, "omero", None)
        if omero is not None and getattr(omero, "channels", None):
            for i, channel in enumerate(omero.channels[:count]):
                if getattr(channel, "label", None):
                    descriptions[i]["label"] = channel.label
                if getattr(channel, "color", None):
                    descriptions[i]["color"] = channel.color
                window = getattr(channel, "window", None)
                if window is not None:
                    descriptions[i]["window"] = (float(window.start), float(window.end))
        return descriptions

    @staticmethod
    def colorNodeIdForHexColor(hexColor):
        try:
            r, g, b = (int(hexColor.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        except (ValueError, AttributeError, TypeError):
            return None
        high = {name for name, value in (("r", r), ("g", g), ("b", b)) if value >= 128}
        return CHANNEL_COLOR_NODE_IDS.get(
            {
                frozenset("r"): "red",
                frozenset("g"): "green",
                frozenset("b"): "blue",
                frozenset("rg"): "yellow",
                frozenset("gb"): "cyan",
                frozenset("rb"): "magenta",
                frozenset("rgb"): "grey",
            }.get(frozenset(high))
        )

    @staticmethod
    def setupDisplay(node, description, multiChannel):
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        if display is None:
            return
        if description.get("window"):
            start, end = description["window"]
            if end > start:
                display.SetAutoWindowLevel(False)
                display.SetWindowLevelMinMax(start, end)
        colorNodeId = OMEZarrLogic.colorNodeIdForHexColor(description.get("color")) if multiChannel else None
        if colorNodeId and slicer.mrmlScene.GetNodeByID(colorNodeId):
            display.SetAndObserveColorNodeID(colorNodeId)

    @staticmethod
    def defaultNodeName(path):
        base = os.path.basename(str(path).rstrip("/"))
        for suffix in (".ome.zarr", ".zarr.zip", ".ozx", ".zarr"):
            if base.lower().endswith(suffix):
                return base[: -len(suffix)]
        return base or "OMEZarr"

    @staticmethod
    def setNodeAttributes(node, path, level, dims, timeIndex, channelIndex, lengthUnit, orientationSource, region):
        node.SetAttribute("OMEZarr.Path", normalizeStorePath(path))
        node.SetAttribute("OMEZarr.Level", str(level))
        node.SetAttribute("OMEZarr.Dims", ",".join(dims))
        node.SetAttribute("OMEZarr.TimeIndex", str(timeIndex))
        node.SetAttribute("OMEZarr.Channel", str(channelIndex))
        node.SetAttribute("OMEZarr.LengthUnit", lengthUnit)
        node.SetAttribute("OMEZarr.OrientationSource", orientationSource)
        if region:
            node.SetAttribute("OMEZarr.Region", ";".join(f"{d}:{s}-{e}" for d, (s, e) in region.items()))

    # ---- loading ----

    @classmethod
    @removesNodesWhenCancelled
    def loadImage(
        cls,
        path,
        level=None,
        timeIndex=None,
        channels=None,
        region=None,
        name=None,
        maxBytes=None,
        userMessages=None,
        multiscales=None,
        labels=None,
        timeMode=None,
        progress=None,
        asLabelMap=None,
        announceLevel=True,
    ):
        """Load a store into volume nodes. Returns the created nodes (volumes, proxies, label maps).

        ``region``: spatial dim -> (start, stop) at the chosen level (partial read).
        ``timeIndex``: a single time point; otherwise the setting decides between a
        Sequence of all time points and index 0. ``labels``: also load the ``labels`` groups.
        ``announceLevel``: warn when a downsampled level is loaded (off while streaming,
        where the coarse level is only the first thing shown).
        """
        if multiscales is None and isBioformats2rawRoot(path):
            series = bioformats2rawSeries(path)
            if not series:
                raise ValueError(f"No image series found in the bioformats2raw container {path}")
            baseName = name or cls.defaultNodeName(path)
            nodes = []
            for index, seriesPath in enumerate(series):
                seriesName = baseName if len(series) == 1 else f"{baseName}_{index}"
                nodes += cls.loadImage(
                    seriesPath,
                    level,
                    timeIndex,
                    channels,
                    region,
                    seriesName,
                    maxBytes,
                    userMessages,
                    None,
                    labels,
                    timeMode,
                    progress,
                    asLabelMap,
                )
            return nodes
        multiscales = multiscales or cls.openMultiscales(path)
        if isLabelStore(path) or (asLabelMap is None and cls.looksLikeLabelMap(multiscales)) or asLabelMap:
            return cls.loadLabelStore(
                path, level, region, name, userMessages, multiscales, progress=progress, maxBytes=maxBytes
            )
        baseImage = multiscales.images[0]
        descriptions = cls.channelDescriptions(multiscales, baseImage)
        channels = list(range(len(descriptions))) if channels is None else list(channels)
        timePoints = cls.axisLength(baseImage, "t")
        if timeIndex is None:
            timeMode = timeMode or Settings.get(Settings.TIME_MODE, "sequence")
            asSequence = timePoints > 1 and timeMode == "sequence"
            timeIndex = 0
        else:
            asSequence = False
            timeIndex = int(timeIndex)
        copies = len(channels) * (timePoints if asSequence else 1)
        if level is None:
            level = cls.selectLevel(multiscales, maxBytes or cls.maxBytesFromSettings(), copies)
        level = int(level)
        if level < 0 or level >= len(multiscales.images):
            raise ValueError(f"Level {level} out of range (0..{len(multiscales.images) - 1})")
        image = multiscales.images[level]
        dims = list(image.dims)

        if (
            region is None
            and userMessages
            and announceLevel
            and cls.volumeBytes(image) * copies > (maxBytes or cls.maxBytesFromSettings())
        ):
            userMessages.AddMessage(
                vtk.vtkCommand.WarningEvent,
                f"No resolution level fits the memory budget; loading level {level} "
                f"({cls.volumeBytes(image) * copies / 2**30:.2f} GiB). Use a region of interest for large stores.",
            )
        if level > 0 and region is None and userMessages and announceLevel:
            full = cls.volumeBytes(multiscales.images[0]) * copies
            factor = cls.levelInfo(multiscales)[level]["downsample"]
            userMessages.AddMessage(
                vtk.vtkCommand.WarningEvent,
                f"Loaded multiscale level {level} (downsampled by "
                f"{', '.join(f'{d}: {v:g}' for d, v in factor.items())}) because level 0 "
                f"would need {full / 2**30:.2f} GiB. Use 'Refine current view' or a region of "
                "interest in the OME-Zarr module for full resolution, or raise the memory budget.",
            )
        if timePoints > 1 and not asSequence and userMessages:
            userMessages.AddMessage(
                vtk.vtkCommand.MessageEvent, f"Time axis has {timePoints} points; loaded index {timeIndex}."
            )

        ijkToRas, orientationSource = cls.regionIjkToRas(image, region, userMessages)

        baseName = name or cls.defaultNodeName(path)
        lengthUnit = cls.lengthUnit(image)
        nodes = []
        sequences = []
        for channelIndex in channels:
            label = descriptions[channelIndex]["label"] or (f"c{channelIndex}" if len(descriptions) > 1 else None)
            nodeName = f"{baseName}_{label}" if label else baseName
            if region:
                nodeName += "_ROI"
            nodeName = slicer.mrmlScene.GenerateUniqueName(nodeName)
            if asSequence:
                sequence = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", nodeName + "_sequence")
                sequence.SetIndexName("time")
                sequence.SetIndexUnit(cls.timeUnit(image))
                timeScale = float(image.scale.get("t", 1.0))
                for t in range(timePoints):
                    dataNode = slicer.vtkMRMLScalarVolumeNode()
                    dataNode.SetName(nodeName)
                    cls.fillVolumeNode(
                        dataNode,
                        image,
                        t,
                        channelIndex,
                        region,
                        ijkToRas,
                        userMessages,
                        progress,
                        f"{nodeName}, t={t + 1}/{timePoints}",
                    )
                    cls.setNodeAttributes(
                        dataNode, path, level, dims, t, channelIndex, lengthUnit, orientationSource, region
                    )
                    sequence.SetDataNodeAtValue(dataNode, f"{t * timeScale:g}")
                cls.setNodeAttributes(
                    sequence, path, level, dims, -1, channelIndex, lengthUnit, orientationSource, region
                )
                sequences.append((sequence, descriptions[channelIndex]))
            else:
                node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
                cls.fillVolumeNode(
                    node, image, timeIndex, channelIndex, region, ijkToRas, userMessages, progress, nodeName,
                    reader=runResponsive(lambda: cls.chunkReader(multiscales, level)),
                )
                cls.setNodeAttributes(
                    node, path, level, dims, timeIndex, channelIndex, lengthUnit, orientationSource, region
                )
                cls.setupDisplay(node, descriptions[channelIndex], multiChannel=len(descriptions) > 1)
                nodes.append(node)

        if sequences:
            browser = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", baseName + "_browser")
            for index, (sequence, _description) in enumerate(sequences):
                if index == 0:
                    browser.SetAndObserveMasterSequenceNodeID(sequence.GetID())
                else:
                    browser.AddSynchronizedSequenceNodeID(sequence.GetID())
            slicer.modules.sequences.logic().UpdateProxyNodesFromSequences(browser)
            for sequence, description in sequences:
                proxy = browser.GetProxyNode(sequence)
                if proxy is not None:
                    cls.setupDisplay(proxy, description, multiChannel=len(descriptions) > 1)
                    nodes.append(proxy)

        if labels is None:
            labels = Settings.get(Settings.LOAD_LABELS, True)
        if labels:
            regionBounds = cls.rasBoundsOfRegion(image, region, ijkToRas) if region else None
            nodes += cls.loadLabelGroups(path, image, level, baseName, regionBounds, userMessages, progress)

        cls.applyDisplayUnits(lengthUnit)
        return nodes

    @staticmethod
    def rasBoundsOfRegion(image, region, ijkToRas):
        """RAS bounds of a region whose ``ijkToRas`` already has the region origin."""
        size = [region[d][1] - region[d][0] if d in region else OMEZarrLogic.axisLength(image, d) for d in SPATIAL_DIMS]
        corners = np.array(
            [[i, j, k, 1.0] for i in (0, size[0] - 1) for j in (0, size[1] - 1) for k in (0, size[2] - 1)]
        )
        ras = (ijkToRas @ corners.T).T[:, :3]
        return [ras[:, 0].min(), ras[:, 0].max(), ras[:, 1].min(), ras[:, 1].max(), ras[:, 2].min(), ras[:, 2].max()]

    # ---- labels ----

    @classmethod
    def matchingLevel(cls, multiscales, referenceImage):
        """Level of ``multiscales`` whose spacing is closest to ``referenceImage``'s."""
        reference = np.array([referenceImage.scale.get(d, 1.0) for d in SPATIAL_DIMS if d in referenceImage.dims])
        best, bestDistance = 0, None
        for index, image in enumerate(multiscales.images):
            scale = np.array([image.scale.get(d, 1.0) for d in SPATIAL_DIMS if d in referenceImage.dims])
            distance = (
                float(np.abs(np.log(scale / reference)).sum()) if scale.shape == reference.shape else float("inf")
            )
            if bestDistance is None or distance < bestDistance:
                best, bestDistance = index, distance
        return best

    @classmethod
    def loadLabelGroups(cls, root, image, level, baseName, rasBounds=None, userMessages=None, progress=None):
        nodes = []
        for labelName in labelGroupNames(root):
            labelPath = joinStorePath(root, "labels", labelName)
            try:
                labelMultiscales = cls.openMultiscales(labelPath)
                labelLevel = cls.matchingLevel(labelMultiscales, image)
                region = None
                if rasBounds is not None:
                    region = cls.regionFromRasBounds(labelMultiscales.images[labelLevel], rasBounds)
                nodes += cls.loadLabelStore(
                    labelPath, labelLevel, region, f"{baseName}_{labelName}", userMessages, labelMultiscales, progress
                )
            except Exception as e:  # noqa: BLE001 - a broken label group must not block the image
                message = f"Could not load labels '{labelName}': {e}"
                logging.exception(message)
                if userMessages:
                    userMessages.AddMessage(vtk.vtkCommand.WarningEvent, message)
        return nodes

    @classmethod
    def looksLikeLabelMap(cls, multiscales, maxValues=64):
        """Integer store without OMERO metadata whose coarsest level has few distinct values.

        Segmentation masks are often written as plain OME-Zarr images without
        ``image-label`` metadata; this heuristic loads them as label maps when the
        setting is on. The coarsest level is small, so the check is cheap.
        """
        if not Settings.get(Settings.DETECT_LABEL_MAPS, True):
            return False
        image = multiscales.images[-1]
        dtype = np.dtype(image.data.dtype)
        if not np.issubdtype(dtype, np.integer) or dtype.itemsize > 2 or "c" in image.dims or "t" in image.dims:
            return False
        if getattr(multiscales.metadata, "omero", None) is not None:
            return False
        if image.data.nbytes > (256 << 20):
            return False
        try:
            values = np.unique(cls.readLevel(multiscales, len(multiscales.images) - 1))
        except Exception:  # noqa: BLE001 - a failed probe just means "not a label map"
            return False
        return 1 < len(values) <= maxValues and values.min() >= 0

    @classmethod
    def loadLabelStore(
        cls,
        path,
        level=None,
        region=None,
        name=None,
        userMessages=None,
        multiscales=None,
        progress=None,
        maxBytes=None,
    ):
        """Load an ``image-label`` multiscales (or a store that looks like one) into a label map."""
        multiscales = multiscales or cls.openMultiscales(path)
        if level is None:
            level = cls.selectLevel(multiscales, maxBytes or cls.maxBytesFromSettings())
        image = multiscales.images[int(level)]
        dims = list(image.dims)
        ijkToRas, orientationSource = cls.regionIjkToRas(image, region, userMessages)
        nodeName = slicer.mrmlScene.GenerateUniqueName((name or cls.defaultNodeName(path)) + ("_ROI" if region else ""))
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", nodeName)
        stored = np.dtype(image.data.dtype)
        dtype = None if np.issubdtype(stored, np.integer) or stored == np.bool_ else np.int32
        cls.fillVolumeNode(node, image, 0, 0, region, ijkToRas, userMessages, progress, nodeName, dtype)
        cls.setNodeAttributes(node, path, int(level), dims, 0, 0, cls.lengthUnit(image), orientationSource, region)
        node.CreateDefaultDisplayNodes()
        attrs = (multiscales.root_attributes or {}).get("image-label") if multiscales.root_attributes else None
        if attrs is None:
            attrs = (readStoreAttributes(path) or {}).get("image-label")
        colorNode = cls.colorTableFromImageLabel(attrs, nodeName + "_colors")
        if colorNode is not None:
            node.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
        cls.applyDisplayUnits(cls.lengthUnit(image))
        if Settings.get(Settings.LABELS_AS_SEGMENTATION, False):
            node = cls.segmentationFromLabelMap(node, colorNode)
        return [node]

    @staticmethod
    def segmentationFromLabelMap(labelNode, colorNode=None):
        """Replace a label map (and its colour table) by a Segmentation node carrying the same attributes."""
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", labelNode.GetName())
        segmentation.CreateDefaultDisplayNodes()
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentation)
        for key in labelNode.GetAttributeNames():
            segmentation.SetAttribute(key, labelNode.GetAttribute(key))
        slicer.mrmlScene.RemoveNode(labelNode)
        if colorNode is not None:
            slicer.mrmlScene.RemoveNode(colorNode)
        return segmentation

    @staticmethod
    def colorTableFromImageLabel(imageLabel, name):
        """Colour table node from ``image-label`` colours and properties, or None."""
        if not isinstance(imageLabel, dict):
            return None
        colors = {}
        for entry in imageLabel.get("colors") or []:
            try:
                value = int(entry["label-value"])
                rgba = [float(v) for v in entry.get("rgba", [0, 0, 0, 255])]
            except (KeyError, TypeError, ValueError):
                continue
            colors.setdefault(value, {})["rgba"] = rgba
        for entry in imageLabel.get("properties") or []:
            try:
                value = int(entry["label-value"])
            except (KeyError, TypeError, ValueError):
                continue
            label = entry.get("name") or entry.get("label") or entry.get("acronym")
            if label:
                colors.setdefault(value, {})["name"] = str(label)
        if not colors:
            return None
        maxValue = max(colors)
        if maxValue >= MAX_COLOR_TABLE_ENTRIES or min(colors) < 0:
            logging.warning(f"Label values up to {maxValue} are too many for a colour table; default colours are used.")
            return None
        colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", name)
        colorNode.SetTypeToUser()
        colorNode.SetHideFromEditors(False)
        colorNode.SetNumberOfColors(maxValue + 1)
        colorNode.SetColor(0, "background", 0.0, 0.0, 0.0, 0.0)
        for value, info in colors.items():
            r, g, b, a = (info.get("rgba") or [128.0, 128.0, 128.0, 255.0])[:4]
            colorNode.SetColor(value, info.get("name", str(value)), r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        return colorNode

    # ---- region of interest ----

    @classmethod
    def regionFromRasBounds(cls, image, rasBounds):
        """Map RAS bounds [xmin,xmax,ymin,ymax,zmin,zmax] to index ranges at this level."""
        ijkToRas, _source = cls.ijkToRasMatrix(image)
        rasToIjk = np.linalg.inv(ijkToRas)
        corners = np.array(
            [[rasBounds[i], rasBounds[2 + j], rasBounds[4 + k], 1.0] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        )
        ijk = (rasToIjk @ corners.T).T[:, :3]
        shape = cls.spatialShape(image)
        region = {}
        for axisIndex, d in enumerate(SPATIAL_DIMS):
            if d not in shape:
                continue
            start = int(np.clip(np.floor(ijk[:, axisIndex].min() + 0.5), 0, shape[d]))
            stop = int(np.clip(np.ceil(ijk[:, axisIndex].max() + 0.5), 0, shape[d]))
            if stop <= start:
                raise ValueError("Region of interest does not intersect the image")
            region[d] = (start, stop)
        return region

    @classmethod
    def loadRegion(
        cls, path, roiNode, level=0, timeIndex=None, channels=None, name=None, userMessages=None, progress=None
    ):
        multiscales = cls.openMultiscales(path)
        bounds = [0.0] * 6
        roiNode.GetRASBounds(bounds)
        region = cls.regionFromRasBounds(multiscales.images[int(level)], bounds)
        return cls.loadImage(
            path,
            level=level,
            timeIndex=timeIndex,
            channels=channels,
            region=region,
            name=name,
            userMessages=userMessages,
            multiscales=multiscales,
            progress=progress,
        )

    # ---- view-driven refinement ----

    @staticmethod
    def sliceViewRasBounds(sliceViewName):
        """RAS bounds of the block a slice view shows: its field of view, extended along the
        normal by half the smaller side."""
        sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
        if sliceWidget is None:
            raise ValueError(f"No slice view named '{sliceViewName}'")
        sliceNode = sliceWidget.mrmlSliceNode()
        sliceToRas = slicer.util.arrayFromVTKMatrix(sliceNode.GetSliceToRAS())
        width, height, _depth = sliceNode.GetFieldOfView()
        thickness = min(width, height) / 2.0
        corners = np.array(
            [
                [x, y, z, 1.0]
                for x in (-width / 2, width / 2)
                for y in (-height / 2, height / 2)
                for z in (-thickness / 2, thickness / 2)
            ]
        )
        ras = (sliceToRas @ corners.T).T[:, :3]
        return [ras[:, 0].min(), ras[:, 0].max(), ras[:, 1].min(), ras[:, 1].max(), ras[:, 2].min(), ras[:, 2].max()]

    @staticmethod
    def currentTimeIndex(path):
        """Selected item of the sequence browser showing this store, else 0."""
        for browser in slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode"):
            master = browser.GetMasterSequenceNode()
            if master is not None and samePath(master.GetAttribute("OMEZarr.Path"), path):
                return max(0, browser.GetSelectedItemNumber())
        return 0

    @staticmethod
    def sourceVolumeNode(path, channel, sliceViewName=None):
        """The full-extent volume loaded from this store and channel: the one shown as
        background of ``sliceViewName`` when it matches, else the most recently loaded."""

        def matches(node):
            return (
                node is not None
                and node.IsA("vtkMRMLScalarVolumeNode")
                and not node.IsA("vtkMRMLLabelMapVolumeNode")
                and samePath(node.GetAttribute("OMEZarr.Path"), path)
                and node.GetAttribute("OMEZarr.Channel") == str(channel)
                and not node.GetAttribute("OMEZarr.Refined")
                and not node.GetAttribute("OMEZarr.Region")
            )

        if sliceViewName:
            sliceWidget = slicer.app.layoutManager().sliceWidget(sliceViewName)
            if sliceWidget is not None:
                background = sliceWidget.sliceLogic().GetBackgroundLayer().GetVolumeNode()
                if matches(background):
                    return background
        candidates = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode") if matches(n)]
        return candidates[-1] if candidates else None

    @classmethod
    def matchDisplay(cls, node, path, sliceViewName=None):
        """Give a refined block the window/level and colour of the volume it refines."""
        source = cls.sourceVolumeNode(path, node.GetAttribute("OMEZarr.Channel"), sliceViewName)
        if source is None or source.GetDisplayNode() is None or node.GetDisplayNode() is None:
            return
        sourceDisplay, display = source.GetDisplayNode(), node.GetDisplayNode()
        display.SetAutoWindowLevel(False)
        display.SetWindowLevel(sourceDisplay.GetWindow(), sourceDisplay.GetLevel())
        if sourceDisplay.GetColorNodeID():
            display.SetAndObserveColorNodeID(sourceDisplay.GetColorNodeID())

    @classmethod
    def refinedNodes(cls, path=None, sliceViewName=None):
        nodes = slicer.util.getNodesByClass("vtkMRMLVolumeNode") + slicer.util.getNodesByClass(
            "vtkMRMLSegmentationNode"
        )
        return [
            n
            for n in nodes
            if n.GetAttribute("OMEZarr.Refined") == "1"
            and (path is None or samePath(n.GetAttribute("OMEZarr.Path"), path))
            and (sliceViewName is None or n.GetAttribute("OMEZarr.RefinedView") == sliceViewName)
        ]

    @classmethod
    def refineView(cls, path, sliceViewName="Red", maxBytes=None, timeIndex=None, userMessages=None, progress=None):
        """Reload what ``sliceViewName`` shows at the finest level whose block fits the budget.

        Each slice view keeps its own refined block, shown in that view's foreground layer
        (labels in its label layer) with the window/level of the coarse volume. The
        previous block of the same view and store is replaced. Returns the new nodes.
        """
        multiscales = cls.openMultiscales(path)
        bounds = cls.sliceViewRasBounds(sliceViewName)
        budget = maxBytes or cls.maxBytesFromSettings()
        channels = len(cls.channelDescriptions(multiscales, multiscales.images[0]))
        chosen = None
        for level, image in enumerate(multiscales.images):
            try:
                region = cls.regionFromRasBounds(image, bounds)
            except ValueError:
                continue
            if cls.volumeBytes(image, region) * channels <= budget:
                chosen = (level, region)
                break
        if chosen is None:
            raise ValueError("The view does not intersect the image, or no level fits the memory budget")
        level, region = chosen
        shown = cls.sourceVolumeNode(path, 0, sliceViewName)
        shownLevel = shown.GetAttribute("OMEZarr.Level") if shown is not None else None
        if shownLevel is not None and level >= int(shownLevel):
            raise ValueError("Zoom in: no finer level than the one displayed fits the memory budget for this view")
        for node in cls.refinedNodes(path, sliceViewName):
            display = node.GetDisplayNode() if node.IsA("vtkMRMLVolumeNode") else None
            colorNode = display.GetColorNode() if display else None
            slicer.mrmlScene.RemoveNode(node)
            if colorNode is not None and colorNode.GetAttribute("OMEZarr.Refined"):
                slicer.mrmlScene.RemoveNode(colorNode)
        nodes = cls.loadImage(
            path,
            level=level,
            timeIndex=cls.currentTimeIndex(path) if timeIndex is None else timeIndex,
            region=region,
            name=f"{cls.defaultNodeName(path)}_{sliceViewName}",
            userMessages=userMessages,
            multiscales=multiscales,
            progress=progress,
        )
        scalars, labels = [], []
        for node in nodes:
            node.SetAttribute("OMEZarr.Refined", "1")
            node.SetAttribute("OMEZarr.RefinedView", sliceViewName)
            if node.IsA("vtkMRMLLabelMapVolumeNode"):
                labels.append(node)
                colorNode = node.GetDisplayNode().GetColorNode() if node.GetDisplayNode() else None
                if colorNode is not None and colorNode.GetAttribute("OMEZarr.Path") is None:
                    colorNode.SetAttribute("OMEZarr.Refined", "1")
            elif node.IsA("vtkMRMLScalarVolumeNode"):
                scalars.append(node)
                cls.matchDisplay(node, path, sliceViewName)
        composite = slicer.app.layoutManager().sliceWidget(sliceViewName).sliceLogic().GetSliceCompositeNode()
        if scalars:
            composite.SetForegroundVolumeID(scalars[0].GetID())
            composite.SetForegroundOpacity(1.0)
        if labels:
            composite.SetLabelVolumeID(labels[0].GetID())
        return nodes

    # ---- automatic refinement ----

    _autoRefiners = {}

    DEFAULT_AUTO_REFINE_VIEWS = ("Red", "Yellow", "Green")

    @classmethod
    def startAutoRefine(cls, path, sliceViewNames=DEFAULT_AUTO_REFINE_VIEWS, delayMs=600, maxBytes=None):
        """Refine each of ``sliceViewNames`` whenever it stops moving. Returns the AutoRefiner.

        The memory budget is shared equally between the views.
        """
        cls.stopAutoRefine(path)
        if isinstance(sliceViewNames, str):
            sliceViewNames = (sliceViewNames,)
        refiner = AutoRefiner(str(path), list(sliceViewNames), delayMs, maxBytes)
        cls._autoRefiners[str(path)] = refiner
        return refiner

    @classmethod
    def stopAutoRefine(cls, path=None):
        keys = [str(path)] if path is not None else list(cls._autoRefiners)
        for key in keys:
            refiner = cls._autoRefiners.pop(key, None)
            if refiner is not None:
                refiner.stop()

    @classmethod
    def autoRefiner(cls, path):
        return cls._autoRefiners.get(str(path))

    # ---- streaming ----

    _streamers = {}
    panel = None  # the module widget, refreshed by streamers while they read

    @classmethod
    def streamingLevels(cls, multiscales, maxBytes=None, timeMode=None):
        """(coarsest, target) levels when the store should be streamed, else None.

        Streaming covers one 3D channel and one time point: it pays off when the level the
        budget allows is finer than the coarsest one, so there is something to wait for.
        """
        base = multiscales.images[0]
        dims = list(base.dims)
        if "z" not in dims or len(cls.channelDescriptions(multiscales, base)) != 1:
            return None
        if cls.axisLength(base, "t") > 1 and (timeMode or Settings.get(Settings.TIME_MODE, "sequence")) == "sequence":
            return None
        coarsest = len(multiscales.images) - 1
        target = cls.selectLevel(multiscales, maxBytes or cls.maxBytesFromSettings())
        return (coarsest, target) if target < coarsest else None

    @classmethod
    def startStreaming(cls, path, node, targetLevel, sliceViewNames=DEFAULT_AUTO_REFINE_VIEWS, timeIndex=0):
        """Stream ``targetLevel`` of ``path`` into ``node`` (which shows a coarser level). Returns the Streamer."""
        cls.stopStreaming(path)
        cls.stopAutoRefine(path)
        streamer = Streamer(str(path), node, cls.openMultiscales(path), int(targetLevel), sliceViewNames, timeIndex)
        cls._streamers[str(path)] = streamer
        return streamer

    @classmethod
    def stopStreaming(cls, path=None, wait=False):
        """Stop streaming ``path`` (all stores without one). ``wait``: also let the reader threads
        finish the chunk they are reading, as Python must not be finalized under them."""
        keys = [str(path)] if path is not None else list(cls._streamers)
        for key in keys:
            streamer = cls._streamers.pop(key, None)
            if streamer is not None:
                streamer.stop(wait)

    @classmethod
    def streamer(cls, path):
        return cls._streamers.get(str(path))

    @classmethod
    def onAboutToQuit(cls):
        HttpClient.shutdown.set()  # readers waiting out a busy server give up at once
        cls.stopStreaming(wait=True)

    @classmethod
    def startVolumeRendering(cls, path):
        """Volume-render a streamed store in the first 3D view. Returns the node the view renders."""
        streamer = cls.streamer(path)
        if streamer is None:
            raise ValueError("This store is not being streamed: render its volume in the Volume Rendering module")
        return streamer.enable3D()

    @classmethod
    def stopVolumeRendering(cls, path):
        streamer = cls.streamer(path)
        if streamer is not None:
            streamer.disable3D()

    # ---- display units ----

    @staticmethod
    def applyDisplayUnits(lengthUnit):
        """Switch Slicer's length display to the store's unit when the setting is on."""
        symbol = UNIT_SYMBOLS.get(lengthUnit)
        if not Settings.get(Settings.DISPLAY_UNITS, False) or symbol in (None, "mm"):
            return
        unitNode = slicer.mrmlScene.GetNodeByID("vtkMRMLUnitNodeApplicationLength")
        if unitNode is None:
            return
        unitNode.SetDisplayCoefficient(1.0 / LENGTH_UNIT_TO_MM[lengthUnit])
        unitNode.SetSuffix(symbol)
        unitNode.SetPrecision(2)

    @staticmethod
    def resetDisplayUnits():
        unitNode = slicer.mrmlScene.GetNodeByID("vtkMRMLUnitNodeApplicationLength")
        if unitNode is None:
            return
        unitNode.SetDisplayCoefficient(1.0)
        unitNode.SetSuffix("mm")
        unitNode.SetPrecision(3)

    # ---- write side ----

    @staticmethod
    def ngffImageFromVolumeNode(volumeNode, name=None):
        """NgffImage (z,y,x, millimeter, RFC-4 orientation) from a scalar or label map volume node."""
        import ngff_zarr
        from ngff_zarr.rfc4 import itk_direction_to_anatomical_orientation

        array = slicer.util.arrayFromVolume(volumeNode)
        if array.ndim != 3:
            raise ValueError("Only single-component volumes can be written as OME-Zarr")
        ijkToRasVtk = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(ijkToRasVtk)
        ijkToRas = slicer.util.arrayFromVTKMatrix(ijkToRasVtk)
        ijkToLps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ ijkToRas
        spacing = np.linalg.norm(ijkToLps[:3, :3], axis=0)
        direction = ijkToLps[:3, :3] / spacing
        origin = ijkToLps[:3, 3]
        orientations = {
            d: itk_direction_to_anatomical_orientation(list(direction[:, i])) for i, d in enumerate(SPATIAL_DIMS)
        }
        image = ngff_zarr.to_ngff_image(
            np.ascontiguousarray(array),
            dims=("z", "y", "x"),
            scale={d: float(spacing[i]) for i, d in enumerate(SPATIAL_DIMS)},
            translation={d: float(origin[i]) for i, d in enumerate(SPATIAL_DIMS)},
            name=name or volumeNode.GetName(),
            axes_units={d: "millimeter" for d in SPATIAL_DIMS},
        )
        image.axes_orientations = orientations
        image._omezarrAffine = OMEZarrLogic.residualRotationAffine(direction, orientations, origin)
        return image

    @staticmethod
    def residualRotationAffine(direction, orientations, origin):
        """LPS (x,y,z) 4x4 affine for the rotation RFC-4 cannot express, or None if axis-aligned.

        RFC-4 records each axis's nearest anatomical direction (a signed permutation P). The
        remainder R = direction @ P^T is written as an OME-Zarr 0.6 (RFC-5) affine about the
        origin, so that affine @ (P, spacing, origin) reproduces the full direction.
        """
        from ngff_zarr.rfc4 import anatomical_orientation_to_itk_direction

        snapped = np.column_stack(
            [anatomical_orientation_to_itk_direction(orientations[d].value) for d in SPATIAL_DIMS]
        )
        rotation = direction @ snapped.T
        if np.allclose(rotation, np.eye(3), atol=1e-9):
            return None
        affine = np.eye(4)
        affine[:3, :3] = rotation
        affine[:3, 3] = origin - rotation @ origin
        return affine

    @staticmethod
    def addAffineToMultiscales(multiscales, affineLps):
        """Add ``affineLps`` as an intrinsic -> physical RFC-5 affine to 0.6 multiscales metadata."""
        from ngff_zarr.v06 import zarr_metadata as v06

        metadata = multiscales.metadata
        intrinsic = metadata.intrinsic_coordinate_system
        physical = v06.CoordinateSystem(
            name="physical", axes=[v06.Axis(name=a.name, type=a.type, unit=a.unit) for a in intrinsic.axes]
        )
        # Metadata arrays are (z, y, x); reverse the LPS (x, y, z) matrix accordingly.
        order = [2, 1, 0]
        matrix = affineLps[np.ix_(order, order)]
        translation = affineLps[order, 3]
        metadata.coordinateSystems = [cs for cs in metadata.coordinateSystems if cs.name != "physical"] + [physical]
        metadata.coordinateTransformations = [
            v06.Affine(
                affine=[[float(v) for v in row] + [float(t)] for row, t in zip(matrix, translation)],
                input=v06.CoordinateSystemIdentifier(name=intrinsic.name),
                output=v06.CoordinateSystemIdentifier(name="physical"),
                name="intrinsic_to_physical",
            )
        ]

    @staticmethod
    def imageLabelFromNode(labelNode, version):
        """``image-label`` metadata for a label map volume, from its colour table."""
        values = np.unique(slicer.util.arrayFromVolume(labelNode))
        values = [int(v) for v in values if v != 0]
        colors, properties = [], []
        display = labelNode.GetDisplayNode()
        colorNode = display.GetColorNode() if display else None
        for value in values:
            rgba = [0.5, 0.5, 0.5, 1.0]
            name = str(value)
            if colorNode is not None and value < colorNode.GetNumberOfColors():
                colorNode.GetColor(value, rgba)
                name = colorNode.GetColorName(value) or name
            colors.append({"label-value": value, "rgba": [int(round(c * 255)) for c in rgba]})
            properties.append({"label-value": value, "name": name})
        return {"version": version, "colors": colors, "properties": properties}

    @classmethod
    def writeVolume(cls, node, storePath, progress=None):
        """Write a scalar volume, label map or segmentation as an OME-Zarr multiscales store."""
        ngff_zarr = cls.ensureNgffZarr()
        from ngff_zarr import Methods

        if isRemoteUrl(storePath):
            raise ValueError("Writing to a remote store is not supported; write to a local directory and upload it.")
        if node.IsA("vtkMRMLSegmentationNode"):
            labelNode = cls.labelMapFromSegmentation(node)
            try:
                return cls.writeVolume(labelNode, storePath, progress)
            finally:
                display = labelNode.GetDisplayNode()
                colorNode = display.GetColorNode() if display else None
                slicer.mrmlScene.RemoveNode(labelNode)
                if colorNode is not None:
                    slicer.mrmlScene.RemoveNode(colorNode)
        isLabel = node.IsA("vtkMRMLLabelMapVolumeNode")
        image = cls.ngffImageFromVolumeNode(node)
        method = Methods.ITKWASM_LABEL_IMAGE if isLabel else None

        def write():
            try:
                multiscales = ngff_zarr.to_multiscales(image, method=method)
            except Exception:  # noqa: BLE001 - fall back to a single level rather than fail
                logging.exception("Multiscale generation failed; writing a single level")
                multiscales = ngff_zarr.to_multiscales(image, scale_factors=[])
            affine = getattr(image, "_omezarrAffine", None)
            if affine is None:
                ngff_zarr.to_ome_zarr(storePath, multiscales, overwrite=True)
            else:
                # Oblique volume: only OME-Zarr 0.6 (RFC-5) can carry the rotation.
                cls.addAffineToMultiscales(multiscales, affine)
                ngff_zarr.to_ome_zarr(storePath, multiscales, version="0.6", overwrite=True)
            return multiscales

        if progress:
            progress(0, 1, f"Writing {os.path.basename(storePath)}")
        multiscales = runResponsive(write)
        version = str(getattr(multiscales.metadata, "version", "0.5") or "0.5")
        if getattr(image, "_omezarrAffine", None) is not None:
            version = "0.6"
        if isLabel:
            cls.addImageLabelMetadata(storePath, cls.imageLabelFromNode(node, version))
            cls.registerLabelInParent(storePath)
        cls.clearCache()
        if progress:
            progress(1, 1)
        return storePath

    @staticmethod
    def labelMapFromSegmentation(segmentationNode):
        """Temporary label map (with a colour table of segment names and colours) of all segments."""
        labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", segmentationNode.GetName())
        logic = slicer.modules.segmentations.logic()
        if not logic.ExportAllSegmentsToLabelmapNode(
            segmentationNode, labelNode, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
        ):
            slicer.mrmlScene.RemoveNode(labelNode)
            raise ValueError("Could not export the segmentation to a label map")
        return labelNode

    @staticmethod
    def _updateGroupMetadata(groupPath, update):
        """Apply ``update(omeAttrs)`` to the OME attributes of a local Zarr group (v2 or v3)."""
        v3 = os.path.join(groupPath, "zarr.json")
        v2 = os.path.join(groupPath, ".zattrs")
        if os.path.isfile(v3):
            with open(v3, encoding="utf-8") as fp:
                document = json.load(fp)
            attributes = document.setdefault("attributes", {})
            ome = attributes.setdefault("ome", {})
            update(ome)
            with open(v3, "w", encoding="utf-8") as fp:
                json.dump(document, fp, indent=2)
        else:
            document = {}
            if os.path.isfile(v2):
                with open(v2, encoding="utf-8") as fp:
                    document = json.load(fp)
            update(document)
            with open(v2, "w", encoding="utf-8") as fp:
                json.dump(document, fp, indent=2)

    @classmethod
    def addImageLabelMetadata(cls, storePath, imageLabel):
        cls._updateGroupMetadata(storePath, lambda ome: ome.__setitem__("image-label", imageLabel))

    @classmethod
    def registerLabelInParent(cls, storePath):
        """When written into ``<image>/labels/<name>``, list the name in the ``labels`` group."""
        labelsDir = os.path.dirname(os.path.abspath(storePath))
        imageRoot = os.path.dirname(labelsDir)
        if os.path.basename(labelsDir) != "labels" or not omeZarrRootFromPath(imageRoot):
            return
        name = os.path.basename(storePath)
        zarrV3 = os.path.isfile(os.path.join(imageRoot, "zarr.json"))
        version = (readStoreAttributes(imageRoot) or {}).get("version", "0.4")
        if zarrV3 and not os.path.isfile(os.path.join(labelsDir, "zarr.json")):
            with open(os.path.join(labelsDir, "zarr.json"), "w", encoding="utf-8") as fp:
                json.dump({"zarr_format": 3, "node_type": "group", "attributes": {"ome": {"version": version}}}, fp)
        elif not zarrV3 and not os.path.isfile(os.path.join(labelsDir, ".zgroup")):
            with open(os.path.join(labelsDir, ".zgroup"), "w", encoding="utf-8") as fp:
                json.dump({"zarr_format": 2}, fp)

        def update(ome):
            names = [n for n in ome.get("labels", []) if n != name]
            ome["labels"] = [*names, name]

        cls._updateGroupMetadata(labelsDir, update)


class AutoRefiner:
    """Reloads the block each observed slice view shows once that view has been still.

    Every change of a slice node restarts a single timer, so nothing loads while the user
    pans or zooms. On timeout, only the views whose bounds changed are refined.
    """

    def __init__(self, path, sliceViewNames, delayMs, maxBytes):
        self.path = path
        self.sliceViewNames = list(sliceViewNames)
        self.maxBytes = maxBytes
        self.lastBounds = {}
        self.busy = False
        self.pending = False
        self.refreshCount = 0
        self.timer = qt.QTimer()
        self.timer.setSingleShot(True)
        self.timer.setInterval(int(delayMs))
        self.timer.timeout.connect(self.refresh)
        self.observers = []
        layoutManager = slicer.app.layoutManager()
        for viewName in self.sliceViewNames:
            sliceWidget = layoutManager.sliceWidget(viewName)
            if sliceWidget is None:
                raise ValueError(f"No slice view named '{viewName}'")
            sliceNode = sliceWidget.mrmlSliceNode()
            self.observers.append(
                (sliceNode, sliceNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onSliceModified))
            )
        self.sceneObserverTag = slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.EndCloseEvent, self.onSceneClosed)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        for node, tag in self.observers:
            node.RemoveObserver(tag)
        self.observers = []
        if self.sceneObserverTag is not None:
            slicer.mrmlScene.RemoveObserver(self.sceneObserverTag)
            self.sceneObserverTag = None

    def onSceneClosed(self, caller=None, event=None):
        OMEZarrLogic.stopAutoRefine(self.path)

    def onSliceModified(self, caller=None, event=None):
        if self.busy:
            self.pending = True  # the application stays responsive while a block loads
        else:
            self.timer.start()

    @staticmethod
    def tolerance(bounds):
        """A view counts as still when it moved by less than 1% of its smallest extent."""
        return 0.01 * min(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    def idle(self):
        return not self.busy and not self.timer.isActive()

    def viewBudget(self):
        return (self.maxBytes or OMEZarrLogic.maxBytesFromSettings()) // max(1, len(self.sliceViewNames))

    def refresh(self):
        if self.busy:
            self.timer.start()
            return
        self.busy = True
        try:
            for viewName in self.sliceViewNames:
                try:
                    bounds = OMEZarrLogic.sliceViewRasBounds(viewName)
                except ValueError:
                    continue
                previous = self.lastBounds.get(viewName)
                if previous is not None and np.allclose(bounds, previous, rtol=0.0, atol=self.tolerance(bounds)):
                    continue
                self.lastBounds[viewName] = bounds
                try:
                    OMEZarrLogic.refineView(self.path, viewName, maxBytes=self.viewBudget())
                    self.refreshCount += 1
                except (ValueError, InterruptedError) as e:
                    logging.debug(f"Auto-refine of {viewName} skipped: {e}")
                except Exception:  # noqa: BLE001 - never let a timer callback raise into Qt
                    logging.exception(f"Auto-refine of {viewName} failed")
        finally:
            self.busy = False
            if self.pending:
                self.pending = False
                self.timer.start()


#
# Streaming
#


class HttpGate:
    """How many requests one server gets at a time, shared by every reader of every store on it.

    It starts at STREAM_READERS. A busy answer (429, 5xx) or a lost connection halves it, at most
    once per round of requests, and holds new requests back for the server's Retry-After, or for
    a delay that doubles while the server stays busy; each answer then adds back about one request
    per round. This is TCP's congestion rule: Slicers sharing a busy server settle at what it can
    serve instead of all retrying at once.
    """

    def __init__(self, maximum):
        self.maximum = float(maximum)
        self.limit = float(maximum)
        self.inFlight = 0
        self.pausedUntil = 0.0
        self.lastDecrease = 0.0
        self.busyRounds = 0  # consecutive rounds the server was busy: sets the delay without Retry-After
        self.busyAnswers = 0
        self.starts = collections.deque()  # start times within the last HTTP_WINDOW_S
        self.condition = threading.Condition()

    def acquire(self):
        """Wait for a turn; returns when it started."""
        with self.condition:
            while True:
                if HttpClient.shutdown.is_set():
                    raise InterruptedError("Slicer is closing")
                now = time.monotonic()
                while self.starts and now - self.starts[0] >= HTTP_WINDOW_S:
                    self.starts.popleft()
                wait = self.pausedUntil - now
                if len(self.starts) >= HTTP_WINDOW_REQUESTS:  # paced below the server's request limit
                    wait = max(wait, self.starts[0] + HTTP_WINDOW_S - now)
                if wait <= 0 and self.inFlight < int(self.limit):
                    self.inFlight += 1
                    self.starts.append(now)
                    return now
                self.condition.wait(min(1.0, wait) if wait > 0 else 1.0)

    def release(self, started, outcome="ok", retryAfter=None):
        """``outcome``: "ok", "busy" (the server is overloaded) or "neutral" (a closed kept-alive connection)."""
        import random

        with self.condition:
            self.inFlight -= 1
            now = time.monotonic()
            if outcome == "busy":
                self.busyAnswers += 1
                if started >= self.lastDecrease:  # requests already in flight do not count again
                    self.limit = max(1.0, self.limit / 2)
                    self.lastDecrease = now
                    self.busyRounds += 1
                delay = retryAfter if retryAfter is not None else min(30.0, 0.5 * 2 ** (self.busyRounds - 1))
                delay = min(HTTP_MAX_WAIT_S, delay) * (1.0 if retryAfter is not None else random.uniform(0.5, 1.5))
                self.pausedUntil = max(self.pausedUntil, now + delay)
            elif outcome == "ok":
                self.busyRounds = 0
                self.limit = min(self.maximum, self.limit + 1.0 / self.limit)
            self.condition.notify_all()

    def state(self):
        """(requests allowed at a time, seconds until requests resume) for the status line."""
        with self.condition:
            return int(self.limit), max(0.0, self.pausedUntil - time.monotonic())


class HttpClient:
    """Kept-alive HTTP(S) connections to one server (one per thread), behind that server's HttpGate."""

    _clients = {}
    _lock = threading.Lock()
    shutdown = threading.Event()  # set when Slicer quits: waiting readers give up
    BUSY = (429, 500, 502, 503, 504)

    @classmethod
    def forUrl(cls, url):
        import urllib.parse

        parts = urllib.parse.urlsplit(url)
        key = (parts.scheme, parts.hostname, parts.port)
        with cls._lock:
            client = cls._clients.get(key)
            if client is None:
                client = cls._clients[key] = cls(*key)
            return client

    def __init__(self, scheme, host, port):
        self.scheme, self.host, self.port = scheme, host, port
        self.gate = HttpGate(STREAM_READERS)
        self.local = threading.local()

    def connection(self, fresh=False):
        import http.client
        import ssl

        if fresh or getattr(self.local, "connection", None) is None:
            if getattr(self.local, "connection", None) is not None:
                self.local.connection.close()
            if self.scheme == "https":
                self.local.connection = http.client.HTTPSConnection(
                    self.host, self.port, timeout=60, context=ssl.create_default_context()
                )
            else:
                self.local.connection = http.client.HTTPConnection(self.host, self.port, timeout=60)
        return self.local.connection

    @staticmethod
    def retryAfter(response):
        """Seconds the server asks to wait (Retry-After, in seconds or as a date), or None."""
        value = response.getheader("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            pass
        try:
            import email.utils

            return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError):
            return None

    def get(self, path, headers=None):
        """(status, response, body) of a GET. Busy answers and lost connections are retried
        after the gate's delay, up to HTTP_ATTEMPTS times; then OSError."""
        import http.client

        fresh, error = False, None
        for attempt in range(HTTP_ATTEMPTS):
            started = self.gate.acquire()
            try:
                connection = self.connection(fresh)
                connection.request("GET", path, headers=headers or {})
                response = connection.getresponse()
                body = response.read()
            except (OSError, http.client.HTTPException) as e:
                error = e
                # The first failure on a kept-alive connection is usually the server having closed it.
                self.gate.release(started, "neutral" if attempt == 0 and not fresh else "busy")
                fresh = True
                continue
            if response.status in self.BUSY:
                error = OSError(f"HTTP {response.status}")
                self.gate.release(started, "busy", self.retryAfter(response))
                continue
            self.gate.release(started)
            return response.status, response, body
        raise OSError(f"{self.host} did not answer {path} after {HTTP_ATTEMPTS} tries: {error}")


class DiskCache:
    """What remote stores sent, kept on disk between sessions; least recently used dropped first.

    A value is the bytes as the server sent them (still compressed). An empty value records that
    the server has no such object: an all-background chunk of a plain store, or a missing shard.
    """

    _instance = None

    @classmethod
    def instance(cls):
        """The cache the settings ask for, or None when it is off. The settings are read on the
        GUI thread; other threads get the cache as it was last set up there."""
        if threading.current_thread() is not threading.main_thread():
            return cls._instance if cls._instance is not None and cls._instance.limit > 0 else None
        limit = Settings.get(Settings.DISK_CACHE, DISK_CACHE_MIB) << 20
        directory = Settings.get(Settings.DISK_CACHE_DIR, "") or os.path.join(slicer.app.cachePath, "OMEZarr")
        if cls._instance is None or cls._instance.root != directory:
            cls._instance = cls(directory)
        cls._instance.limit = max(0, limit)
        return cls._instance if limit > 0 else None

    def __init__(self, root):
        self.root = root
        self.limit = DISK_CACHE_MIB << 20
        self.lock = threading.Lock()
        self.size = 0
        self.measured = False
        self.trimming = False
        threading.Thread(target=self.measure, name="OMEZarr cache", daemon=True).start()

    def entries(self):
        for directory, _subdirectories, files in os.walk(self.root):
            for name in files:
                path = os.path.join(directory, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                yield path, stat.st_size, stat.st_mtime

    def measure(self):
        total = sum(size for _path, size, _mtime in self.entries())
        with self.lock:
            self.size += total
            self.measured = True
        self.trimIfNeeded()

    def path(self, key):
        import hashlib

        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(self.root, digest[:2], digest[2:])

    def get(self, key):
        """The bytes kept under ``key``, or None."""
        path = self.path(key)
        try:
            with open(path, "rb") as file:
                data = file.read()
        except OSError:
            return None
        try:
            os.utime(path)  # most recently used
        except OSError:
            pass
        return data

    def put(self, key, data):
        path = self.path(key)
        temporary = f"{path}.{threading.get_ident()}.tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(temporary, "wb") as file:
                file.write(data)
            os.replace(temporary, path)  # readers never see half a file
        except OSError:
            logging.debug(f"OME-Zarr: could not write {path} to the disk cache", exc_info=True)
            return
        with self.lock:
            self.size += len(data)
        self.trimIfNeeded()

    def trimIfNeeded(self):
        with self.lock:
            if not self.measured or self.trimming or self.size <= self.limit:
                return
            self.trimming = True
        threading.Thread(target=self.trim, name="OMEZarr cache", daemon=True).start()

    def trim(self):
        """Drop the least recently used entries down to 90% of the limit."""
        try:
            entries = sorted(self.entries(), key=lambda entry: entry[2])
            total = sum(size for _path, size, _mtime in entries)
            for path, size, _mtime in entries:
                if total <= 0.9 * self.limit:
                    break
                try:
                    os.remove(path)
                    total -= size
                except OSError:
                    pass
            with self.lock:
                self.size = total
        finally:
            with self.lock:
                self.trimming = False

    def clear(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)
        with self.lock:
            self.size = 0


class ChunkReader:
    """Reads the chunks of one remote Zarr v3 array straight over HTTP(S), plain or sharded.

    ngff-zarr's remote reader fetches the whole shard for every inner chunk it reads (a 3D view of
    neotoma moved 10.9 GB for 2012 requests, against 108 MB unsharded). Here each shard's index is
    read once (a suffix range of the shard file) and kept; an inner chunk the index marks empty
    costs no request at all, and the others are read by byte range. Requests go through the
    server's HttpClient (kept-alive connections, fewer at a time while the server is busy), and
    what the server sends is kept in the DiskCache:

    * a shard's index, with the shard's ETag; it is checked once per session by a conditional
      request (304 when unchanged), so a replaced shard is never served from the cache;
    * an inner chunk, under its shard's ETag, so it needs no check of its own;
    * a plain chunk, or the server's "no such chunk", under the ETag and date of the array's
      zarr.json: rewriting a store rewrites it.

    Handled: z/y/x arrays, default chunk keys, bytes + zstd/gzip/blosc, shard index at the end.
    Anything else returns None from ``forArray`` and is read through ngff-zarr.
    """

    EMPTY = 2**64 - 1
    COMPRESSORS = ("zstd", "gzip", "blosc")

    @classmethod
    def forArray(cls, storeUrl, arrayPath, cache=None):
        import urllib.parse

        url = f"{str(storeUrl).rstrip('/')}/{arrayPath.strip('/')}"
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or parts.query:
            return None
        try:
            client = HttpClient.forUrl(url)
            base = parts.path.rstrip("/")
            status, response, body = client.get(f"{base}/zarr.json")
            if status != 200:
                return None
            meta = json.loads(body)
            if meta.get("zarr_format") != 3 or len(meta.get("shape") or ()) != 3:
                return None
            if tuple(meta.get("dimension_names") or ()) not in ((), ("z", "y", "x")):
                return None
            if (meta.get("chunk_grid") or {}).get("name") != "regular":
                return None
            encoding = meta.get("chunk_key_encoding") or {}
            if encoding.get("name", "default") != "default":
                return None
            separator = (encoding.get("configuration") or {}).get("separator", "/")
            gridShape = meta["chunk_grid"]["configuration"]["chunk_shape"]
            codecs = meta.get("codecs") or []
            shardShape, checksum = None, False
            if len(codecs) == 1 and codecs[0].get("name") == "sharding_indexed":
                config = codecs[0]["configuration"]
                indexNames = [c.get("name") for c in config.get("index_codecs") or []]
                if config.get("index_location", "end") != "end" or indexNames not in (["bytes"], ["bytes", "crc32c"]):
                    return None
                shardShape, checksum = gridShape, "crc32c" in indexNames
                innerShape, codecs = config["chunk_shape"], config.get("codecs") or []
            else:
                innerShape = gridShape
            names = [c.get("name") for c in codecs]
            if not names or names[0] != "bytes" or any(name not in cls.COMPRESSORS for name in names[1:]):
                return None
            endian = ((codecs[0].get("configuration") or {}).get("endian") or "little")[0]
            dtype = np.dtype(meta["data_type"]).newbyteorder("<" if endian == "l" else ">")
            fill = np.array(meta.get("fill_value") or 0, dtype=dtype.newbyteorder("="))
            generation = f"{response.getheader('ETag') or ''}|{response.getheader('Last-Modified') or ''}"
            return cls(client, base, innerShape, dtype, names[1:], separator, shardShape, checksum, fill, generation, cache)
        except Exception:  # noqa: BLE001 - fall back to ngff-zarr
            logging.debug(f"OME-Zarr: {url} is read through ngff-zarr", exc_info=True)
            return None

    def __init__(self, client, base, innerShape, dtype, compressors, separator, shardShape, checksum, fill, generation, cache):
        import numcodecs

        self.client = client
        self.base = base
        self.innerShape = tuple(int(n) for n in innerShape)
        self.dtype = dtype
        self.fill = fill
        codecs = {"zstd": numcodecs.Zstd, "gzip": numcodecs.GZip, "blosc": numcodecs.Blosc}
        self.decoders = [codecs[name]() for name in reversed(compressors)]
        self.separator = separator
        self.sharded = shardShape is not None
        self.perShard = tuple(int(s) // int(i) for s, i in zip(shardShape, innerShape)) if self.sharded else None
        self.indexBytes = int(np.prod(self.perShard)) * 16 + (4 if checksum else 0) if self.sharded else 0
        self.generation = generation if generation.strip("|") else None  # no version: nothing is cached
        self.cache = cache
        self.indexes = {}  # shard key -> (n, 2) uint64 array of (offset, nbytes), or None for a missing shard
        self.versions = {}  # shard key -> ETag (or date and size) the cached chunks of that shard are kept under
        self.fetching = {}  # shard key -> Event set once its index is in ``indexes`` (one fetch per shard)
        self.lock = threading.Lock()
        self.local = threading.local()  # .fromCache: whether this thread's last read needed no transfer
        self.requests = 0
        self.indexReads = 0
        self.bytesRead = 0
        self.cacheHits = 0

    def objectPath(self, key):
        return f"{self.base}/c{self.separator}" + self.separator.join(str(k) for k in key)

    def request(self, path, headers=None):
        status, response, body = self.client.get(path, headers)
        with self.lock:
            self.requests += 1
            self.bytesRead += len(body)
        return status, response, body

    def fetch(self, path, cacheKey, byteRange=None):
        """The bytes of an object, or of ``byteRange`` of it, from the disk cache or the server;
        b"" when the server has no such object."""
        if self.cache is not None and cacheKey is not None:
            data = self.cache.get(cacheKey)
            if data is not None:
                self.local.fromCache = True
                with self.lock:
                    self.cacheHits += 1
                return data
        status, _response, body = self.request(path, {"Range": f"bytes={byteRange}"} if byteRange else None)
        if status == 404:
            body = b""
        elif status == 200 and byteRange:  # a server that ignores Range
            first, last = byteRange.split("-")
            body = body[-int(last) :] if first == "" else body[int(first) : int(last) + 1]
        elif status not in (200, 206):
            raise OSError(f"HTTP {status} reading {path}")
        if self.cache is not None and cacheKey is not None:
            self.cache.put(cacheKey, body)
        return body

    def shardIndex(self, shardKey):
        """The shard's index, read once: readers needing it meanwhile wait for that one read."""
        while True:
            with self.lock:
                if shardKey in self.indexes:
                    return self.indexes[shardKey]
                event = self.fetching.get(shardKey)
                if event is None:
                    event = self.fetching[shardKey] = threading.Event()
                    break  # this thread reads it
            event.wait(60)
        try:
            index, version = self.readIndex(shardKey)
            with self.lock:
                self.indexes[shardKey], self.versions[shardKey] = index, version
            return index
        finally:
            with self.lock:
                self.fetching.pop(shardKey, None)
            event.set()

    def readIndex(self, shardKey):
        """(index, version) of a shard; the cached index is used when the server says the shard is unchanged."""
        path = self.objectPath(shardKey)
        cacheKey = f"index|{path}"
        cached = self.cache.get(cacheKey) if self.cache is not None else None
        headers = {"Range": f"bytes=-{self.indexBytes}"}
        cachedVersion, cachedIndex = None, None
        if cached:  # version, newline, index bytes
            cachedVersion, _newline, cachedIndex = cached.partition(b"\n")
            headers["If-None-Match"] = cachedVersion.decode("utf-8")
        status, response, body = self.request(path, headers)
        with self.lock:
            self.indexReads += 1
        if status == 304 and cachedIndex is not None:
            return self.parseIndex(cachedIndex), cachedVersion.decode("utf-8")
        if status == 404:
            return None, None
        if status not in (200, 206):
            raise OSError(f"HTTP {status} reading the index of shard {shardKey}")
        raw = body[-self.indexBytes :]
        version = response.getheader("ETag") or ""
        if not version and response.getheader("Last-Modified"):
            version = f"{response.getheader('Last-Modified')} {response.getheader('Content-Range') or len(body)}"
        if version and self.cache is not None:
            self.cache.put(cacheKey, version.encode("utf-8") + b"\n" + raw)
        return self.parseIndex(raw), version or None

    def parseIndex(self, raw):
        entries = int(np.prod(self.perShard))
        return np.frombuffer(raw[: entries * 16], "<u8").reshape(entries, 2)

    def read(self, key):
        """The chunk ``key`` (z, y, x) as a full chunk array, or None when it holds only the fill value."""
        self.local.fromCache = False
        if not self.sharded:
            path = self.objectPath(key)
            data = self.fetch(path, f"{self.generation}|{path}" if self.generation else None)
        else:
            shardKey = tuple(k // p for k, p in zip(key, self.perShard))
            index = self.shardIndex(shardKey)
            if index is None:
                return None
            inner = np.ravel_multi_index(tuple(k % p for k, p in zip(key, self.perShard)), self.perShard)
            offset, nbytes = (int(v) for v in index[inner])
            if offset == self.EMPTY and nbytes == self.EMPTY:
                return None
            path, version = self.objectPath(shardKey), self.versions.get(shardKey)
            cacheKey = f"{version}|{path}|{offset}+{nbytes}" if version else None
            data = self.fetch(path, cacheKey, f"{offset}-{offset + nbytes - 1}")
        if not data:
            return None
        for decoder in self.decoders:
            data = decoder.decode(data)
        return np.frombuffer(data, self.dtype).reshape(self.innerShape)


class LevelChunks:
    """Chunk grid of one level, for one time point and channel, in (z, y, x) order."""

    def __init__(self, image, timeIndex, channelIndex, reader=None):
        self.image = image
        self.timeIndex = timeIndex
        self.channelIndex = channelIndex
        self.reader = reader  # ChunkReader of a remote store; ngff-zarr reads the others
        dims = list(image.dims)
        self.edges = [
            np.concatenate([[0], np.cumsum(image.data.chunks[dims.index(d)])]).astype(int) for d in ("z", "y", "x")
        ]
        self.shape = tuple(int(edges[-1]) for edges in self.edges)

    def keys(self, region=None):
        """Chunk indices intersecting ``region`` ((start, stop) per z, y, x); all of them without one."""
        ranges = []
        for axis, edges in enumerate(self.edges):
            start, stop = region[axis] if region else (0, self.shape[axis])
            first = int(np.searchsorted(edges, start, side="right")) - 1
            last = int(np.searchsorted(edges, stop - 1, side="right")) - 1
            ranges.append(range(first, last + 1))
        return list(itertools.product(*ranges))

    def bounds(self, key):
        return [(int(self.edges[axis][k]), int(self.edges[axis][k + 1])) for axis, k in enumerate(key)]

    def read(self, key):
        if self.reader is not None:
            bounds = self.bounds(key)
            block = self.reader.read(key)
            shape = [stop - start for start, stop in bounds]
            if block is None:
                return np.full(shape, self.reader.fill, self.reader.fill.dtype)
            return block[: shape[0], : shape[1], : shape[2]].astype(block.dtype.newbyteorder("="), copy=False)
        region = dict(zip(("z", "y", "x"), self.bounds(key)))
        sub, _addZ = OMEZarrLogic.spatialDaskArray(self.image, self.timeIndex, self.channelIndex, region)
        return np.asarray(sub.compute(scheduler="synchronous"))


class Streamer:
    """Shows a store at once and reads it in behind the slice views.

    The volume node starts with the coarsest level. Each observed slice view then gets, as
    its foreground, the plane it shows at the coarsest level whose voxels are no larger than
    a screen pixel: only the chunks that plane crosses are read, before anything else. Behind
    that, the whole target level is read chunk by chunk into the buffer the node will use
    (a chunk a view already read is not read again). When the buffer is complete the node
    switches to it, and a view keeps a block of its own only where it asks for more detail
    than the target level has.
    """

    RETRIES = 3

    def __init__(
        self,
        path,
        node,
        multiscales,
        targetLevel,
        sliceViewNames,
        timeIndex=0,
        readers=None,
        cacheBytes=STREAM_CACHE_BYTES,
    ):
        from vtk.util import numpy_support

        self.path = path
        self.node = node
        self.target = targetLevel
        self.shownLevel = int(node.GetAttribute("OMEZarr.Level"))
        self.sliceViewNames = list(sliceViewNames)
        images = multiscales.images
        # The readers ask the server for each level's metadata once: off the GUI thread.
        chunkReaders = runResponsive(lambda: [OMEZarrLogic.chunkReader(multiscales, level) for level in range(len(images))])
        self.levels = [LevelChunks(image, timeIndex, 0, reader) for image, reader in zip(images, chunkReaders)]
        self.client = next((reader.client for reader in chunkReaders if reader is not None), None)
        self.ijkToRas = [OMEZarrLogic.ijkToRasMatrix(image)[0] for image in images]
        self.spacing = [float(np.linalg.norm(m[:3, :3], axis=0).max()) for m in self.ijkToRas]
        self.dtype = OMEZarrLogic.vtkCompatibleDtype(images[0].data.dtype)
        self.cacheBytes = cacheBytes
        self.cache = collections.OrderedDict()  # (level, key) -> array, for levels other than the target
        self.cachedBytes = 0
        self.lock = threading.Lock()
        self.wake = threading.Condition(self.lock)  # readers holding back a 3D or background read
        # Reads waiting, as heaps of (priority, order, level, key), one per kind (lock held):
        # "view" 0 slice views; "volume" 0.5 3D view, 0.75 3D fallback; "fill" 1 the level read whole.
        self.queues = {"view": [], "volume": [], "fill": []}
        self.order = itertools.count()
        self.inFlight = set()
        self.kindInFlight = collections.Counter()  # reads in flight per kind
        self.backgroundInFlight = 0  # reads in flight for anything but the slice views
        self.lastBackground = None  # kind that started the last 3D-or-fill read: the other goes next on a tie
        self.viewItems = set()  # (level, key) the slice views need now
        self.attempts = collections.Counter()
        self.failed = set()
        self.wanted = set()  # (level, key) the views need now
        self.views = {}  # view name -> {"level", "region", "keys", "shown"}
        self.overlays = {}  # view name -> volume node
        self.stopped = False
        self.complete = False
        self.lastPercent = -1
        # The coarsest level stays in memory: the 3D view falls back to it.
        self.contextLevel = len(images) - 1
        if self.shownLevel == self.contextLevel:
            self.contextArray = slicer.util.arrayFromVolume(node).copy()
        else:
            grid = self.levels[-1]
            self.contextArray = np.empty(grid.shape, self.dtype)
            if grid.reader is not None:
                OMEZarrLogic.readChunks(grid, tuple((0, n) for n in grid.shape), self.contextArray)
            else:
                coarsest, _addZ = OMEZarrLogic.spatialDaskArray(images[-1], timeIndex, 0)
                OMEZarrLogic.computeArray(coarsest.astype(self.dtype), self.contextArray)
        self.contextImage = None  # a finer whole level for the 3D fallback, once it is in memory
        self.contextRequest = None  # whole level being read to become the 3D fallback
        self.maxTextureDim, self.maxTextureBytes = STREAM_3D_MAX_DIM, STREAM_3D_MAX_BYTES
        self.volume3D = None  # volume node the 3D view renders
        self.request3D = None  # {"level", "region", "keys", "shown"} for the 3D view
        self.shown3D = None  # (level, region) in the 3D node
        self.cameraObservers = []
        self.reusedChunks = 0  # chunks copied from the previous 3D texture instead of read again
        self.opacityCache = None  # (MTimes, opacity per value bin, low value, bins per value, unit distance mm)
        self.reads = collections.deque(maxlen=4096)  # (end time, decoded bytes, seconds, from disk) per chunk read
        self.lastStatus = 0.0
        self.idleShown = False

        # The volume sharpens level by level, from the one shown down to the target: each coarser
        # level costs an eighth of the next. The level wholly in memory serves the views' chunks.
        self.residentLevel = self.shownLevel
        self.residentArray = slicer.util.arrayFromVolume(node)
        self.downloaded = {self.shownLevel}  # levels read whole at some point
        self.empty = set()  # (level, key) of chunks read as all zeros: kept as this mark, never read again
        self.partial = collections.defaultdict(set)  # level -> chunks read for the views
        self.startFill(max(targetLevel, self.shownLevel - 1))

        self.observers = []
        layoutManager = slicer.app.layoutManager()
        for viewName in self.sliceViewNames:
            sliceWidget = layoutManager.sliceWidget(viewName) if layoutManager else None
            if sliceWidget is not None:
                sliceNode = sliceWidget.mrmlSliceNode()
                self.observers.append(
                    (sliceNode, sliceNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onViewChanged))
                )
        display = node.GetDisplayNode()
        if display is not None:
            self.observers.append((display, display.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onDisplayChanged)))
        self.observers.append(
            (slicer.mrmlScene, slicer.mrmlScene.AddObserver(slicer.vtkMRMLScene.EndCloseEvent, self.onSceneClosed))
        )

        self.threads = [threading.Thread(target=self.readLoop, daemon=True) for _ in range(readers or STREAM_READERS)]
        for thread in self.threads:
            thread.start()
        self.viewTimer = qt.QTimer()
        self.viewTimer.setSingleShot(True)
        self.viewTimer.setInterval(50)
        self.viewTimer.timeout.connect(self.updateViews)
        self.pollTimer = qt.QTimer()
        self.pollTimer.setInterval(100)
        self.pollTimer.timeout.connect(self.poll)
        self.cameraTimer = qt.QTimer()
        self.cameraTimer.setSingleShot(True)
        self.cameraTimer.setInterval(STREAM_3D_SETTLE_MS)
        self.cameraTimer.timeout.connect(self.update3D)
        self.pollTimer.start()
        self.viewTimer.start()

    def startFill(self, level):
        """Read ``level`` whole, in the background, into the image the volume node will switch to."""
        from vtk.util import numpy_support

        shape = self.levels[level].shape
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(shape[2], shape[1], shape[0])
        imageData.AllocateScalars(numpy_support.get_vtk_array_type(self.dtype), 1)
        grid = self.levels[level]
        center = np.array(shape) / 2.0
        keys = sorted(grid.keys(), key=lambda key: float(np.linalg.norm([np.mean(b) for b in grid.bounds(key)] - center)))
        with self.lock:
            self.fillLevel = level
            self.targetImageData = imageData
            self.targetArray = numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars()).reshape(shape)
            self.targetKeys = keys
            self.targetHave = set()
            for key in keys:  # chunks the views already read are not read again
                cached = self.cache.get((level, key))
                if cached is not None or (level, key) in self.empty:
                    (z0, z1), (y0, y1), (x0, x1) = grid.bounds(key)
                    self.targetArray[z0:z1, y0:y1, x0:x1] = 0 if cached is None else cached
                    self.targetHave.add(key)
            for key in keys:
                if key not in self.targetHave:
                    self.push(1, level, key)
            self.wake.notify_all()

    # -- reading (worker threads) --

    @staticmethod
    def kindOf(priority):
        return "view" if priority == 0 else "volume" if priority < 1 else "fill"

    def push(self, priority, level, key):
        """Queue a read (lock held)."""
        heapq.heappush(self.queues[self.kindOf(priority)], (priority, next(self.order), level, key))

    def has(self, item):
        level, key = item
        if level == self.fillLevel:
            return key in self.targetHave
        return level == self.residentLevel or item in self.cache or item in self.empty

    def viewsWaiting(self):
        """Whether a slice view still lacks a chunk, read or not yet (lock held)."""
        return any(not self.has(item) for item in self.viewItems)

    def nextRead(self):
        """The next read to start, or None (lock held). A slice view's always comes first. The 3D
        view's and the background fill's start only while no view waits, within
        STREAM_BACKGROUND_READERS, and take turns: neither holds back the other."""
        if self.queues["view"]:
            return heapq.heappop(self.queues["view"])
        if self.backgroundInFlight >= STREAM_BACKGROUND_READERS or self.viewsWaiting():
            return None
        ready = [kind for kind in ("volume", "fill") if self.queues[kind]]
        if not ready:
            return None
        kind = min(ready, key=lambda k: (self.kindInFlight[k], k == self.lastBackground))
        self.lastBackground = kind
        return heapq.heappop(self.queues[kind])

    def readLoop(self):
        while not self.stopped:
            with self.lock:
                entry = self.nextRead()
                if entry is None:
                    self.wake.wait(0.2)
                    continue
                priority, _order, level, key = entry
                item = (level, key)
                if self.has(item) or item in self.inFlight or item in self.failed:
                    continue
                if priority < 1 and item not in self.wanted:
                    continue  # the view has moved on
                if priority >= 1 and level != self.fillLevel:
                    continue  # a level already filled
                forView = priority == 0
                kind = self.kindOf(priority)
                self.inFlight.add(item)
                self.kindInFlight[kind] += 1
                if not forView:
                    self.backgroundInFlight += 1
            started = time.monotonic()
            grid = self.levels[level]
            try:
                block = self.readChunk(level, key, forView)
            except Exception:  # noqa: BLE001 - retried, then reported and left empty
                block = None
                if not self.stopped:
                    logging.warning(f"OME-Zarr streaming: reading chunk {key} of level {level} failed", exc_info=True)
            ended = time.monotonic()  # the read alone, not the wait for the lock below
            fromCache = bool(grid.reader is not None and getattr(grid.reader.local, "fromCache", False))
            with self.lock:
                self.inFlight.discard(item)
                self.kindInFlight[kind] -= 1
                if not forView:
                    self.backgroundInFlight -= 1
                self.wake.notify_all()
                self.reads.append((ended, block.nbytes if block is not None else 0, ended - started, fromCache))
                if block is None:
                    self.attempts[item] += 1
                    if self.attempts[item] < self.RETRIES:
                        self.push(priority, level, key)
                        continue
                    self.failed.add(item)
                    logging.error(f"OME-Zarr streaming: chunk {key} of level {level} left empty")
                    block = np.zeros([stop - start for start, stop in self.levels[level].bounds(key)], self.dtype)
                self.store(item, block)

    def readChunk(self, level, key, forView):
        """One chunk, read off the GUI thread; ``forView``: for a slice view (tests watch this)."""
        return self.levels[level].read(key)

    def store(self, item, block):
        """Keep a chunk that was read (lock held)."""
        if self.stopped:
            return
        level, key = item
        empty = not block.any()  # most chunks of a specimen are background
        if empty:
            self.empty.add(item)
        if level == self.fillLevel:
            (z0, z1), (y0, y1), (x0, x1) = self.levels[level].bounds(key)
            self.targetArray[z0:z1, y0:y1, x0:x1] = 0 if empty else block
            self.targetHave.add(key)
            return
        self.partial[level].add(key)
        if not empty:
            self.cache[item] = block
            self.cachedBytes += block.nbytes
            self.trimCache()

    def trimCache(self):
        """Drop the least recently used chunks no view needs now, down to the cache size (lock held)."""
        for old in list(self.cache):
            if self.cachedBytes <= self.cacheBytes:
                break
            if old not in self.wanted:
                self.cachedBytes -= self.cache.pop(old).nbytes

    def chunk(self, item):
        """A chunk that was read (lock held); None when it is all zeros."""
        level, key = item
        if item in self.empty and level != self.fillLevel and level != self.residentLevel:
            return None
        if level == self.fillLevel or level == self.residentLevel:
            array = self.targetArray if level == self.fillLevel else self.residentArray
            (z0, z1), (y0, y1), (x0, x1) = self.levels[level].bounds(key)
            return array[z0:z1, y0:y1, x0:x1]
        self.cache.move_to_end(item)
        return self.cache[item]

    # -- views (main thread) --

    def onViewChanged(self, caller=None, event=None):
        if not self.stopped:
            self.viewTimer.start()

    def onDisplayChanged(self, caller=None, event=None):
        for node in self.overlays.values():
            self.copyDisplay(node)

    def onSceneClosed(self, caller=None, event=None):
        OMEZarrLogic.stopStreaming(self.path)

    def planeRegion(self, level, ras):
        """(start, stop) per z, y, x of the voxels of ``level`` around the plane through the RAS ``ras`` corners."""
        ijk = np.linalg.inv(self.ijkToRas[level]) @ ras
        region = []
        for row, size in zip((2, 1, 0), self.levels[level].shape):
            start = max(0, int(np.floor(ijk[row].min())) - 1)  # one voxel more on each side for interpolation
            stop = min(size, int(np.ceil(ijk[row].max())) + 2)
            if stop <= start:
                return None
            region.append((start, stop))
        return tuple(region)

    def viewRequest(self, sliceNode):
        """What a slice view needs: the plane it shows at screen resolution, or None."""
        fov = sliceNode.GetFieldOfView()
        dims = sliceNode.GetDimensions()
        if dims[0] <= 0 or dims[1] <= 0:
            return None
        pixel = min(fov[0] / dims[0], fov[1] / dims[1])
        level = next((lv for lv in reversed(range(len(self.levels))) if self.spacing[lv] <= pixel * 1.001), 0)
        sliceToRas = slicer.util.arrayFromVTKMatrix(sliceNode.GetSliceToRAS())
        corners = np.array([[x, y, 0.0, 1.0] for x in (-fov[0] / 2, fov[0] / 2) for y in (-fov[1] / 2, fov[1] / 2)])
        ras = sliceToRas @ corners.T
        while level < self.shownLevel:
            region = self.planeRegion(level, ras)
            if region is None:
                return None
            keys = self.levels[level].keys(region)
            if len(keys) <= STREAM_MAX_CHUNKS_PER_VIEW:
                return {"level": level, "region": region, "keys": keys, "shown": False}
            level += 1
        return None  # the volume node already shows this much detail

    def updateViews(self):
        if self.stopped:
            return
        layoutManager = slicer.app.layoutManager()
        for viewName in self.sliceViewNames:
            sliceWidget = layoutManager.sliceWidget(viewName) if layoutManager else None
            request = self.viewRequest(sliceWidget.mrmlSliceNode()) if sliceWidget is not None else None
            if request is None:
                self.views.pop(viewName, None)
                self.removeOverlay(viewName)
                continue
            previous = self.views.get(viewName)
            if previous and (previous["level"], previous["region"]) == (request["level"], request["region"]):
                request = previous
            else:
                self.views[viewName] = request
        self.requestChunks()
        self.poll()

    def requestChunks(self):
        """Make what the views need now the only wanted chunks, and queue the missing ones."""
        viewItems = {(r["level"], key) for r in self.views.values() for key in r["keys"]}
        # A 3D texture's chunks are wanted only until it is built: it can be up to STREAM_3D_MAX_BYTES.
        request3D = self.request3D
        volumeItems = (
            {(request3D["level"], key) for key in request3D["keys"] if key not in request3D["reuse"]}
            if request3D and not request3D["shown"]
            else set()
        )
        contextRequest = self.contextRequest
        contextItems = (
            {(contextRequest["level"], key) for key in contextRequest["keys"]}
            if contextRequest and not contextRequest["shown"]
            else set()
        )
        volumeItems |= contextItems
        with self.lock:
            self.wanted = viewItems | volumeItems
            self.viewItems = viewItems
            self.wake.notify_all()  # held-back reads may go once the views have what they need
            self.trimCache()
            for item in self.wanted:
                if not self.has(item) and item not in self.inFlight:
                    self.push(0 if item in viewItems else 0.75 if item in contextItems else 0.5, *item)

    def ready(self, request):
        """Whether every chunk of the request has been read, or can be copied from the texture
        shown before (lock held)."""
        reuse = request.get("reuse", ())
        return all(key in reuse or self.has((request["level"], key)) for key in request["keys"])

    def assemble(self, request, block=None, partial=False):
        """The request's region from its chunks (lock held), into ``block`` when given; None while
        a chunk is missing. The chunks cover the region, so ``block`` needs no initialization.
        ``partial``: the chunks read so far, over the level in memory where the others go."""
        level, region = request["level"], request["region"]
        if not partial and not self.ready(request):
            return None
        grid = self.levels[level]
        if block is None:
            block = self.residentBlock(level, region) if partial else np.zeros([stop - start for start, stop in region], self.dtype)
        for key in request["keys"]:
            if partial and key not in request.get("reuse", ()) and not self.has((level, key)):
                continue
            target, source = [], []
            for (regionStart, regionStop), (chunkStart, chunkStop) in zip(region, grid.bounds(key)):
                start, stop = max(regionStart, chunkStart), min(regionStop, chunkStop)
                target.append(slice(start - regionStart, stop - regionStart))
                source.append(slice(start - chunkStart, stop - chunkStart))
            if key in request.get("reuse", ()):
                oldRegion, oldVoxels = request["reuseFrom"]
                block[tuple(target)] = oldVoxels[
                    tuple(slice(t.start + r[0] - o[0], t.stop + r[0] - o[0]) for t, r, o in zip(target, region, oldRegion))
                ]
            else:
                data = self.chunk((level, key))
                block[tuple(target)] = 0 if data is None else data[tuple(source)]
        return block

    def residentBlock(self, level, region):
        """``region`` of ``level`` sampled from the level wholly in memory, nearest voxel (lock held):
        what a view shows where its own chunks have not arrived yet."""
        shape = [stop - start for start, stop in region]
        levelToResident = np.linalg.inv(self.ijkToRas[self.residentLevel]) @ self.ijkToRas[level]
        linear = levelToResident[:3, :3]
        if not np.allclose(linear, np.diag(np.diag(linear)), atol=1e-6):
            return np.zeros(shape, self.dtype)  # levels not axis-aligned to each other: leave the gaps empty
        indices = []
        for axis, (start, stop) in enumerate(region):  # z, y, x
            ijk = 2 - axis
            index = np.floor(linear[ijk, ijk] * np.arange(start, stop) + levelToResident[ijk, 3] + 0.5).astype(int)
            indices.append(np.clip(index, 0, self.residentArray.shape[axis] - 1))
        return self.residentArray[np.ix_(*indices)].astype(self.dtype, copy=False)

    def copyDisplay(self, node):
        source, display = self.node.GetDisplayNode(), node.GetDisplayNode()
        if source is None or display is None:
            return
        display.SetAutoWindowLevel(False)
        display.SetWindowLevel(source.GetWindow(), source.GetLevel())
        display.SetInterpolate(source.GetInterpolate())
        if source.GetColorNodeID():
            display.SetAndObserveColorNodeID(source.GetColorNodeID())

    def showOverlay(self, viewName, request, block):
        node = self.overlays.get(viewName)
        if node is None or node.GetScene() is None:
            name = slicer.mrmlScene.GenerateUniqueName(f"{self.node.GetName()}_{viewName}")
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
            node.SetHideFromEditors(True)
            node.SetSaveWithScene(False)
            node.SetAttribute("OMEZarr.Path", normalizeStorePath(self.path))
            node.SetAttribute("OMEZarr.Channel", "0")
            node.SetAttribute("OMEZarr.Refined", "1")
            node.SetAttribute("OMEZarr.RefinedView", viewName)
            node.SetAttribute("OMEZarr.Streamed", "1")
            node.CreateDefaultDisplayNodes()
            self.overlays[viewName] = node
        level, region = request["level"], request["region"]
        self.replaceImage(node, block)
        ijkToRas = self.ijkToRas[level].copy()
        ijkToRas[:3, 3] = (ijkToRas @ np.array([region[2][0], region[1][0], region[0][0], 1.0]))[:3]
        node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(ijkToRas))
        node.SetAttribute("OMEZarr.Level", str(level))
        self.copyDisplay(node)
        composite = slicer.app.layoutManager().sliceWidget(viewName).sliceLogic().GetSliceCompositeNode()
        composite.SetForegroundVolumeID(node.GetID())
        composite.SetForegroundOpacity(1.0)

    def removeOverlay(self, viewName):
        node = self.overlays.pop(viewName, None)
        if node is not None and node.GetScene() is not None:
            slicer.mrmlScene.RemoveNode(node)

    def poll(self):
        if self.stopped:
            return
        if self.node.GetScene() is None:  # the volume was deleted
            OMEZarrLogic.stopStreaming(self.path)
            return
        now = time.monotonic()
        for viewName, request in list(self.views.items()):
            if request["shown"]:
                continue
            # The plane sharpens chunk by chunk: a view never keeps showing a place it has left.
            with self.lock:
                have = sum(1 for key in request["keys"] if self.has((request["level"], key)))
                complete = have == len(request["keys"])
                block = None
                if complete:
                    block = self.assemble(request)
                elif have > request.get("drawn", -1) and now - request.get("drawnAt", 0.0) >= STREAM_PARTIAL_S:
                    block = self.assemble(request, partial=True)
            if block is not None:
                self.showOverlay(viewName, request, block)
                request["drawn"], request["drawnAt"] = have, now
                request["shown"] = complete
        request3D = self.request3D
        if request3D and not request3D["shown"]:
            with self.lock:
                ready = self.ready(request3D)
            if ready:
                # Read straight into the new image: a texture can be up to STREAM_3D_MAX_BYTES.
                imageData, voxels = self.newImage([stop - start for start, stop in request3D["region"]], self.dtype)
                with self.lock:
                    self.assemble(request3D, voxels)
                self.show3D(request3D["level"], request3D["region"], imageData)
                request3D["shown"] = True
                request3D["reuseFrom"] = request3D["reuseImage"] = None
                request3D["reuse"] = set()
                self.requestChunks()  # its chunks may now leave the cache
                self.cameraTimer.start()  # a finer level may come next
        contextRequest = self.contextRequest
        if contextRequest and not contextRequest["shown"]:
            with self.lock:
                ready = self.ready(contextRequest)
            if ready:
                imageData, voxels = self.newImage(self.levels[contextRequest["level"]].shape, self.dtype)
                with self.lock:
                    self.assemble(contextRequest, voxels)
                contextRequest["shown"] = True
                self.downloaded.add(contextRequest["level"])
                self.setContext(contextRequest["level"], imageData)
                self.requestChunks()
        if not self.complete and len(self.targetHave) == len(self.targetKeys):
            self.downloaded.add(self.fillLevel)
            if self.fillLevel == self.target:
                self.switchToTarget()
            else:
                level = self.fillLevel
                self.switchToLevel(level)
                self.startFill(level - 1)  # before the views refresh: they poll again
                self.planContext()
                self.updateViews()  # views keep their own planes only where they need more
            return
        self.reportProgress()

    def progress(self):
        return len(self.targetHave) / max(1, len(self.targetKeys))

    def transferRate(self, window=3.0):
        """(chunks/s, decoded MB/s, mean seconds per chunk, share read from the disk cache) over
        the last ``window`` seconds."""
        now = time.monotonic()
        with self.lock:
            recent = [r for r in self.reads if now - r[0] <= window]
        if not recent:
            return 0.0, 0.0, 0.0, 0.0
        return (
            len(recent) / window,
            sum(r[1] for r in recent) / window / 1e6,
            sum(r[2] for r in recent) / len(recent),
            sum(1 for r in recent if r[3]) / len(recent),
        )

    def pendingForViews(self):
        """(chunks the slice views lack, chunks the 3D view lacks, reads in flight)."""
        with self.lock:
            views = sum(1 for item in self.viewItems if not self.has(item))
            volume = sum(1 for item in self.wanted - self.viewItems if not self.has(item))
            return views, volume, len(self.inFlight)

    def statusText(self):
        """One line for the status bar: background level, what the views wait for, and the rate."""
        parts = []
        if not self.complete:
            text = _("level {level} {percent}%").format(level=self.fillLevel, percent=int(100 * self.progress()))
            if self.fillLevel != self.target:
                text += _(" (then down to level {level})").format(level=self.target)
            parts.append(text)
        pendingViews, pendingVolume, inFlight = self.pendingForViews()
        pending = pendingViews + pendingVolume
        if pendingViews:
            parts.append(_("slice views waiting for {count} chunks").format(count=pendingViews))
        if pendingVolume:
            parts.append(_("3D view waiting for {count} chunks").format(count=pendingVolume))
        chunksPerSecond, megabytesPerSecond, secondsPerChunk, cached = self.transferRate()
        if chunksPerSecond:
            text = _("{rate:.0f} chunks/s, {mb:.0f} MB/s decoded, {ms:.0f} ms per chunk, {n} in flight").format(
                rate=chunksPerSecond, mb=megabytesPerSecond, ms=1000 * secondsPerChunk, n=inFlight
            )
            if cached:
                text += _(", {percent}% from the disk cache").format(percent=int(round(100 * cached)))
            parts.append(text)
        if self.client is not None and (pending or not self.complete):
            allowed, paused = self.client.gate.state()
            if paused > 0:
                parts.append(_("server busy: resuming in {seconds:.0f} s").format(seconds=max(1.0, paused)))
            elif allowed < STREAM_READERS:
                parts.append(_("server busy: {count} requests at a time").format(count=allowed))
        if not parts:
            return None
        return _("Streaming {name}: ").format(name=self.node.GetName()) + " · ".join(parts)

    LEVEL_CORNER = 3  # upper right; Slicer's own slice annotations use the other corners

    def levelText(self, level):
        return _("OME-Zarr level {level} · {size:g} µm").format(level=level, size=round(self.spacing[level] * 1000, 1))

    def viewWidgets(self):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return []
        widgets = []
        for viewName in self.sliceViewNames:
            sliceWidget = layoutManager.sliceWidget(viewName)
            if sliceWidget is not None:
                widgets.append((viewName, sliceWidget.sliceView()))
        if layoutManager.threeDViewCount:
            widgets.append(("3D", layoutManager.threeDWidget(0).threeDView()))
        return widgets

    def updateLevelLabels(self, clear=False):
        """Show, in each view's corner, the resolution level it displays."""
        for viewName, view in self.viewWidgets():
            if viewName == "3D":
                text = self.levelText(self.shown3D[0]) if (self.volume3D is not None and self.shown3D) else ""
            else:
                request, overlay = self.views.get(viewName), self.overlays.get(viewName)
                shown = int(overlay.GetAttribute("OMEZarr.Level")) if overlay is not None and overlay.GetScene() else None
                text = self.levelText(shown if shown is not None else self.shownLevel)
                if request is not None and not request["shown"]:
                    text += _(" (loading level {level})").format(level=request["level"])
            if clear:
                text = ""
            try:
                annotation = view.cornerAnnotation()
            except AttributeError:
                continue
            if annotation.GetText(self.LEVEL_CORNER) != text:
                annotation.SetText(self.LEVEL_CORNER, text)
                view.scheduleRender()

    def levelState(self, level):
        """(mark, description) of a level for the module's level table."""
        if level in self.downloaded:
            return "✓", _("Downloaded in full")
        if level == self.fillLevel and not self.complete:
            percent = int(100 * self.progress())
            return f"↓ {percent}%", _("Being read in the background: {percent}%").format(percent=percent)
        read = len(self.partial.get(level, ()))
        if read:
            percent = max(1, int(100 * read / max(1, len(self.levels[level].keys()))))
            return f"◐ {percent}%", _("{count} chunks read for the views ({percent}% of the level)").format(
                count=read, percent=percent
            )
        return "", ""

    def reportProgress(self):
        """Keep the status bar current while anything is being read (it does not time out)."""
        now = time.monotonic()
        if now - self.lastStatus < 0.5:
            return
        self.lastStatus = now
        self.updateLevelLabels()
        widget = OMEZarrLogic.panel
        if widget is not None and samePath(widget.path, self.path):
            widget.updateLevelStatus()
        text = self.statusText()
        if text:
            self.idleShown = False
            slicer.util.showStatusMessage(text, 0)
        elif not self.idleShown:
            self.idleShown = True
            slicer.util.showStatusMessage(
                _("Streaming {name}: level {level} loaded, views up to date").format(
                    name=self.node.GetName(), level=self.target
                ),
                5000,
            )

    def switchToLevel(self, level):
        """Show the level just filled in the volume node; it then serves the views' chunks."""
        display = self.node.GetDisplayNode()
        if display is not None:
            display.SetAutoWindowLevel(False)  # keep the window/level the user sees
        self.node.SetAndObserveImageData(self.targetImageData)
        self.node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(self.ijkToRas[level]))
        self.node.SetAttribute("OMEZarr.Level", str(level))
        with self.lock:
            self.shownLevel = self.residentLevel = level
            self.residentArray = self.targetArray

    def switchToTarget(self):
        self.complete = True
        self.switchToLevel(self.target)
        if self.failed:
            message = _("{count} chunks of {name} could not be read and are shown empty").format(
                count=len(self.failed), name=self.node.GetName()
            )
            logging.error(message)
            slicer.util.showStatusMessage(message, 10000)
        else:
            slicer.util.showStatusMessage(
                _("{name}: level {level} loaded").format(name=self.node.GetName(), level=self.target), 3000
            )
        if self.target == 0 and self.volume3D is None:
            OMEZarrLogic.stopStreaming(self.path)  # nothing finer to show
        else:
            self.planContext()
            self.updateViews()  # views keep blocks only where they need more than the target level

    # -- 3D view (main thread) --

    def cameraNode3D(self):
        layoutManager = slicer.app.layoutManager()
        widget = layoutManager.threeDWidget(0) if layoutManager and layoutManager.threeDViewCount else None
        if widget is None:
            return None, None
        return widget, slicer.modules.cameras.logic().GetViewActiveCameraNode(widget.mrmlViewNode())

    def enable3D(self):
        """Render the store in the first 3D view from a node that holds only what that view needs."""
        if self.volume3D is not None and self.volume3D.GetScene() is not None:
            return self.volume3D
        widget, cameraNode = self.cameraNode3D()
        if cameraNode is None:
            raise ValueError("No 3D view to render in")
        name = slicer.mrmlScene.GenerateUniqueName(f"{self.node.GetName()} (3D)")
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", name)
        node.SetSaveWithScene(False)
        node.SetAttribute("OMEZarr.Path", normalizeStorePath(self.path))
        node.SetAttribute("OMEZarr.Streamed3D", "1")
        self.volume3D = node
        self.detectGpuLimits(widget)
        self.showContext3D()
        self.planContext()
        volumeRenderingLogic = slicer.modules.volumerendering.logic()
        display = volumeRenderingLogic.CreateDefaultVolumeRenderingNodes(node)
        display.SetVisibility(True)
        self.frameCamera(cameraNode)
        self.cameraObservers = [(cameraNode, cameraNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onCameraChanged))]
        self.cameraObservers.append((display, display.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onCameraChanged)))
        self.cameraTimer.start()
        return node

    def frameCamera(self, cameraNode):
        """Fit the specimen in the 3D view, keeping the viewing direction. The level follows the zoom,
        so a view left far away (or zoomed out in parallel projection) would only ever show the coarsest one."""
        low, high = self.regionRasBounds(self.contextLevel, tuple((0, n) for n in self.levels[self.contextLevel].shape))
        center = (low + high) / 2.0
        radius = 0.55 * float(np.linalg.norm(high - low))  # half the diagonal, with a margin
        camera = cameraNode.GetCamera()
        direction = np.array(camera.GetDirectionOfProjection())
        viewUp = camera.GetViewUp()
        distance = radius / np.sin(np.radians(camera.GetViewAngle() / 2.0))
        cameraNode.SetFocalPoint(*center)
        cameraNode.SetPosition(*(center - direction * distance))
        cameraNode.SetViewUp(*viewUp)
        camera.SetParallelScale(radius)
        if hasattr(cameraNode, "ResetClippingRange"):
            cameraNode.ResetClippingRange()

    def disable3D(self, stopWhenDone=True):
        self.cameraTimer.stop()
        for caller, tag in self.cameraObservers:
            caller.RemoveObserver(tag)
        self.cameraObservers = []
        node, self.volume3D = self.volume3D, None
        self.request3D = None
        self.shown3D = None
        if node is not None and node.GetScene() is not None:
            for index in reversed(range(node.GetNumberOfDisplayNodes())):
                display = node.GetNthDisplayNode(index)
                if display is not None:
                    slicer.mrmlScene.RemoveNode(display)
            slicer.mrmlScene.RemoveNode(node)
        if not self.stopped:
            self.requestChunks()
            if stopWhenDone and self.complete and self.target == 0:
                OMEZarrLogic.stopStreaming(self.path)

    def onCameraChanged(self, caller=None, event=None):
        if not self.stopped and self.volume3D is not None:
            self.cameraTimer.start()

    def regionRasBounds(self, level, region):
        corners = np.array(
            [[x - 0.5, y - 0.5, z - 0.5, 1.0] for z in region[0] for y in region[1] for x in region[2]]
        )
        ras = (self.ijkToRas[level] @ corners.T)[:3]
        return ras.min(axis=1), ras.max(axis=1)

    def volumeRequest(self, finest=0):
        """What the 3D view needs: the part of the volume inside the camera's view (and the
        cropping ROI) between the depths it shows, at the coarsest level whose voxels are no larger
        than a screen pixel at the focal point (never finer than ``finest``), within the texture
        limits. None when the context level will do."""
        widget, cameraNode = self.cameraNode3D()
        if cameraNode is None:
            return None
        camera = cameraNode.GetCamera()
        width, height = widget.threeDView().renderWindow().GetSize()
        if width <= 0 or height <= 0:
            return None
        planes = [0.0] * 24
        camera.GetFrustumPlanes(width / height, planes)
        sides = np.array(planes[:16]).reshape(4, 4)  # left, right, bottom, top; depth is not limited
        focal = np.append(camera.GetFocalPoint(), 1.0)
        sides *= np.where(sides @ focal < 0, -1.0, 1.0)[:, None]

        # A lattice over the whole volume, kept where it is in view, then grown by one cell.
        steps = 33
        shape = self.levels[self.contextLevel].shape  # z, y, x
        axes = [np.linspace(-0.5, n - 0.5, steps) for n in shape[::-1]]  # i, j, k
        i, j, k = np.meshgrid(*axes, indexing="ij")
        ijk = np.stack([i.ravel(), j.ravel(), k.ravel(), np.ones(i.size)])
        ras = self.ijkToRas[self.contextLevel] @ ijk
        inside = (sides @ ras >= 0).all(axis=0)
        display = self.volume3D.GetDisplayNode() if self.volume3D is not None else None
        roi = display.GetROINode() if display is not None and display.GetCroppingEnabled() else None
        roiBounds = None
        if roi is not None:
            roiBounds = [0.0] * 6
            roi.GetRASBounds(roiBounds)
            for axis in range(3):
                inside &= (ras[axis] >= roiBounds[2 * axis]) & (ras[axis] <= roiBounds[2 * axis + 1])
        # In depth, only between the nearest voxel the view shows and where its rays turn opaque.
        position = np.array(camera.GetPosition())
        direction = np.array(camera.GetDirectionOfProjection())
        direction /= np.linalg.norm(direction)
        depthRange = self.visibleDepthRange(camera, width, height, roiBounds)
        if depthRange is None:
            return None  # nothing the opacity shows is in view: the context level will do
        near, far = depthRange
        if np.isfinite(near):
            depth = direction @ (ras[:3] - position[:, None])
            cell = float(np.linalg.norm(self.ijkToRas[self.contextLevel][:3, :3] @ (np.array(shape[::-1]) / (steps - 1))))
            inside &= (depth >= near - cell) & (depth <= far + cell)
        inside = inside.reshape(i.shape)
        grown = inside.copy()
        for axis in range(3):  # one cell more on each side, without wrapping around
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis], upper[axis] = slice(0, -1), slice(1, None)
            grown[tuple(lower)] |= inside[tuple(upper)]
            grown[tuple(upper)] |= inside[tuple(lower)]
        if not grown.any():
            return None
        points = ras[:, grown.ravel()]

        if camera.GetParallelProjection():
            pixel = 2.0 * camera.GetParallelScale() / height
        else:
            pixel = 2.0 * camera.GetDistance() * np.tan(np.radians(camera.GetViewAngle() / 2.0)) / height
        level = next((lv for lv in reversed(range(len(self.levels))) if self.spacing[lv] <= pixel * 1.001), 0)
        level = max(level, finest)
        itemSize = np.dtype(self.dtype).itemsize
        for level in range(level, self.contextLevel):
            index = np.linalg.inv(self.ijkToRas[level]) @ points
            region = []
            for row, size in zip((2, 1, 0), self.levels[level].shape):
                start = max(0, int(np.floor(index[row].min())))
                stop = min(size, int(np.ceil(index[row].max())) + 1)
                region.append((start, stop))
            region = tuple(region)
            dims = [stop - start for start, stop in region]
            if min(dims) <= 0:
                return None
            if max(dims) <= self.maxTextureDim and int(np.prod(dims)) * itemSize <= self.maxTextureBytes:
                return {"level": level, "region": region, "keys": self.levels[level].keys(region), "shown": False}
        return None

    def opacityLookup(self):
        """(opacity per value bin, low value, bins per value unit, opacity unit distance in mm) of the
        3D view's scalar opacity over the coarsest level's values; None without a volume rendering.
        Recomputed when the opacity changes."""
        logic = slicer.modules.volumerendering.logic()
        display = logic.GetFirstVolumeRenderingDisplayNode(self.volume3D) if self.volume3D is not None else None
        propertyNode = display.GetVolumePropertyNode() if display is not None else None
        volumeProperty = propertyNode.GetVolumeProperty() if propertyNode is not None else None
        if volumeProperty is None:
            return None
        opacity = volumeProperty.GetScalarOpacity()
        stamp = (opacity.GetMTime(), volumeProperty.GetMTime())
        if self.opacityCache is None or self.opacityCache[0] != stamp:
            low, high = float(self.contextArray.min()), float(self.contextArray.max())
            bins = 4096
            table = np.array([opacity.GetValue(v) for v in np.linspace(low, high, bins)])
            unit = float(volumeProperty.GetScalarOpacityUnitDistance()) or 1.0
            self.opacityCache = (stamp, table, low, (bins - 1) / max(high - low, 1e-12), unit)
        return self.opacityCache[1:]

    def visibleDepthRange(self, camera, width, height, roiBounds):
        """(near, far): distances along the view direction between the nearest voxel the 3D view's
        opacity shows and the deepest point where a ray's accumulated opacity reaches
        STREAM_3D_OPAQUE (or leaves the volume). Rays on a STREAM_3D_RAYS grid over the view are
        marched through the coarsest level, half a voxel at a time, with the opacity corrected for
        that step as the renderer does. None when no ray meets anything shown; (-inf, inf) without
        a volume rendering. Averaging in the coarse level lowers the opacity of thin structures, so
        the depth is overestimated rather than cut short."""
        lookup = self.opacityLookup()
        if lookup is None:
            return -np.inf, np.inf
        table, low, binsPerValue, unit = lookup
        level = len(self.levels) - 1  # contextArray always holds the coarsest level
        voxels = self.contextArray
        rasToIjk = np.linalg.inv(self.ijkToRas[level])

        position = np.array(camera.GetPosition())
        direction = np.array(camera.GetDirectionOfProjection())
        direction /= np.linalg.norm(direction)
        right = np.cross(direction, camera.GetViewUp())
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        u, v = (grid.ravel() for grid in np.meshgrid(np.linspace(-1, 1, STREAM_3D_RAYS), np.linspace(-1, 1, STREAM_3D_RAYS)))
        aspect = width / height
        if camera.GetParallelProjection():
            half = camera.GetParallelScale()
            origins = position + np.outer(u * half * aspect, right) + np.outer(v * half, up)
            rays = np.tile(direction, (u.size, 1))
        else:
            tangent = np.tan(np.radians(camera.GetViewAngle() / 2.0))
            origins = np.tile(position, (u.size, 1))
            rays = direction + np.outer(u * tangent * aspect, right) + np.outer(v * tangent, up)
            rays /= np.linalg.norm(rays, axis=1)[:, None]

        # Where each ray is inside the volume (slab test in the coarsest level's index space).
        originsIjk = (rasToIjk[:3, :3] @ origins.T + rasToIjk[:3, 3:4]).T
        raysIjk = (rasToIjk[:3, :3] @ rays.T).T
        boxLow = np.full(3, -0.5)
        boxHigh = np.array(voxels.shape[::-1], float) - 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (boxLow - originsIjk) / raysIjk
            t1 = (boxHigh - originsIjk) / raysIjk
        enter = np.nanmax(np.where(np.isnan(t0), -np.inf, np.minimum(t0, t1)), axis=1)
        leave = np.nanmin(np.where(np.isnan(t1), np.inf, np.maximum(t0, t1)), axis=1)
        enter = np.maximum(enter, 0.0)
        hits = leave > enter
        if not hits.any():
            return None
        step = 0.5 * float(min(self.spacing[level], *np.linalg.norm(self.ijkToRas[level][:3, :3], axis=0)))
        ts = np.arange(enter[hits].min(), leave[hits].max() + step, step)
        if ts.size > 4000:  # bounds the arrays below (rays x samples)
            ts = np.linspace(ts[0], ts[-1], 4000)
            step = float(ts[1] - ts[0])
        sampleIjk = originsIjk[:, None, :] + raysIjk[:, None, :] * ts[None, :, None]  # rays, samples, ijk
        index = np.rint(sampleIjk).astype(np.int64)
        within = np.all((index >= 0) & (index < np.array(voxels.shape[::-1])), axis=2)
        within &= (ts[None, :] >= enter[:, None]) & (ts[None, :] <= leave[:, None])
        index = np.clip(index, 0, np.array(voxels.shape[::-1]) - 1)
        values = voxels[index[..., 2], index[..., 1], index[..., 0]].astype(np.float32)
        alpha = table[np.clip(((values - low) * binsPerValue).astype(np.int64), 0, table.size - 1)]
        alpha = np.where(within, alpha, 0.0)
        if roiBounds is not None:
            samples = origins[:, None, :] + rays[:, None, :] * ts[None, :, None]
            for axis in range(3):
                alpha = np.where(
                    (samples[..., axis] >= roiBounds[2 * axis]) & (samples[..., axis] <= roiBounds[2 * axis + 1]), alpha, 0.0
                )
        seen = alpha > STREAM_3D_VISIBLE_OPACITY
        anySeen = seen.any(axis=1)
        if not anySeen.any():
            return None
        stepAlpha = 1.0 - np.power(1.0 - np.clip(alpha, 0.0, 1.0), step / unit)  # the renderer's opacity correction
        accumulated = 1.0 - np.cumprod(1.0 - stepAlpha, axis=1)
        opaque = accumulated >= STREAM_3D_OPAQUE
        first = np.argmax(seen, axis=1)
        last = np.where(opaque.any(axis=1), np.argmax(opaque, axis=1), np.searchsorted(ts, leave, side="right") - 1)
        last = np.clip(last, 0, ts.size - 1)
        cosine = rays @ direction  # distance along a ray -> depth along the view direction
        near = float((ts[first] * cosine)[anySeen].min())
        far = float((ts[last] * cosine)[anySeen].max())
        margin = float(self.spacing[level])  # one coarse voxel each side
        return near - margin, far + margin

    def update3D(self):
        if self.stopped or self.volume3D is None:
            return
        if self.volume3D.GetScene() is None:  # deleted by the user
            self.disable3D()
            return
        request = self.volumeRequest()
        shownLevel = self.shown3D[0] if self.shown3D else self.contextLevel
        if request is not None and request["level"] < shownLevel - 1:
            # Sharpen one level at a time: neighbouring levels look alike, and each costs an eighth of the next.
            request = self.volumeRequest(finest=shownLevel - 1) or request
        if request is None:
            self.request3D = None
            if self.shown3D is None or self.shown3D[0] != self.contextLevel:
                self.showContext3D()
            self.requestChunks()
            return
        current = self.request3D
        if current and (current["level"], current["region"]) == (request["level"], request["region"]):
            return
        self.markReusable(request)
        self.request3D = request
        if not self.covers3D(request):
            self.showContext3D()  # never leave part of the view empty while the new texture loads
        self.requestChunks()
        self.poll()

    def markReusable(self, request):
        """Let the new texture copy, from the one shown now, every chunk whose part it needs lies
        inside it: moving the camera a little then reads only the newly visible chunks. The old
        image is kept referenced until the new one is built (its GPU copy is freed when replaced)."""
        request["reuse"], request["reuseFrom"] = set(), None
        if self.shown3D is None or self.shown3D[0] != request["level"] or self.volume3D is None:
            return
        from vtk.util import numpy_support

        oldRegion = self.shown3D[1]
        image = self.volume3D.GetImageData()
        if image is None:
            return
        oldVoxels = numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(
            [stop - start for start, stop in oldRegion]
        )
        grid = self.levels[request["level"]]
        for key in request["keys"]:
            part = [(max(r0, c0), min(r1, c1)) for (r0, r1), (c0, c1) in zip(request["region"], grid.bounds(key))]
            if all(o0 <= p0 and p1 <= o1 for (p0, p1), (o0, o1) in zip(part, oldRegion)):
                request["reuse"].add(key)
        if request["reuse"]:
            self.reusedChunks += len(request["reuse"])
            request["reuseFrom"] = (oldRegion, oldVoxels)
            request["reuseImage"] = image  # keeps the voxels alive if the node switches to the coarse level meanwhile

    def covers3D(self, request):
        """Whether what the 3D node holds spans the requested region."""
        if self.shown3D is None:
            return False
        low, high = self.regionRasBounds(*self.shown3D)
        wantLow, wantHigh = self.regionRasBounds(request["level"], request["region"])
        tolerance = self.spacing[request["level"]]
        return bool(np.all(low <= wantLow + tolerance) and np.all(high >= wantHigh - tolerance))

    @staticmethod
    def newImage(shape, dtype):
        """A vtkImageData of (z, y, x) ``shape`` and a numpy view of its voxels."""
        from vtk.util import numpy_support

        imageData = vtk.vtkImageData()
        imageData.SetDimensions(int(shape[2]), int(shape[1]), int(shape[0]))
        imageData.AllocateScalars(numpy_support.get_vtk_array_type(np.dtype(dtype)), 1)
        return imageData, numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars()).reshape(shape)

    @staticmethod
    def replaceImage(node, array):
        """Give ``node`` a new image holding ``array``, built completely before it is attached.

        slicer.util.updateVolumeFromArray resizes the node's image in place: SetDimensions fires
        Modified before AllocateScalars runs, the volume rendering displayable manager renders
        synchronously on it, and the texture upload then reads the old, smaller buffer with the new
        dimensions, which crashes in the OpenGL driver."""
        imageData, voxels = Streamer.newImage(array.shape, array.dtype)
        voxels[:] = array
        node.SetAndObserveImageData(imageData)

    def setVolume3D(self, image, ijkToRas, level):
        """Show ``image`` (a complete vtkImageData, or an array copied into a new one) in the 3D node."""
        if isinstance(image, np.ndarray):
            self.replaceImage(self.volume3D, image)
        else:
            self.volume3D.SetAndObserveImageData(image)
        self.volume3D.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(ijkToRas))
        self.volume3D.SetAttribute("OMEZarr.Level", str(level))

    def showContext3D(self):
        image = self.contextImage if self.contextImage is not None else self.contextArray
        self.setVolume3D(image, self.ijkToRas[self.contextLevel], self.contextLevel)
        self.shown3D = (self.contextLevel, tuple((0, n) for n in self.levels[self.contextLevel].shape))

    def detectGpuLimits(self, widget):
        """Largest 3D texture side the GPU accepts, and the memory a 3D texture may use: the module's
        "3D texture memory" setting when set; otherwise a cube of the largest side (2048 -> 8 Gi
        voxels), capped at 75% of the video memory where VTK can read it (Windows, Linux; macOS
        reports none). Slicer's own GPU memory size is not used: it has had no GUI since 2020, as
        VTK ignores it."""
        try:
            renderWindow = widget.threeDView().renderWindow()
            renderWindow.MakeCurrent()
            size = int(vtk.vtkTextureObject.GetMaximumTextureSize3D(renderWindow))
            if size > 0:
                self.maxTextureDim = size
        except Exception:  # noqa: BLE001 - keep the fallback
            logging.debug("OME-Zarr streaming: could not read the maximum 3D texture size", exc_info=True)
        budget = Settings.get(Settings.STREAM_3D_MEMORY, 0) << 20
        if not budget:
            budget = self.maxTextureDim**3 * np.dtype(self.dtype).itemsize
            try:
                gpus = vtk.vtkGPUInfoList()
                gpus.Probe()
                video = max((gpus.GetGPUInfo(i).GetDedicatedVideoMemory() for i in range(gpus.GetNumberOfGPUs())), default=0)
                if video > 0:
                    budget = min(budget, int(0.75 * video))
            except Exception:  # noqa: BLE001 - keep the cube
                pass
        self.maxTextureBytes = budget or STREAM_3D_MAX_BYTES
        logging.info(
            f"OME-Zarr streaming: 3D textures up to {self.maxTextureDim} voxels per side, {self.maxTextureBytes / 2**30:.1f} GiB"
        )

    def fitsTexture(self, level):
        shape = self.levels[level].shape
        return max(shape) <= self.maxTextureDim and int(np.prod(shape)) * np.dtype(self.dtype).itemsize <= self.maxTextureBytes

    def planContext(self):
        """Make the 3D fallback the finest whole level the GPU holds, never finer than the level kept
        in memory: that level itself (shared with the volume node) once streamed, or a coarser one
        read once. When it is full resolution, the 3D view no longer follows the camera."""
        if self.volume3D is None:
            return
        fitting = next((level for level in range(len(self.levels)) if self.fitsTexture(level)), len(self.levels) - 1)
        level = max(self.target, fitting)
        if level >= self.contextLevel:
            return
        if level == self.residentLevel:  # wholly in memory: share the volume node's voxels
            self.setContext(level, self.node.GetImageData())
            return
        if self.target <= level < self.residentLevel:
            return  # the level-by-level fill gets there
        if self.contextRequest is None or self.contextRequest["level"] != level:
            region = tuple((0, n) for n in self.levels[level].shape)
            self.contextRequest = {"level": level, "region": region, "keys": self.levels[level].keys(region), "shown": False}
            self.requestChunks()

    def setContext(self, level, image):
        self.contextLevel, self.contextImage = level, image
        if self.volume3D is not None:
            if self.shown3D is None or self.shown3D[0] >= level:  # showing something coarser: upgrade now
                self.showContext3D()
            self.request3D = None
            self.update3D()

    def show3D(self, level, region, image):
        ijkToRas = self.ijkToRas[level].copy()
        ijkToRas[:3, 3] = (ijkToRas @ np.array([region[2][0], region[1][0], region[0][0], 1.0]))[:3]
        self.setVolume3D(image, ijkToRas, level)
        self.shown3D = (level, region)

    def stop(self, wait=False):
        if self.stopped:
            if wait:
                self.joinReaders()
            return
        self.disable3D(stopWhenDone=False)
        self.updateLevelLabels(clear=not self.complete or self.node.GetScene() is None)
        if not self.complete and slicer.util.mainWindow():
            slicer.util.showStatusMessage("", 1)  # the streaming line does not time out by itself
        self.stopped = True
        self.viewTimer.stop()
        self.pollTimer.stop()
        for caller, tag in self.observers:
            caller.RemoveObserver(tag)
        self.observers = []
        for viewName in list(self.overlays):
            self.removeOverlay(viewName)
        with self.lock:
            self.cache.clear()
            self.cachedBytes = 0
            if not self.complete:  # the buffer was never handed to the node
                self.targetArray = None
                self.targetImageData = None
        if wait:
            self.joinReaders()

    def joinReaders(self, timeoutSeconds=30.0):
        for thread in self.threads:
            thread.join(timeoutSeconds)


#
# File reader (drives Add Data, drag-and-drop, slicer.util.loadNodeFromFile)
#


class OMEZarrFileReader:
    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return _("OME-Zarr image")

    def fileType(self):
        return "OMEZarr"

    def extensions(self):
        return [
            _("OME-Zarr") + " (*.ome.zarr)",
            _("OME-Zarr") + " (*.zarr)",
            _("OME-Zarr zip") + " (*.ozx *.zarr.zip)",
            _("OME-Zarr metadata") + " (zarr.json .zattrs)",
        ]

    def canLoadFileConfidence(self, filePath):
        # Do not use self.parent.supportedNameFilters(): it rejects directories.
        # 0.9 outranks the default extension-based confidence of core readers
        # (0.5 + 0.01 * extension length), which matters for zarr.json/.zattrs
        # entries listed by the Add Data dialog.
        return 0.9 if omeZarrRootFromPath(filePath) else 0.0

    def load(self, properties):
        try:
            root = omeZarrRootFromPath(properties["fileName"])
            if not root:
                raise ValueError(f"Not an OME-Zarr multiscales store: {properties['fileName']}")

            def optional(key, cast):
                value = properties.get(key)
                return cast(value) if value not in (None, "") else None

            level = optional("level", int)
            timeIndex = optional("timeIndex", int)
            maxBytes = optional("maxBytes", int)
            timeMode = optional("timeMode", str)
            asLabelMap = optional("asLabelMap", lambda v: str(v).lower() in ("true", "1"))
            with Progress(_("Loading OME-Zarr..."), properties["fileName"]) as progress:
                # Reading the store's metadata (and probing for a label map) goes over the network:
                # off the GUI thread, under the progress dialog, so a slow server cannot freeze Slicer.
                progress(0, 1, _("Reading {name}...").format(name=os.path.basename(root.rstrip("/"))))
                DiskCache.instance()  # set up from the settings here; the readers use it from their threads
                streaming = None
                if self.mayStream(properties, asLabelMap):
                    streaming = runResponsive(
                        lambda: self.streamingLevels(root, level, maxBytes, timeMode, asLabelMap),
                        lambda: progress(0, 1) or None,
                    )
                if progress.cancelled:
                    raise InterruptedError("Loading cancelled")
                nodes = OMEZarrLogic.loadImage(
                    root,
                    level=streaming[0] if streaming else level,
                    timeIndex=timeIndex,
                    name=properties.get("name"),
                    maxBytes=maxBytes,
                    labels=optional("labels", lambda v: str(v).lower() in ("true", "1")),
                    timeMode=timeMode,
                    asLabelMap=asLabelMap,
                    userMessages=self.parent.userMessages(),
                    progress=progress,
                    announceLevel=streaming is None,
                )
        except InterruptedError:
            self.parent.userMessages().AddMessage(vtk.vtkCommand.WarningEvent, "Loading cancelled")
            return False
        except Exception as e:  # noqa: BLE001 - report everything to the user
            import traceback

            traceback.print_exc()
            self.parent.userMessages().AddMessage(vtk.vtkCommand.ErrorEvent, f"Failed to read OME-Zarr: {e}")
            return False

        if properties.get("show", True) and nodes:
            scalars = scalarVolumes(nodes)
            labels = labelMaps(nodes)
            slicer.util.setSliceViewerLayers(
                background=scalars[0] if scalars else "keep-current",
                label=labels[0] if labels else "keep-current",
                fit=True,
            )
            coarse = any(n.GetAttribute("OMEZarr.Level") not in (None, "0") for n in scalars)
            if streaming and scalars:
                streamer = OMEZarrLogic.startStreaming(root, scalars[0], streaming[1], timeIndex=timeIndex or 0)
                if Settings.get(Settings.STREAM_3D, False):
                    with slicer.util.tryWithErrorDisplay(_("Failed to start volume rendering")):
                        streamer.enable3D()
            elif coarse and Settings.get(Settings.AUTO_REFINE, False) and slicer.util.mainWindow():
                OMEZarrLogic.startAutoRefine(root)
        self.parent.loadedNodes = [node.GetID() for node in nodes]
        return True

    @staticmethod
    def mayStream(properties, asLabelMap):
        """The streaming conditions that need the GUI thread (settings, views)."""
        return bool(
            not asLabelMap
            and properties.get("show", True)
            and Settings.get(Settings.STREAM, True)
            and slicer.app.layoutManager() is not None
        )

    @staticmethod
    def streamingLevels(root, level, maxBytes, timeMode, asLabelMap):
        """(shown, target) when this load should show a level at once and stream a finer target:
        the coarsest level by default, or the level asked for when the budget allows a finer one.
        Reads the store over the network: run it off the GUI thread."""
        if isBioformats2rawRoot(root) or isLabelStore(root):
            return None
        multiscales = OMEZarrLogic.openMultiscales(root)
        if asLabelMap is None and OMEZarrLogic.looksLikeLabelMap(multiscales):
            return None
        levels = OMEZarrLogic.streamingLevels(multiscales, maxBytes, timeMode)
        if levels is not None:
            for index in range(len(multiscales.images)):  # the streamer then has its readers at once
                OMEZarrLogic.chunkReader(multiscales, index)
        if levels is None or level is None:
            return levels
        target = levels[1]
        return (level, target) if 0 <= level < len(multiscales.images) and target < level else None


#
# File writer (File > Save, slicer.util.saveNode)
#


class OMEZarrFileWriter:
    def __init__(self, parent):
        self.parent = parent

    def description(self):
        return _("OME-Zarr image")

    def fileType(self):
        return "OMEZarr"

    def extensions(self, obj):
        # The second, generic filter lets a label map be written as ``<image>/labels/<name>``.
        return [_("OME-Zarr") + " (.ome.zarr)", _("OME-Zarr directory") + " (*)"]

    def canWriteObjectConfidence(self, obj):
        # Below the default 0.5 so the native formats stay the default choice in the save dialog.
        if (
            obj.IsA("vtkMRMLLabelMapVolumeNode")
            or obj.IsA("vtkMRMLScalarVolumeNode")
            or obj.IsA("vtkMRMLSegmentationNode")
        ):
            return 0.3
        return 0.0

    def write(self, properties):
        try:
            node = slicer.mrmlScene.GetNodeByID(properties["nodeID"])
            with Progress(_("Writing OME-Zarr...")) as progress:
                OMEZarrLogic.writeVolume(node, properties["fileName"], progress)
        except Exception as e:  # noqa: BLE001 - report everything to the user
            import traceback

            traceback.print_exc()
            self.parent.userMessages().AddMessage(vtk.vtkCommand.ErrorEvent, f"Failed to write OME-Zarr: {e}")
            return False
        self.parent.writtenNodes = [node.GetID()]
        return True


#
# Drop target for OME-Zarr directories (same mechanism the DICOM module uses)
#


class OMEZarrFileDialog:
    """Detected by name by qSlicerScriptedLoadableModule and registered with the IO manager."""

    def __init__(self, qSlicerFileDialog):
        self.qSlicerFileDialog = qSlicerFileDialog
        qSlicerFileDialog.fileType = "OMEZarr"
        qSlicerFileDialog.description = _("Load OME-Zarr image")
        qSlicerFileDialog.action = slicer.qSlicerFileDialog.Read
        self.pathsToLoad = []

    def execDialog(self):
        # Not used: loading is triggered from dropEvent.
        return True

    def isMimeDataAccepted(self):
        """Accept only when every dropped URL is an OME-Zarr store (so plain files/dirs keep their usual handling)."""
        self.pathsToLoad = []
        mimeData = self.qSlicerFileDialog.mimeData()
        accepted = False
        if mimeData.hasFormat("text/uri-list"):
            roots = [omeZarrRootFromPath(url.toLocalFile() or url.toString()) for url in mimeData.urls()]
            if roots and all(roots):
                self.pathsToLoad = roots
                accepted = True
        self.qSlicerFileDialog.acceptMimeData(accepted)

    def dropEvent(self):
        paths = list(self.pathsToLoad)
        self.pathsToLoad = []
        # Defer so the drag source application is not blocked while we read data.
        qt.QTimer.singleShot(0, lambda: self._loadPaths(paths))

    @staticmethod
    def _loadPaths(paths):
        for path in paths:
            with slicer.util.tryWithErrorDisplay(_("Failed to load OME-Zarr image"), waitCursor=True):
                slicer.util.loadNodeFromFile(path, "OMEZarr")


#
# Widget
#


class OMEZarrWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        import ctk

        self.logic = OMEZarrLogic()
        self.multiscales = None
        self.path = None
        self._inspecting = False

        # -- Store --
        storeBox = ctk.ctkCollapsibleButton()
        storeBox.text = _("Store")
        self.layout.addWidget(storeBox)
        storeLayout = qt.QVBoxLayout(storeBox)  # a form layout mis-measures word-wrapped labels

        self.pathEdit = ctk.ctkPathLineEdit()
        self.pathEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.pathEdit.settingKey = "OMEZarr/LastPath"
        if not self.pathEdit.currentPath and not slicer.app.testingEnabled():
            self.pathEdit.currentPath = SAMPLE_STORE_URL  # nothing used yet on this system: the large test store
        self.pathEdit.setToolTip(_("Local .ome.zarr directory, .ozx file, or https:// / s3:// URL"))
        self.inspectButton = qt.QToolButton()
        self.inspectButton.setIcon(slicer.app.style().standardIcon(qt.QStyle.SP_BrowserReload))
        self.inspectButton.setToolTip(_("Read the store's metadata again"))
        pathRow = qt.QHBoxLayout()
        pathRow.addWidget(self.pathEdit, 1)
        pathRow.addWidget(self.inspectButton)
        storeLayout.addLayout(pathRow)

        self.infoLabel = qt.QLabel(
            _("Choose an OME-Zarr store, or drop one onto Slicer: its resolution levels are listed here.")
        )
        self.infoLabel.wordWrap = True
        self.infoLabel.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        storeLayout.addWidget(self.infoLabel)

        self.levelTable = qt.QTableWidget(0, 4)
        self.levelTable.setHorizontalHeaderLabels([_("Level"), _("Voxels (x, y, z)"), _("Spacing"), _("Memory")])
        self.levelTable.setTextElideMode(qt.Qt.ElideRight)
        self.levelTable.setWordWrap(False)
        self.levelTable.verticalHeader().setVisible(False)
        self.levelTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.levelTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.levelTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self.levelTable.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.levelTable.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        header = self.levelTable.horizontalHeader()
        header.setSectionResizeMode(qt.QHeaderView.Stretch)
        for column in (0, 1, 3):
            header.setSectionResizeMode(column, qt.QHeaderView.ResizeToContents)
        storeLayout.addWidget(self.levelTable)

        self.legendLabel = qt.QLabel(_("Bold: the level the memory budget selects. ✓: loaded in the scene."))
        self.legendLabel.wordWrap = True
        self.legendLabel.enabled = False  # greyed like a hint, in any Slicer style
        storeLayout.addWidget(self.legendLabel)

        self.timeIndexLabel = qt.QLabel(_("Time point:"))
        self.timeIndexSpinBox = qt.QSpinBox()
        self.timeIndexSpinBox.setRange(0, 0)
        self.timeIndexSpinBox.setSpecialValueText(_("all (sequence)"))
        timeRow = qt.QHBoxLayout()
        timeRow.addWidget(self.timeIndexLabel)
        timeRow.addWidget(self.timeIndexSpinBox, 1)
        storeLayout.addLayout(timeRow)

        self.loadButton = qt.QPushButton(_("Load selected level"))
        storeLayout.addWidget(self.loadButton)

        # -- Full resolution --
        refineBox = ctk.ctkCollapsibleButton()
        refineBox.text = _("Full resolution")
        self.layout.addWidget(refineBox)
        refineBoxLayout = qt.QVBoxLayout(refineBox)
        refineLayout = qt.QFormLayout()
        refineBoxLayout.addLayout(refineLayout)

        self.viewSelector = qt.QComboBox()
        self.viewSelector.setToolTip(_("The 2D view that 'Refine view' and 'New ROI in view' act on"))
        self.updateViewSelector()
        self.refineButton = qt.QPushButton(_("Refine view"))
        self.refineButton.setToolTip(
            _("Reload the block shown by the slice view at the finest level that fits the memory budget")
        )
        refineRow = qt.QHBoxLayout()
        refineRow.addWidget(self.viewSelector)
        refineRow.addWidget(self.refineButton, 1)
        refineLayout.addRow(_("2D view:"), refineRow)

        self.autoRefineCheckBox = qt.QCheckBox(_("Refine the slice views automatically while browsing"))
        self.autoRefineCheckBox.setToolTip(_("Reloads a view's block after it has been still for half a second"))
        refineLayout.addRow(self.autoRefineCheckBox)

        self.volumeRenderingCheckBox = qt.QCheckBox(_("Volume-render in the 3D view, at the resolution it shows"))
        self.volumeRenderingCheckBox.checked = Settings.get(Settings.STREAM_3D, False)
        self.volumeRenderingCheckBox.setToolTip(
            _(
                "Can be set before loading. Streamed stores are then rendered as they load: the 3D view holds "
                "only what is in view (and in the cropping ROI), at the level its screen resolution needs, "
                "and sharpens after the camera stops"
            )
        )
        refineLayout.addRow(self.volumeRenderingCheckBox)

        self.roiSelector = slicer.qMRMLNodeComboBox()
        self.roiSelector.nodeTypes = ["vtkMRMLMarkupsROINode"]
        self.roiSelector.addEnabled = False  # "New ROI in view" creates one that is already placed
        self.roiSelector.removeEnabled = True
        self.roiSelector.renameEnabled = True
        self.roiSelector.noneEnabled = True
        self.roiSelector.setMRMLScene(slicer.mrmlScene)
        self.createRoiButton = qt.QPushButton(_("New ROI in view"))
        self.createRoiButton.setToolTip(
            _("Create a region of interest covering the middle of the selected slice view; drag its handles to adjust")
        )
        roiRow = qt.QHBoxLayout()
        roiRow.addWidget(self.roiSelector, 1)
        roiRow.addWidget(self.createRoiButton)
        refineLayout.addRow(_("Region of interest:"), roiRow)
        self.loadRegionButton = qt.QPushButton(_("Load region at the selected level"))
        self.loadRegionButton.setToolTip(_("Only the chunks the region intersects are read"))
        refineLayout.addRow(self.loadRegionButton)

        self.statusLabel = qt.QLabel()
        self.statusLabel.wordWrap = True
        refineBoxLayout.addWidget(self.statusLabel)

        # -- Settings --
        settingsBox = ctk.ctkCollapsibleButton()
        settingsBox.text = _("Settings")
        settingsBox.collapsed = True
        self.layout.addWidget(settingsBox)
        settingsLayout = qt.QFormLayout(settingsBox)

        self.maxBytesSpinBox = qt.QSpinBox()
        self.maxBytesSpinBox.setRange(0, 1 << 20)
        self.maxBytesSpinBox.setSuffix(" MiB")
        self.maxBytesSpinBox.setSpecialValueText(_("automatic (25% of free RAM)"))
        self.maxBytesSpinBox.setValue(Settings.get(Settings.MAX_BYTES, 0) >> 20)
        self.maxBytesSpinBox.setToolTip(_("Budget for automatic level selection"))
        settingsLayout.addRow(_("Memory budget:"), self.maxBytesSpinBox)

        self.orientationSelector = qt.QComboBox()
        self.orientationSelector.addItems(["LPS", "RAS"])
        self.orientationSelector.setCurrentText(Settings.get(Settings.ORIENTATION, "LPS"))
        self.orientationSelector.setToolTip(_("How x/y/z axes are interpreted when a store has no RFC-4 orientation"))
        settingsLayout.addRow(_("Axes without RFC-4:"), self.orientationSelector)

        self.loadLabelsCheckBox = qt.QCheckBox()
        self.loadLabelsCheckBox.checked = Settings.get(Settings.LOAD_LABELS, True)
        settingsLayout.addRow(_("Load labels:"), self.loadLabelsCheckBox)

        self.labelsAsSegmentationCheckBox = qt.QCheckBox()
        self.labelsAsSegmentationCheckBox.checked = Settings.get(Settings.LABELS_AS_SEGMENTATION, False)
        self.labelsAsSegmentationCheckBox.setToolTip(_("Import labels as Segmentation nodes instead of label maps"))
        settingsLayout.addRow(_("Labels as segmentation:"), self.labelsAsSegmentationCheckBox)

        self.timeModeSelector = qt.QComboBox()
        self.timeModeSelector.addItem(_("all time points as a Sequence"), "sequence")
        self.timeModeSelector.addItem(_("first time point only"), "index")
        self.timeModeSelector.setCurrentIndex(0 if Settings.get(Settings.TIME_MODE, "sequence") == "sequence" else 1)
        settingsLayout.addRow(_("Time axis:"), self.timeModeSelector)

        self.displayUnitsCheckBox = qt.QCheckBox()
        self.displayUnitsCheckBox.checked = Settings.get(Settings.DISPLAY_UNITS, False)
        self.displayUnitsCheckBox.setToolTip(_("Show lengths in the store's unit (µm, nm) instead of mm"))
        settingsLayout.addRow(_("Display in store units:"), self.displayUnitsCheckBox)

        self.detectLabelMapsCheckBox = qt.QCheckBox()
        self.detectLabelMapsCheckBox.checked = Settings.get(Settings.DETECT_LABEL_MAPS, True)
        self.detectLabelMapsCheckBox.setToolTip(_("Integer stores with few distinct values load as label maps"))
        settingsLayout.addRow(_("Detect label maps:"), self.detectLabelMapsCheckBox)

        self.storageOptionsEdit = qt.QLineEdit(Settings.get(Settings.STORAGE_OPTIONS, ""))
        self.storageOptionsEdit.setPlaceholderText('{"anon": true, "region": "us-west-2"}')
        self.storageOptionsEdit.setToolTip(
            _(
                "JSON storage options for remote stores: S3 credentials, endpoint, region. "
                "S3 is read anonymously unless credentials are given here or in the environment "
                "(AWS_ACCESS_KEY_ID); ~/.aws files are not read."
            )
        )
        settingsLayout.addRow(_("Remote storage options:"), self.storageOptionsEdit)

        self.autoRefineOnLoadCheckBox = qt.QCheckBox()
        self.autoRefineOnLoadCheckBox.checked = Settings.get(Settings.AUTO_REFINE, False)
        self.autoRefineOnLoadCheckBox.setToolTip(
            _("After loading a downsampled level, refine the slice views automatically")
        )
        settingsLayout.addRow(_("Auto-refine after load:"), self.autoRefineOnLoadCheckBox)

        self.streamCheckBox = qt.QCheckBox()
        self.streamCheckBox.checked = Settings.get(Settings.STREAM, True)
        self.streamCheckBox.setToolTip(
            _(
                "Show a store at once from its coarsest level. The slice views then get the plane "
                "they show at screen resolution first, and the level that fits the memory budget is "
                "read in the background"
            )
        )
        settingsLayout.addRow(_("Stream large stores:"), self.streamCheckBox)

        self.textureMemorySpinBox = qt.QSpinBox()
        self.textureMemorySpinBox.setRange(0, 1 << 20)
        self.textureMemorySpinBox.setSingleStep(512)
        self.textureMemorySpinBox.setSuffix(" MiB")
        self.textureMemorySpinBox.setSpecialValueText(_("automatic (largest texture side, cubed)"))
        self.textureMemorySpinBox.setValue(Settings.get(Settings.STREAM_3D_MEMORY, 0))
        self.textureMemorySpinBox.setToolTip(
            _(
                "Most memory one streamed 3D texture may use. Automatic: the cube of the largest 3D texture "
                "side the GPU accepts (2048 -> 8 GiB of 8-bit voxels), capped at 75% of the video memory where "
                "it can be read (Windows, Linux). Applies when 3D rendering of a store starts"
            )
        )
        settingsLayout.addRow(_("3D texture memory:"), self.textureMemorySpinBox)

        self.diskCacheSpinBox = qt.QSpinBox()
        self.diskCacheSpinBox.setRange(0, 1 << 22)
        self.diskCacheSpinBox.setSingleStep(1024)
        self.diskCacheSpinBox.setSuffix(" MiB")
        self.diskCacheSpinBox.setSpecialValueText(_("off"))
        self.diskCacheSpinBox.setValue(Settings.get(Settings.DISK_CACHE, DISK_CACHE_MIB))
        self.diskCacheSpinBox.setToolTip(
            _(
                "Chunks read from remote stores are kept on disk, compressed as sent, so opening a store "
                "again reads them from disk instead of the server. The least recently used are dropped "
                "beyond this size. Sharded stores are checked with the server once per session, so a "
                "replaced store is read again"
            )
        )
        self.clearDiskCacheButton = qt.QPushButton(_("Clear"))
        diskCacheRow = qt.QHBoxLayout()
        diskCacheRow.addWidget(self.diskCacheSpinBox, 1)
        diskCacheRow.addWidget(self.clearDiskCacheButton)
        settingsLayout.addRow(_("Disk cache:"), diskCacheRow)

        self.resetUnitsButton = qt.QPushButton(_("Reset display to mm"))
        settingsLayout.addRow(self.resetUnitsButton)

        self.layout.addStretch(1)
        OMEZarrLogic.panel = self  # streamers refresh the level table while they read

        self.inspectButton.connect("clicked(bool)", self.onInspect)
        self.pathEdit.connect("currentPathChanged(QString)", self.onPathChanged)
        self.loadButton.connect("clicked(bool)", self.onLoad)
        self.refineButton.connect("clicked(bool)", self.onRefine)
        self.autoRefineCheckBox.connect("toggled(bool)", self.onAutoRefineToggled)
        self.volumeRenderingCheckBox.connect("toggled(bool)", self.onVolumeRenderingToggled)
        self.loadRegionButton.connect("clicked(bool)", self.onLoadRegion)
        self.createRoiButton.connect("clicked(bool)", self.onCreateRoi)
        self.levelTable.connect("itemSelectionChanged()", self.updateButtons)
        self.levelTable.connect("cellDoubleClicked(int,int)", lambda row, column: self.onLoad())
        self.roiSelector.connect("currentNodeChanged(vtkMRMLNode*)", lambda node: self.updateButtons())
        self.updateButtons()
        self.maxBytesSpinBox.connect("valueChanged(int)", lambda mib: Settings.set(Settings.MAX_BYTES, int(mib) << 20))
        self.orientationSelector.connect("currentTextChanged(QString)", lambda t: Settings.set(Settings.ORIENTATION, t))
        self.loadLabelsCheckBox.connect("toggled(bool)", lambda b: Settings.set(Settings.LOAD_LABELS, bool(b)))
        self.labelsAsSegmentationCheckBox.connect(
            "toggled(bool)", lambda b: Settings.set(Settings.LABELS_AS_SEGMENTATION, bool(b))
        )
        self.timeModeSelector.connect(
            "currentIndexChanged(int)", lambda i: Settings.set(Settings.TIME_MODE, self.timeModeSelector.itemData(i))
        )
        self.displayUnitsCheckBox.connect("toggled(bool)", lambda b: Settings.set(Settings.DISPLAY_UNITS, bool(b)))
        self.resetUnitsButton.connect("clicked(bool)", OMEZarrLogic.resetDisplayUnits)
        self.autoRefineOnLoadCheckBox.connect("toggled(bool)", lambda b: Settings.set(Settings.AUTO_REFINE, bool(b)))
        self.streamCheckBox.connect("toggled(bool)", lambda b: Settings.set(Settings.STREAM, bool(b)))
        self.textureMemorySpinBox.connect(
            "valueChanged(int)", lambda mib: Settings.set(Settings.STREAM_3D_MEMORY, int(mib))
        )
        self.diskCacheSpinBox.connect("valueChanged(int)", self.onDiskCacheSizeChanged)
        self.clearDiskCacheButton.connect("clicked(bool)", self.onClearDiskCache)
        self.updateDiskCacheButton()
        self.detectLabelMapsCheckBox.connect(
            "toggled(bool)", lambda b: Settings.set(Settings.DETECT_LABEL_MAPS, bool(b))
        )
        self.storageOptionsEdit.connect(
            "editingFinished()", lambda: Settings.set(Settings.STORAGE_OPTIONS, self.storageOptionsEdit.text)
        )

    def updateViewSelector(self):
        """List the slice views of the current layout as "Red (Axial)", keeping the selection."""
        layoutManager = slicer.app.layoutManager()
        current = self.viewSelector.currentData or "Red"
        self.viewSelector.clear()
        for name in layoutManager.sliceViewNames():
            orientation = layoutManager.sliceWidget(name).mrmlSliceNode().GetOrientation()
            self.viewSelector.addItem(f"{name} ({orientation})", name)
        self.viewSelector.setCurrentIndex(max(0, self.viewSelector.findData(current)))

    def onDiskCacheSizeChanged(self, mib):
        Settings.set(Settings.DISK_CACHE, int(mib))
        DiskCache.instance()  # stores opened from now on use the new size
        self.updateDiskCacheButton()

    def onClearDiskCache(self):
        cache = DiskCache.instance()
        if cache is not None:
            cache.clear()
        self.updateDiskCacheButton()

    def updateDiskCacheButton(self):
        cache = DiskCache.instance()
        used = cache.size if cache is not None else 0
        self.clearDiskCacheButton.text = _("Clear ({size})").format(size=self.formatBytes(used)) if used else _("Clear")
        self.clearDiskCacheButton.enabled = bool(used)
        self.clearDiskCacheButton.toolTip = cache.root if cache is not None else ""

    def enter(self):
        """Show the store of the displayed OME-Zarr volume, else the last one used."""
        self.updateViewSelector()
        self.updateDiskCacheButton()
        if self.path:
            self.updateLevelStatus()
            return
        sliceWidget = slicer.app.layoutManager().sliceWidget("Red")
        background = sliceWidget.sliceLogic().GetBackgroundLayer().GetVolumeNode() if sliceWidget else None
        candidates = [background, *slicer.util.getNodesByClass("vtkMRMLVolumeNode")[::-1]]
        stored = next(
            (n.GetAttribute("OMEZarr.Path") for n in candidates if n and n.GetAttribute("OMEZarr.Path")), None
        )
        if stored and not samePath(stored, self.pathEdit.currentPath):
            self.pathEdit.currentPath = stored  # triggers onPathChanged
        else:
            self.onPathChanged(self.pathEdit.currentPath)

    def onPathChanged(self, path):
        """Inspect as soon as the path names a store; stay quiet while it does not."""
        if self._inspecting:
            return
        if omeZarrRootFromPath(path):
            if not samePath(path, self.path):
                self.onInspect()
        else:
            self.path = None
            self.multiscales = None
            self.levelTable.setRowCount(0)
            self.fitTableHeight()
            self.infoLabel.text = _(
                "Choose an OME-Zarr store, or drop one onto Slicer: its resolution levels are listed here."
            )
            self.updateButtons()

    def selectedLevel(self):
        rows = self.levelTable.selectionModel().selectedRows() if self.levelTable.rowCount else []
        return rows[0].row() if rows else -1

    def updateButtons(self):
        hasStore = self.path is not None
        self.loadButton.setEnabled(hasStore and self.selectedLevel() >= 0)
        self.refineButton.setEnabled(hasStore)
        self.createRoiButton.setEnabled(hasStore)
        self.autoRefineCheckBox.setEnabled(hasStore)
        self.loadRegionButton.setEnabled(hasStore and self.roiSelector.currentNode() is not None)

    def fitTableHeight(self):
        """Size the table to its rows so that it never shows an empty area."""
        table = self.levelTable
        rows = sum(table.rowHeight(row) for row in range(table.rowCount)) or table.verticalHeader().defaultSectionSize
        table.setFixedHeight(table.horizontalHeader().height + rows + 2 * table.frameWidth)

    def onInspect(self):
        path = omeZarrRootFromPath(self.pathEdit.currentPath)
        if not path:
            slicer.util.errorDisplay(_("Not an OME-Zarr multiscales store."))
            return
        self._inspecting = True  # adding the path to the history re-emits currentPathChanged
        try:
            with slicer.util.tryWithErrorDisplay(_("Failed to open store"), waitCursor=True):
                self.multiscales = self.logic.openMultiscales(path, useCache=False)
                self.path = path
                self.pathEdit.addCurrentPathToHistory()
                self.showLevels()
        finally:
            self._inspecting = False
        self.updateButtons()

    def showLevels(self):
        """Fill the level table and the summary line from the open store."""
        info = self.logic.levelInfo(self.multiscales)
        self.levelTable.setRowCount(len(info))
        for row, level in enumerate(info):
            shape = dict(zip(level["dims"], level["shape"], strict=False))
            unit = next((level["units"].get(d) for d in SPATIAL_DIMS if level["units"].get(d)), None)
            unit = UNIT_SYMBOLS.get(unit, unit or "")
            values = [
                str(level["level"]),
                " × ".join(str(shape[d]) for d in SPATIAL_DIMS if d in shape),
                (" × ".join(f"{level['scale'][d]:.4g}" for d in SPATIAL_DIMS if d in shape) + f" {unit}").strip(),
                self.formatBytes(level["bytes"]),
            ]
            for column, value in enumerate(values):
                item = qt.QTableWidgetItem(value)
                item.setToolTip(value)
                if column == 3:
                    item.setTextAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
                self.levelTable.setItem(row, column, item)
        self.fitTableHeight()

        image = self.multiscales.images[0]
        channels = self.logic.axisLength(image, "c")
        timePoints = self.logic.axisLength(image, "t")
        self.timeIndexSpinBox.setRange(-1 if timePoints > 1 else 0, max(0, timePoints - 1))
        self.timeIndexSpinBox.value = -1 if timePoints > 1 else 0
        self.timeIndexLabel.setVisible(timePoints > 1)
        self.timeIndexSpinBox.setVisible(timePoints > 1)
        _matrix, source = self.logic.ijkToRasMatrix(image)
        labels = labelGroupNames(self.path)
        orientation = _("RFC-4 orientation") if source == "rfc4" else _("axes {0}").format(source.replace("-", " "))
        self.infoLabel.text = " · ".join(
            [
                info[0]["dtype"],
                _("{n} channel(s)").format(n=channels),
                _("{n} time point(s)").format(n=timePoints),
                _("labels: {names}").format(names=", ".join(labels)) if labels else _("no labels"),
                orientation,
            ]
        )
        self.levelTable.selectRow(self.logic.selectLevel(self.multiscales, self.logic.maxBytesFromSettings()))
        self.updateLevelStatus()

    @staticmethod
    def formatBytes(size):
        return f"{size / 2**30:.2f} GiB" if size >= 2**30 else f"{size / 2**20:.1f} MiB"

    def updateLevelStatus(self):
        """Bold the level the budget selects. While the store streams, mark what was downloaded
        (✓ whole, ↓ being read, ◐ parts for the views) and the level the volume shows (▸);
        otherwise tick the levels present in the scene."""
        if not self.path or self.multiscales is None:
            return
        recommended = self.logic.selectLevel(self.multiscales, self.logic.maxBytesFromSettings())
        streamer = OMEZarrLogic.streamer(self.path)
        if streamer is not None:
            self.legendLabel.text = _(
                "Bold: the level the memory budget selects. ▸ shown · ✓ downloaded · ↓ being read · ◐ parts read for the views."
            )
            for row in range(self.levelTable.rowCount):
                mark, note = streamer.levelState(row) if row < len(streamer.levels) else ("", "")
                shown = row == streamer.shownLevel
                notes = [n for n in (note, _("The volume shows this level") if shown else "") if n]
                if row == recommended:
                    notes.append(_("Selected by the memory budget"))
                levelItem = self.levelTable.item(row, 0)
                levelItem.setText(" ".join(t for t in (("▸" if shown else "") + str(row), mark) if t))
                for column in range(self.levelTable.columnCount):
                    item = self.levelTable.item(row, column)
                    font = item.font()
                    font.setBold(row == recommended)
                    item.setFont(font)
                    if column == 0:
                        item.setToolTip(". ".join(notes))
            return
        self.legendLabel.text = _("Bold: the level the memory budget selects. ✓: loaded in the scene.")
        loaded, regions = set(), set()
        for node in slicer.util.getNodesByClass("vtkMRMLVolumeNode"):
            if samePath(node.GetAttribute("OMEZarr.Path"), self.path) and node.GetAttribute("OMEZarr.Level"):
                target = regions if node.GetAttribute("OMEZarr.Region") else loaded
                target.add(int(node.GetAttribute("OMEZarr.Level")))
        for row in range(self.levelTable.rowCount):
            notes = []
            if row == recommended:
                notes.append(_("Selected by the memory budget"))
            if row in loaded:
                notes.append(_("Loaded in the scene"))
            if row in regions:
                notes.append(_("A region of this level is loaded"))
            levelItem = self.levelTable.item(row, 0)
            levelItem.setText(f"{row} ✓" if row in loaded or row in regions else str(row))
            for column in range(self.levelTable.columnCount):
                item = self.levelTable.item(row, column)
                font = item.font()
                font.setBold(row == recommended)
                item.setFont(font)
                if column == 0:
                    item.setToolTip(". ".join(notes))

    def onLoad(self):
        if not self.path or self.selectedLevel() < 0:
            return
        properties = {"level": self.selectedLevel()}
        if self.timeIndexSpinBox.value >= 0:
            properties["timeIndex"] = self.timeIndexSpinBox.value
        with slicer.util.tryWithErrorDisplay(_("Failed to load level"), waitCursor=True):
            slicer.util.loadNodeFromFile(self.path, "OMEZarr", properties)
        self.updateLevelStatus()

    def describeNodes(self, prefix, nodes):
        volumes = scalarVolumes(nodes)
        if not volumes:
            return prefix
        dims = volumes[0].GetImageData().GetDimensions()
        size = sum(n.GetImageData().GetActualMemorySize() for n in volumes) * 1024
        return _("{prefix}: level {level} · {x} × {y} × {z} voxels · {size}").format(
            prefix=prefix,
            level=volumes[0].GetAttribute("OMEZarr.Level"),
            x=dims[0],
            y=dims[1],
            z=dims[2],
            size=self.formatBytes(size),
        )

    def onRefine(self):
        if not self.path:
            return
        view = self.viewSelector.currentData
        try:
            with Progress(_("Refining view...")) as progress:
                nodes = self.logic.refineView(
                    self.path, view, timeIndex=max(0, self.timeIndexSpinBox.value), progress=progress
                )
            self.statusLabel.text = self.describeNodes(view, nodes)
        except InterruptedError:
            self.statusLabel.text = _("{view}: cancelled").format(view=view)
        except ValueError as e:  # expected refusals are shown in place, not in a popup
            self.statusLabel.text = f"{view}: {e}"
        except Exception as e:  # noqa: BLE001
            slicer.util.errorDisplay(_("Failed to refine view: {error}").format(error=e))
        self.updateLevelStatus()

    def onAutoRefineToggled(self, enabled):
        if not enabled:
            if self.path:
                OMEZarrLogic.stopAutoRefine(self.path)
            return
        if not self.path:
            self.autoRefineCheckBox.checked = False
            return
        with slicer.util.tryWithErrorDisplay(_("Failed to start automatic refinement")):
            OMEZarrLogic.startAutoRefine(self.path)

    def onVolumeRenderingToggled(self, enabled):
        """A preference, usable before any store is chosen: streamed stores are volume-rendered as
        they load, and a store already streaming starts or stops at once."""
        Settings.set(Settings.STREAM_3D, bool(enabled))
        streamer = OMEZarrLogic.streamer(self.path) if self.path else None
        if not enabled:
            if streamer is not None:
                streamer.disable3D()
            self.statusLabel.text = ""
            return
        if streamer is None:
            self.statusLabel.text = _("The 3D view will render the next store you load (streamed)")
            return
        try:
            node = streamer.enable3D()
            self.statusLabel.text = _("3D view: rendering {name}").format(name=node.GetName())
        except ValueError as e:
            self.statusLabel.text = str(e)

    def cleanup(self):
        if OMEZarrLogic.panel is self:
            OMEZarrLogic.panel = None
        OMEZarrLogic.stopAutoRefine()
        OMEZarrLogic.stopStreaming()

    def onCreateRoi(self):
        """A region of interest already placed: the middle half of what the slice view shows."""
        view = self.viewSelector.currentData
        bounds = self.logic.sliceViewRasBounds(view)
        center = [(bounds[i] + bounds[i + 1]) / 2.0 for i in (0, 2, 4)]
        size = [(bounds[i + 1] - bounds[i]) / 2.0 for i in (0, 2, 4)]
        roiNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsROINode", slicer.mrmlScene.GenerateUniqueName("OME-Zarr region")
        )
        roiNode.CreateDefaultDisplayNodes()
        roiNode.SetCenter(*center)
        roiNode.SetSize(*size)
        roiNode.GetDisplayNode().SetHandlesInteractive(True)
        roiNode.GetDisplayNode().SetFillOpacity(0.1)
        self.roiSelector.setCurrentNode(roiNode)
        self.statusLabel.text = _(
            "Drag the handles of the region in the slice views to adjust it, select a level above, "
            "then click 'Load region at the selected level'."
        )
        self.updateButtons()

    def onLoadRegion(self):
        roiNode = self.roiSelector.currentNode()
        if roiNode is None or not self.path:
            return
        try:
            with Progress(_("Loading region...")) as progress:
                nodes = self.logic.loadRegion(
                    self.path,
                    roiNode,
                    level=max(0, self.selectedLevel()),
                    timeIndex=max(0, self.timeIndexSpinBox.value),
                    progress=progress,
                )
        except InterruptedError:
            self.statusLabel.text = _("Region: cancelled")
            return
        except ValueError as e:
            self.statusLabel.text = _("Region: {error}").format(error=e)
            return
        scalars = scalarVolumes(nodes)
        if scalars:
            slicer.util.setSliceViewerLayers(background=scalars[0], fit=True)
        self.statusLabel.text = self.describeNodes(roiNode.GetName(), nodes)
        self.updateLevelStatus()


#
# Tests
#


class OMEZarrTest(ScriptedLoadableModuleTest):
    def setUp(self):
        slicer.mrmlScene.Clear()
        self.tempDir = slicer.util.tempDirectory("OMEZarrTest")
        OMEZarrLogic.ensureNgffZarr()
        OMEZarrLogic.clearCache()
        self.savedSettings = {
            key: qt.QSettings().value(key)
            for key in (
                Settings.MAX_BYTES,
                Settings.ORIENTATION,
                Settings.LOAD_LABELS,
                Settings.TIME_MODE,
                Settings.DISPLAY_UNITS,
                Settings.AUTO_REFINE,
                Settings.LABELS_AS_SEGMENTATION,
                Settings.STORAGE_OPTIONS,
                Settings.DETECT_LABEL_MAPS,
                Settings.STREAM,
                Settings.STREAM_3D,
                Settings.STREAM_3D_MEMORY,
                Settings.DISK_CACHE,
                Settings.DISK_CACHE_DIR,
            )
        }
        for key in self.savedSettings:
            qt.QSettings().remove(key)
        # Most tests check what a load returns; the streaming tests turn it back on.
        Settings.set(Settings.STREAM, False)
        Settings.set(Settings.DISK_CACHE_DIR, os.path.join(self.tempDir, "cache"))  # never the user's cache
        DiskCache.instance()

    def tearDown(self):
        import shutil

        OMEZarrLogic.stopAutoRefine()
        OMEZarrLogic.stopStreaming()
        shutil.rmtree(self.tempDir, ignore_errors=True)
        for key, value in self.savedSettings.items():
            if value is None:
                qt.QSettings().remove(key)
            else:
                qt.QSettings().setValue(key, value)
        OMEZarrLogic.resetDisplayUnits()

    def runTest(self):
        self.setUp()
        try:
            self.test_ReaderRegistration()
            self.test_RoundTripFromSlicerVolume()
            self.test_LevelSelection()
            self.test_RegionLoading()
            self.test_MicroscopyAxes()
            self.test_Labels()
            self.test_TimeSeries()
            self.test_Writer()
            self.test_RefineView()
            self.test_AutoRefine()
            self.test_MultiViewRefine()
            self.test_LabelsAsSegmentation()
            self.test_Bioformats2raw()
            self.test_LabelMapDetection()
            self.test_RefineSkipsWhenNotFiner()
            self.test_StorageOptions()
            self.test_SegmentationWriter()
            self.test_WidgetInspectsByItself()
            self.test_ReadsKeepTheApplicationResponsive()
            self.test_Settings()
            self.test_CancelledLoadLeavesNothing()
            self.test_ObliqueRoundTrip()
            self.test_Streaming()
            self.test_StreamingCompletes()
            self.test_StreamingViewsFirst()
            self.test_FillAnd3DTakeTurns()
            self.test_3DDepthFollowsOpacity()
            self.test_RequestPacing()
            self.test_StreamedVolumeRendering()
            self.test_StreamedVolumeRenderingGpuLimits()
            self.test_StreamingShardedRemote()
            if os.environ.get("OMEZARR_TEST_REMOTE"):
                self.test_RemoteStore()
        finally:
            self.tearDown()
        self.delayDisplay("OMEZarr tests passed")

    # helpers

    def writeMRHeadStore(self, chunks=None):
        import ngff_zarr
        import SampleData

        mrHead = SampleData.SampleDataLogic().downloadMRHead()
        image = OMEZarrLogic.ngffImageFromVolumeNode(mrHead, name="MRHead")
        if chunks:
            multiscales = ngff_zarr.to_multiscales(image, scale_factors=[2, 4], chunks=chunks)
        else:
            multiscales = ngff_zarr.to_multiscales(image, scale_factors=[2, 4])
        storePath = os.path.join(self.tempDir, "MRHead.ome.zarr")
        ngff_zarr.to_ome_zarr(storePath, multiscales, overwrite=True)
        OMEZarrLogic.clearCache()
        return mrHead, storePath

    @staticmethod
    def ijkToRasArray(volumeNode):
        matrix = vtk.vtkMatrix4x4()
        volumeNode.GetIJKToRASMatrix(matrix)
        return slicer.util.arrayFromVTKMatrix(matrix)

    @staticmethod
    def saveAsOmeZarr(node, path):
        # slicer.util.saveNode ignores properties["fileType"] (it tests hasattr on a dict),
        # so select the writer through the IO manager as the save dialog does.
        messages = slicer.vtkMRMLMessageCollection()
        success = slicer.app.coreIOManager().saveNodes("OMEZarr", {"nodeID": node.GetID(), "fileName": path}, messages)
        if not success:
            logging.error(messages.GetAllMessagesAsString())
        return success

    # tests

    def test_ReaderRegistration(self):
        self.delayDisplay("Reader registration")
        ioManager = slicer.app.coreIOManager()
        self.assertEqual(str(ioManager.fileTypeFromDescription("OME-Zarr image")), "OMEZarr")
        self.assertIsNone(omeZarrRootFromPath(self.tempDir))
        self.assertEqual(omeZarrRootFromPath("https://example.org/a/b.ome.zarr/"), "https://example.org/a/b.ome.zarr")
        self.assertGreater(OMEZarrLogic.maxBytesFromSettings(), 0)

    def test_RoundTripFromSlicerVolume(self):
        self.delayDisplay("Round trip Slicer volume -> OME-Zarr -> Slicer")
        mrHead, storePath = self.writeMRHeadStore()
        ioManager = slicer.app.coreIOManager()
        # This is the detection path used by Add Data and drag-and-drop.
        self.assertEqual(str(ioManager.fileType(storePath)), "OMEZarr")
        self.assertEqual(str(ioManager.fileType(os.path.join(storePath, "zarr.json"))), "OMEZarr")

        loaded = slicer.util.loadNodeFromFile(storePath, "OMEZarr")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.GetAttribute("OMEZarr.Level"), "0")
        self.assertEqual(loaded.GetAttribute("OMEZarr.OrientationSource"), "rfc4")
        np.testing.assert_allclose(self.ijkToRasArray(loaded), self.ijkToRasArray(mrHead), atol=1e-6)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(loaded), slicer.util.arrayFromVolume(mrHead))

        # Automatic detection without an explicit file type.
        loadedAuto = slicer.util.loadNodeFromFile(storePath)
        self.assertIsNotNone(loadedAuto)
        self.assertTrue(samePath(loadedAuto.GetAttribute("OMEZarr.Path"), storePath))

    def test_LevelSelection(self):
        self.delayDisplay("Multiscale level selection by memory budget")
        mrHead, storePath = self.writeMRHeadStore()
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        self.assertEqual(len(multiscales.images), 3)
        full = OMEZarrLogic.volumeBytes(multiscales.images[0])
        self.assertEqual(OMEZarrLogic.selectLevel(multiscales, full), 0)
        self.assertEqual(OMEZarrLogic.selectLevel(multiscales, full // 4), 1)
        self.assertEqual(OMEZarrLogic.selectLevel(multiscales, full, copies=2), 1)
        self.assertEqual(OMEZarrLogic.selectLevel(multiscales, 1), 2)

        node = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": full // 4})
        self.assertEqual(node.GetAttribute("OMEZarr.Level"), "1")
        np.testing.assert_allclose(np.array(node.GetSpacing()), 2.0 * np.array(mrHead.GetSpacing()), rtol=1e-6)
        # A downsampled level must cover the same physical extent.
        boundsFull, boundsLevel = [0.0] * 6, [0.0] * 6
        mrHead.GetRASBounds(boundsFull)
        node.GetRASBounds(boundsLevel)
        np.testing.assert_allclose(boundsLevel, boundsFull, atol=max(mrHead.GetSpacing()) * 2.5)

    def test_RegionLoading(self):
        self.delayDisplay("Region-of-interest loading reads a sub-block")
        mrHead, storePath = self.writeMRHeadStore()
        full = slicer.util.arrayFromVolume(mrHead)
        ijkToRas = self.ijkToRasArray(mrHead)
        # Sub-block in IJK: i 40..100, j 60..120, k 20..50 (half-open).
        i0, i1, j0, j1, k0, k1 = 40, 100, 60, 120, 20, 50
        corners = np.array([[i, j, k, 1.0] for i in (i0, i1 - 1) for j in (j0, j1 - 1) for k in (k0, k1 - 1)])
        ras = (ijkToRas @ corners.T).T[:, :3]
        roi = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        roi.SetCenter(*((ras.min(axis=0) + ras.max(axis=0)) / 2.0))
        roi.SetSize(*(ras.max(axis=0) - ras.min(axis=0)))

        nodes = OMEZarrLogic.loadRegion(storePath, roi, level=0)
        self.assertEqual(len(nodes), 1)
        region = slicer.util.arrayFromVolume(nodes[0])
        self.assertEqual(region.shape, (k1 - k0, j1 - j0, i1 - i0))
        np.testing.assert_array_equal(region, full[k0:k1, j0:j1, i0:i1])
        expectedOrigin = (ijkToRas @ np.array([i0, j0, k0, 1.0]))[:3]
        np.testing.assert_allclose(self.ijkToRasArray(nodes[0])[:3, 3], expectedOrigin, atol=1e-6)

    def writeMicroscopyStore(self, name="cells", withTime=False):
        import ngff_zarr
        from ngff_zarr import Omero, OmeroChannel, OmeroWindow

        rng = np.random.default_rng(0)
        shape = (3, 2, 12, 30, 40) if withTime else (2, 12, 30, 40)
        dims = ("t", "c", "z", "y", "x") if withTime else ("c", "z", "y", "x")
        data = rng.integers(0, 4000, size=shape, dtype=np.uint16)
        scale = {"z": 2.0, "y": 0.25, "x": 0.25}
        translation = {"z": 10.0, "y": -5.0, "x": 3.0}
        units = {"z": "micrometer", "y": "micrometer", "x": "micrometer"}
        if withTime:
            scale["t"], translation["t"], units["t"] = 0.5, 0.0, "second"
        image = ngff_zarr.to_ngff_image(data, dims=dims, scale=scale, translation=translation, axes_units=units)
        multiscales = ngff_zarr.to_multiscales(image, scale_factors=[2])
        multiscales.metadata.omero = Omero(
            channels=[
                OmeroChannel(color="0000FF", window=OmeroWindow(0, 4000, 100, 3000), label="DAPI"),
                OmeroChannel(color="00FF00", window=OmeroWindow(0, 4000, 200, 2500), label="GFP"),
            ]
        )
        storePath = os.path.join(self.tempDir, f"{name}.ome.zarr")
        ngff_zarr.to_ome_zarr(storePath, multiscales, overwrite=True)
        OMEZarrLogic.clearCache()
        return data, storePath

    def writeLabelGroup(self, storePath, name, shape):
        """Add a two-label ``labels/<name>`` group, with image-label metadata, to a microscopy store."""
        import ngff_zarr

        labelData = np.zeros(shape, dtype=np.uint16)
        labelData[2:6, 5:15, 10:20] = 1
        labelData[6:10, 15:25, 20:35] = 3
        labelImage = ngff_zarr.to_ngff_image(
            labelData,
            dims=("z", "y", "x"),
            scale={"z": 2.0, "y": 0.25, "x": 0.25},
            translation={"z": 10.0, "y": -5.0, "x": 3.0},
            axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},
        )
        labelPath = os.path.join(storePath, "labels", name)
        ngff_zarr.to_ome_zarr(
            labelPath,
            ngff_zarr.to_multiscales(labelImage, scale_factors=[2], method=ngff_zarr.Methods.ITKWASM_LABEL_IMAGE),
        )
        OMEZarrLogic.addImageLabelMetadata(
            labelPath,
            {
                "version": "0.5",
                "colors": [{"label-value": 1, "rgba": [255, 0, 0, 255]}, {"label-value": 3, "rgba": [0, 0, 255, 128]}],
                "properties": [{"label-value": 1, "name": "nucleus"}, {"label-value": 3, "name": "cytoplasm"}],
            },
        )
        OMEZarrLogic.registerLabelInParent(labelPath)
        OMEZarrLogic.clearCache()
        return labelData, labelPath

    def test_MicroscopyAxes(self):
        self.delayDisplay("c,z,y,x microscopy store in micrometres, two channels")
        data, storePath = self.writeMicroscopyStore()
        nodes = OMEZarrLogic.loadImage(storePath, level=0)
        self.assertEqual([n.GetName() for n in nodes], ["cells_DAPI", "cells_GFP"])
        self.assertEqual(nodes[0].GetDisplayNode().GetColorNodeID(), "vtkMRMLColorTableNodeBlue")
        self.assertEqual(nodes[1].GetDisplayNode().GetColorNodeID(), "vtkMRMLColorTableNodeGreen")
        self.assertAlmostEqual(nodes[1].GetDisplayNode().GetWindow(), 2300.0)
        self.assertEqual(nodes[0].GetAttribute("OMEZarr.OrientationSource"), "assumed-LPS")
        for channel, node in enumerate(nodes):
            np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), data[channel])
            np.testing.assert_allclose(node.GetSpacing(), [0.25e-3, 0.25e-3, 2.0e-3])
            # LPS origin (3, -5, 10) um -> RAS (-3, 5, 10) um -> mm
            np.testing.assert_allclose(node.GetOrigin(), [-3.0e-3, 5.0e-3, 10.0e-3])
            self.assertEqual(node.GetAttribute("OMEZarr.LengthUnit"), "micrometer")

        # The RAS assumption keeps x/y unflipped.
        Settings.set(Settings.ORIENTATION, "RAS")
        node = OMEZarrLogic.loadImage(storePath, level=0, channels=[0])[0]
        self.assertEqual(node.GetAttribute("OMEZarr.OrientationSource"), "assumed-RAS")
        np.testing.assert_allclose(node.GetOrigin(), [3.0e-3, -5.0e-3, 10.0e-3])
        Settings.set(Settings.ORIENTATION, "LPS")

    def test_Labels(self):
        self.delayDisplay("Labels group loads as a label map with its colour table")
        data, storePath = self.writeMicroscopyStore("labelled")
        labelData, labelPath = self.writeLabelGroup(storePath, "nuclei", data.shape[1:])
        self.assertEqual(labelGroupNames(storePath), ["nuclei"])
        self.assertTrue(isLabelStore(labelPath))

        nodes = OMEZarrLogic.loadImage(storePath, level=0)
        labelNodes = labelMaps(nodes)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(labelNodes), 1)
        labelNode = labelNodes[0]
        self.assertEqual(labelNode.GetName(), "labelled_nuclei")
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(labelNode), labelData)
        np.testing.assert_allclose(self.ijkToRasArray(labelNode), self.ijkToRasArray(nodes[0]), atol=1e-9)
        colorNode = labelNode.GetDisplayNode().GetColorNode()
        self.assertEqual(colorNode.GetColorName(1), "nucleus")
        self.assertEqual(colorNode.GetColorName(3), "cytoplasm")
        rgba = [0.0] * 4
        colorNode.GetColor(3, rgba)
        np.testing.assert_allclose(rgba, [0.0, 0.0, 1.0, 128 / 255.0], atol=1e-6)

        # Region loading and a lower level carry the labels along.
        nodes = OMEZarrLogic.loadImage(storePath, level=1, channels=[0])
        self.assertEqual([n.GetAttribute("OMEZarr.Level") for n in nodes], ["1", "1"])
        self.assertTrue(nodes[1].IsA("vtkMRMLLabelMapVolumeNode"))
        np.testing.assert_allclose(nodes[1].GetSpacing(), nodes[0].GetSpacing())

        # A label store dropped on its own loads as a label map too.
        direct = slicer.util.loadNodeFromFile(labelPath, "OMEZarr")
        self.assertTrue(direct.IsA("vtkMRMLLabelMapVolumeNode"))

        # The setting turns labels off.
        nodes = OMEZarrLogic.loadImage(storePath, level=0, labels=False)
        self.assertEqual(len(nodes), 2)

    def test_TimeSeries(self):
        self.delayDisplay("Time axis loads as a Sequence, or as one time point")
        data, storePath = self.writeMicroscopyStore("timelapse", withTime=True)
        nodes = OMEZarrLogic.loadImage(storePath, level=0)
        self.assertEqual(len(nodes), 2)
        browsers = slicer.util.getNodesByClass("vtkMRMLSequenceBrowserNode")
        self.assertEqual(len(browsers), 1)
        browser = browsers[0]
        sequence = browser.GetMasterSequenceNode()
        self.assertEqual(sequence.GetNumberOfDataNodes(), 3)
        self.assertEqual(sequence.GetIndexUnit(), "second")
        self.assertEqual(sequence.GetNthIndexValue(2), "1")
        browser.SetSelectedItemNumber(2)
        slicer.modules.sequences.logic().UpdateProxyNodesFromSequences(browser)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(nodes[0]), data[2, 0])
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(nodes[1]), data[2, 1])

        # Refinement follows the browser's selected time point (a coarse level is shown so
        # that there is something finer to load).
        self.assertEqual(OMEZarrLogic.currentTimeIndex(storePath), 2)
        coarse = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"level": 1, "timeIndex": 0})
        slicer.util.setSliceViewerLayers(background=coarse, fit=True)
        refinedT = OMEZarrLogic.refineView(storePath, "Red")
        self.assertEqual(refinedT[0].GetAttribute("OMEZarr.TimeIndex"), "2")
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        region = OMEZarrLogic.regionFromRasBounds(multiscales.images[0], OMEZarrLogic.sliceViewRasBounds("Red"))
        expected = data[2, 0][tuple(slice(*region[d]) for d in ("z", "y", "x"))]
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(refinedT[0]), expected)

        # Budget accounts for every time point and channel.
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        single = OMEZarrLogic.volumeBytes(multiscales.images[0])
        self.assertEqual(OMEZarrLogic.selectLevel(multiscales, single * 6, copies=6), 0)
        self.assertEqual(OMEZarrLogic.selectLevel(multiscales, single * 6 - 1, copies=6), 1)

        node = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"level": 0, "timeIndex": 1})
        self.assertFalse(node.IsA("vtkMRMLSequenceNode"))
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), data[1, 0])
        self.assertEqual(node.GetAttribute("OMEZarr.TimeIndex"), "1")

        Settings.set(Settings.TIME_MODE, "index")
        node = OMEZarrLogic.loadImage(storePath, level=0, channels=[1])[0]
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), data[0, 1])
        Settings.set(Settings.TIME_MODE, "sequence")

    def test_Writer(self):
        self.delayDisplay("Scalar and label map volumes are written and read back")
        import SampleData

        mrHead = SampleData.SampleDataLogic().downloadMRHead()
        storePath = os.path.join(self.tempDir, "written.ome.zarr")
        self.assertTrue(self.saveAsOmeZarr(mrHead, storePath))
        self.assertTrue(os.path.isfile(os.path.join(storePath, "zarr.json")))
        loaded = OMEZarrLogic.loadImage(storePath, level=0)[0]
        np.testing.assert_allclose(self.ijkToRasArray(loaded), self.ijkToRasArray(mrHead), atol=1e-6)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(loaded), slicer.util.arrayFromVolume(mrHead))
        self.assertGreater(len(OMEZarrLogic.openMultiscales(storePath).images), 1)

        labelArray = np.zeros(slicer.util.arrayFromVolume(mrHead).shape, dtype=np.uint8)
        labelArray[40:80, 100:150, 100:160] = 2
        labelNode = slicer.util.addVolumeFromArray(
            labelArray, ijkToRAS=self.ijkToRasArray(mrHead), name="mask", nodeClassName="vtkMRMLLabelMapVolumeNode"
        )
        labelNode.CreateDefaultDisplayNodes()
        colorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", "maskColors")
        colorNode.SetTypeToUser()
        colorNode.SetNumberOfColors(3)
        colorNode.SetColor(2, "lesion", 1.0, 0.5, 0.0, 1.0)
        labelNode.GetDisplayNode().SetAndObserveColorNodeID(colorNode.GetID())
        labelPath = os.path.join(storePath, "labels", "mask")
        self.assertTrue(self.saveAsOmeZarr(labelNode, labelPath))
        self.assertTrue(isLabelStore(labelPath))
        self.assertEqual(labelGroupNames(storePath), ["mask"])
        imageLabel = readStoreAttributes(labelPath)["image-label"]
        self.assertEqual(imageLabel["properties"], [{"label-value": 2, "name": "lesion"}])
        self.assertEqual(imageLabel["colors"][0]["rgba"], [255, 128, 0, 255])

        OMEZarrLogic.clearCache()
        nodes = OMEZarrLogic.loadImage(storePath, level=0)
        self.assertEqual(len(nodes), 2)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(nodes[1]), labelArray)
        self.assertEqual(nodes[1].GetDisplayNode().GetColorNode().GetColorName(2), "lesion")

    def test_RefineView(self):
        self.delayDisplay("Refining a slice view reloads its block at a finer level")
        mrHead, storePath = self.writeMRHeadStore()
        full = slicer.util.arrayFromVolume(mrHead)
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        budget = OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4
        coarse = OMEZarrLogic.loadImage(storePath, maxBytes=budget)[0]
        self.assertEqual(coarse.GetAttribute("OMEZarr.Level"), "1")
        slicer.util.setSliceViewerLayers(background=coarse, fit=True)

        sliceNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode()
        sliceNode.SetOrientationToAxial()
        center = np.array(mrHead.GetOrigin())
        bounds = [0.0] * 6
        mrHead.GetRASBounds(bounds)
        center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        sliceNode.JumpSliceByCentering(*center)
        sliceNode.SetFieldOfView(40.0, 40.0, 1.0)

        nodes = OMEZarrLogic.refineView(storePath, "Red", maxBytes=budget)
        self.assertEqual(len(nodes), 1)
        refined = nodes[0]
        self.assertEqual(refined.GetAttribute("OMEZarr.Level"), "0")
        self.assertEqual(refined.GetAttribute("OMEZarr.Refined"), "1")
        array = slicer.util.arrayFromVolume(refined)
        self.assertLess(array.size, full.size // 8)
        # The refined block is a verbatim sub-block of the full-resolution image.
        rasToIjk = np.linalg.inv(self.ijkToRasArray(mrHead))
        start = np.rint((rasToIjk @ np.append(self.ijkToRasArray(refined)[:3, 3], 1.0))[:3]).astype(int)
        k, j, i = array.shape
        np.testing.assert_array_equal(
            array, full[start[2] : start[2] + k, start[1] : start[1] + j, start[0] : start[0] + i]
        )
        redLogic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
        self.assertEqual(redLogic.GetForegroundLayer().GetVolumeNode().GetID(), refined.GetID())
        self.assertEqual(refined.GetAttribute("OMEZarr.RefinedView"), "Red")

        # A second refinement of the same view replaces the first.
        nodes = OMEZarrLogic.refineView(storePath, "Red", maxBytes=budget)
        self.assertEqual(len(OMEZarrLogic.refinedNodes(storePath)), 1)

    @staticmethod
    def waitFor(condition, timeoutSeconds=10.0):
        import time

        deadline = time.time() + timeoutSeconds
        while time.time() < deadline:
            slicer.app.processEvents()
            if condition():
                return True
            time.sleep(0.05)
        return condition()

    def test_AutoRefine(self):
        self.delayDisplay("Automatic refinement follows the Red view")
        mrHead, storePath = self.writeMRHeadStore()
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        budget = OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4
        coarse = OMEZarrLogic.loadImage(storePath, maxBytes=budget)[0]
        slicer.util.setSliceViewerLayers(background=coarse, fit=True)
        sliceNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode()
        sliceNode.SetOrientationToAxial()
        bounds = [0.0] * 6
        mrHead.GetRASBounds(bounds)
        center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]
        sliceNode.JumpSliceByCentering(*center)
        sliceNode.SetFieldOfView(40.0, 40.0, 1.0)

        def refined():
            return OMEZarrLogic.refinedNodes(storePath)

        refiner = OMEZarrLogic.startAutoRefine(storePath, "Red", delayMs=200, maxBytes=budget)
        # The layout may still settle while the first block loads; wait until the refiner rests.
        self.assertTrue(self.waitFor(lambda: refiner.refreshCount >= 1 and refiner.idle()))
        first = refined()
        self.assertEqual(len(first), 1)
        firstOrigin = np.array(first[0].GetOrigin())

        # A still view does not reload.
        settled = refiner.refreshCount
        self.assertFalse(self.waitFor(lambda: refiner.refreshCount > settled, timeoutSeconds=1.0))

        # Panning reloads once the view is still again, replacing the previous block.
        sliceNode.JumpSliceByCentering(center[0] + 25.0, center[1], center[2])
        self.assertTrue(self.waitFor(lambda: refiner.refreshCount > settled and refiner.idle()))
        second = refined()
        self.assertEqual(len(second), 1)
        self.assertGreater(np.abs(np.array(second[0].GetOrigin()) - firstOrigin).max(), 10.0)

        OMEZarrLogic.stopAutoRefine(storePath)
        self.assertIsNone(OMEZarrLogic.autoRefiner(storePath))
        stopped = refiner.refreshCount
        sliceNode.JumpSliceByCentering(center[0] - 25.0, center[1], center[2])
        self.assertFalse(self.waitFor(lambda: refiner.refreshCount > stopped, timeoutSeconds=1.0))

    def centerRedViewOn(self, volumeNode, fieldOfView):
        sliceNode = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode()
        sliceNode.SetOrientationToAxial()
        bounds = [0.0] * 6
        volumeNode.GetRASBounds(bounds)
        sliceNode.JumpSliceByCentering(
            (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2
        )
        sliceNode.SetFieldOfView(fieldOfView, fieldOfView, 1.0)
        return sliceNode

    def test_Streaming(self):
        self.delayDisplay("Streaming shows the coarsest level at once and each view's plane at screen resolution")
        mrHead, storePath = self.writeMRHeadStore(chunks=32)
        full = slicer.util.arrayFromVolume(mrHead)
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        budget = OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4  # the budget allows level 1
        Settings.set(Settings.STREAM, True)

        node = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": budget})
        streamer = OMEZarrLogic.streamer(storePath)
        self.assertIsNotNone(streamer)
        self.assertEqual(streamer.target, 1)
        self.assertEqual(streamer.shownLevel, 2)

        # Zoomed in, the Red view needs more detail than level 1 has: it gets its own level-0 plane.
        sliceNode = self.centerRedViewOn(mrHead, 40.0)
        self.assertTrue(
            self.waitFor(lambda: streamer.complete and streamer.views.get("Red", {}).get("shown", False), 20.0)
        )
        self.assertEqual(node.GetAttribute("OMEZarr.Level"), "1")
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), np.asarray(multiscales.images[1].data))

        overlay = streamer.overlays["Red"]
        self.assertEqual(overlay.GetAttribute("OMEZarr.Level"), "0")
        request = streamer.views["Red"]
        self.assertLess(len(request["keys"]), len(streamer.levels[0].keys()) // 8)  # only what the view shows
        array = slicer.util.arrayFromVolume(overlay)
        self.assertLessEqual(min(array.shape), 4)  # a thin slab around the plane, not a block
        rasToIjk = np.linalg.inv(self.ijkToRasArray(mrHead))
        start = np.rint((rasToIjk @ np.append(self.ijkToRasArray(overlay)[:3, 3], 1.0))[:3]).astype(int)
        k, j, i = array.shape
        np.testing.assert_array_equal(
            array, full[start[2] : start[2] + k, start[1] : start[1] + j, start[0] : start[0] + i]
        )
        redLogic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
        self.assertEqual(redLogic.GetForegroundLayer().GetVolumeNode().GetID(), overlay.GetID())

        # Scrolling to another slice reads the new plane.
        sliceNode.SetSliceOffset(sliceNode.GetSliceOffset() + 10.0)
        self.assertTrue(
            self.waitFor(
                lambda: streamer.views.get("Red", {}).get("region") not in (None, request["region"])
                and streamer.views["Red"]["shown"]
            )
        )

        # Empty chunks are kept as marks, not as blocks of zeros; the level table reports what was read.
        self.assertTrue(all(block.any() for block in streamer.cache.values()))
        self.assertEqual(streamer.levelState(2)[0], "✓")  # shown first, read whole
        self.assertEqual(streamer.levelState(1)[0], "✓")  # filled in the background
        self.assertTrue(streamer.levelState(0)[0].startswith("◐"))  # only the planes the views needed

        # Zoomed out, level 1 is detailed enough and the view's own plane goes away.
        sliceNode.SetFieldOfView(2000.0, 2000.0, 1.0)
        self.assertTrue(self.waitFor(lambda: "Red" not in streamer.overlays))
        self.assertIsNone(overlay.GetScene())

        OMEZarrLogic.stopStreaming(storePath)
        self.assertIsNone(OMEZarrLogic.streamer(storePath))

    def test_StreamingCompletes(self):
        self.delayDisplay("Streaming ends with the full-resolution volume and nothing else in the scene")
        mrHead, storePath = self.writeMRHeadStore(chunks=32)
        Settings.set(Settings.STREAM, True)
        self.centerRedViewOn(mrHead, 40.0)
        # Hold every chunk read, so the load is seen returning before any of them.
        gate = threading.Event()
        read = LevelChunks.read
        LevelChunks.read = lambda chunks, key: gate.wait(20.0) and read(chunks, key)
        try:
            node = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": 1 << 30})
            self.assertEqual(node.GetAttribute("OMEZarr.Level"), "2")  # shown at once, from the coarsest level
            self.assertEqual(OMEZarrLogic.streamer(storePath).target, 0)
            gate.set()
            shown = []

            def finished():
                level = node.GetAttribute("OMEZarr.Level")
                if not shown or shown[-1] != level:
                    shown.append(level)
                return OMEZarrLogic.streamer(storePath) is None

            self.assertTrue(self.waitFor(finished, 20.0))
            self.assertEqual(shown, ["2", "1", "0"])  # sharpened level by level, not straight to the target
        finally:
            LevelChunks.read = read
            gate.set()
        self.assertEqual(node.GetAttribute("OMEZarr.Level"), "0")
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), slicer.util.arrayFromVolume(mrHead))
        np.testing.assert_allclose(self.ijkToRasArray(node), self.ijkToRasArray(mrHead), atol=1e-6)
        streamed = [n for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode") if n.GetAttribute("OMEZarr.Streamed")]
        self.assertEqual(streamed, [])

        # A level picked in the module panel is shown at once, and the budget's finer level streamed in.
        picked = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"level": 2, "maxBytes": 1 << 30})
        self.assertEqual(picked.GetAttribute("OMEZarr.Level"), "2")
        self.assertTrue(self.waitFor(lambda: picked.GetAttribute("OMEZarr.Level") == "0", 20.0))
        # Picking the level the budget allows (or a finer one) loads it directly.
        direct = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"level": 0, "maxBytes": 1 << 30})
        self.assertEqual(direct.GetAttribute("OMEZarr.Level"), "0")
        self.assertIsNone(OMEZarrLogic.streamer(storePath))

        # With streaming off, the same load returns the full level straight away.
        Settings.set(Settings.STREAM, False)
        direct = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": 1 << 30})
        self.assertEqual(direct.GetAttribute("OMEZarr.Level"), "0")
        self.assertIsNone(OMEZarrLogic.streamer(storePath))

    def test_StreamingViewsFirst(self):
        self.delayDisplay("Slice views are read before anything else, and sharpen as their chunks arrive")
        import time

        mrHead, storePath = self.writeMRHeadStore(chunks=32)
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        budget = OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4  # level 1 fills in the background
        Settings.set(Settings.STREAM, True)
        sliceNode = self.centerRedViewOn(mrHead, 2000.0)
        bounds = [0.0] * 6
        mrHead.GetRASBounds(bounds)
        center = [(bounds[2 * axis] + bounds[2 * axis + 1]) / 2 for axis in range(3)]

        # Background reads wait for `background`; the slice views' reads take `fineSeconds` each.
        background = threading.Event()
        fineSeconds = [0.0]
        starts, fineEnds = [], []
        readChunk = Streamer.readChunk

        def gatedRead(streamer, level, key, forView):
            starts.append((time.monotonic(), forView))
            if forView:
                time.sleep(fineSeconds[0])
            else:
                background.wait(20.0)
            block = readChunk(streamer, level, key, forView)
            if forView:
                fineEnds.append(time.monotonic())
            return block

        Streamer.readChunk = gatedRead
        try:
            slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": budget})
            streamer = OMEZarrLogic.streamer(storePath)
            self.assertEqual((streamer.shownLevel, streamer.fillLevel), (2, 1))
            # Zoomed out (loading fits the views to the volume): the views need nothing finer yet.
            for viewName in ("Red", "Green", "Yellow"):
                slicer.app.layoutManager().sliceWidget(viewName).mrmlSliceNode().SetFieldOfView(2000.0, 2000.0, 1.0)
            self.assertTrue(self.waitFor(lambda: streamer.views == {}))
            # The background fill holds at most half the readers, however many chunks it has queued...
            self.assertTrue(self.waitFor(lambda: streamer.backgroundInFlight == STREAM_BACKGROUND_READERS))
            self.assertFalse(self.waitFor(lambda: streamer.backgroundInFlight > STREAM_BACKGROUND_READERS, 0.5))
            # ...so a view that needs level 0 is read at once by the readers left free.
            sliceNode.SetFieldOfView(40.0, 40.0, 1.0)
            self.assertTrue(self.waitFor(lambda: streamer.views.get("Red", {}).get("shown", False), 10.0))
            self.assertEqual(streamer.overlays["Red"].GetAttribute("OMEZarr.Level"), "0")
            self.assertEqual(sum(1 for _started, fine in starts if not fine), STREAM_BACKGROUND_READERS)

            # Panned to new chunks: while the view waits for them, no background read starts, and
            # its plane shows the new place at once, sharpening as the chunks come in.
            fineSeconds[0] = 0.5
            first = streamer.views["Red"]
            sliceNode.JumpSliceByCentering(center[0] + 60.0, center[1], center[2])
            self.assertTrue(self.waitFor(lambda: streamer.views.get("Red") is not first, 5.0))
            released = time.monotonic()
            background.set()
            drawnPartly = []

            def panned():
                request = streamer.views.get("Red")
                if request is None or request is first:
                    return False
                overlay = streamer.overlays.get("Red")
                if not request["shown"] and request.get("drawn", -1) >= 0 and overlay is not None:
                    level, region = request["level"], request["region"]
                    expected = (self.ijkToRasArray(overlay)[:3, 3], (streamer.ijkToRas[level] @ np.array(
                        [region[2][0], region[1][0], region[0][0], 1.0]))[:3])
                    drawnPartly.append(bool(np.allclose(*expected, atol=1e-6)))
                return request["shown"]

            self.assertTrue(self.waitFor(panned, 20.0))
            self.assertTrue(drawnPartly)
            self.assertTrue(all(drawnPartly))  # drawn where the view is now, before its chunks were all in
            lastFine = max(fineEnds)
            self.assertEqual([t for t, fine in starts if not fine and released <= t < lastFine], [])
            # Once the view has its chunks, the background fill goes on to the end.
            self.assertTrue(self.waitFor(lambda: streamer.complete, 20.0))
        finally:
            Streamer.readChunk = readChunk
            background.set()
            OMEZarrLogic.stopStreaming(storePath, wait=True)

    def test_FillAnd3DTakeTurns(self):
        self.delayDisplay("The 3D view and the background fill share the readers: neither starves the other")
        mrHead, storePath = self.writeMRHeadStore(chunks=32)
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        Settings.set(Settings.STREAM, True)
        Settings.set(Settings.STREAM_3D, True)
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        tokens = threading.Semaphore(0)  # each 3D or background read waits for one
        readChunk = Streamer.readChunk

        def gatedRead(streamer, level, key, forView):
            if not forView:
                tokens.acquire(timeout=20.0)
            return readChunk(streamer, level, key, forView)

        Streamer.readChunk = gatedRead
        try:
            slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4})
            streamer = OMEZarrLogic.streamer(storePath)
            for viewName in ("Red", "Green", "Yellow"):
                slicer.app.layoutManager().sliceWidget(viewName).mrmlSliceNode().SetFieldOfView(2000.0, 2000.0, 1.0)
            self.assertEqual(streamer.fillLevel, 1)
            self.assertTrue(self.waitFor(lambda: streamer.kindInFlight["fill"] == STREAM_BACKGROUND_READERS))

            # The 3D view then asks for eight level-0 chunks, queued behind the fill already reading.
            self.waitFor(lambda: False, 1.0)  # the 3D view's first update has run
            region = ((0, 32), (0, 32), (0, streamer.levels[0].shape[2]))
            keys = streamer.levels[0].keys(region)
            self.assertEqual(len(keys), STREAM_BACKGROUND_READERS)
            streamer.request3D = {"level": 0, "region": region, "keys": keys, "shown": False, "reuse": set()}
            streamer.requestChunks()
            self.assertEqual(streamer.kindInFlight["volume"], 0)

            # As reads finish, the freed readers go to whichever kind has fewer in flight.
            for _ in range(STREAM_BACKGROUND_READERS):
                tokens.release()
            half = STREAM_BACKGROUND_READERS // 2
            self.assertTrue(
                self.waitFor(lambda: streamer.kindInFlight["volume"] == half and streamer.kindInFlight["fill"] == half)
            )
            for _ in range(10000):
                tokens.release()
            self.assertTrue(self.waitFor(lambda: streamer.complete and streamer.request3D and streamer.request3D["shown"], 20.0))
        finally:
            Streamer.readChunk = readChunk
            for _ in range(10000):
                tokens.release()
            OMEZarrLogic.stopStreaming(storePath, wait=True)

    def test_3DDepthFollowsOpacity(self):
        self.delayDisplay("The 3D view reads as deep as its transfer function lets it see")
        mrHead, storePath = self.writeMRHeadStore(chunks=32)
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        Settings.set(Settings.STREAM, True)
        Settings.set(Settings.STREAM_3D, True)
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4})
        streamer = OMEZarrLogic.streamer(storePath)
        widget = slicer.app.layoutManager().threeDWidget(0)
        cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(widget.mrmlViewNode())
        bounds = [0.0] * 6
        mrHead.GetRASBounds(bounds)
        center = np.array([(bounds[2 * axis] + bounds[2 * axis + 1]) / 2 for axis in range(3)])
        cameraNode.SetFocalPoint(*center)
        cameraNode.SetPosition(*(center + [0.0, -400.0, 0.0]))  # outside the head, looking at its face
        cameraNode.SetViewUp(0.0, 0.0, 1.0)
        camera = cameraNode.GetCamera()
        width, height = widget.threeDView().renderWindow().GetSize()
        display = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(streamer.volume3D)
        opacity = display.GetVolumePropertyNode().GetVolumeProperty().GetScalarOpacity()

        def depthWith(points):
            opacity.RemoveAllPoints()
            for value, alpha in points:
                opacity.AddPoint(value, alpha)
            return streamer.visibleDepthRange(camera, width, height, None)

        front = 400.0 - (center[1] - bounds[2])  # depth of the volume's front face
        back = 400.0 + (bounds[3] - center[1])  # and of its back face
        opaqueNear, opaqueFar = depthWith([(0, 0.0), (30, 0.0), (31, 1.0)])
        clearNear, clearFar = depthWith([(0, 0.0), (30, 0.0), (31, 0.002)])
        self.assertGreaterEqual(opaqueNear, front - 10.0)  # starts at the skin, not at the camera
        self.assertLess(opaqueFar, back - 50.0)  # opaque: stops at the visible skin, well before the back
        self.assertGreater(clearFar, back - 10.0)  # transparent: the rays see through to the back
        self.assertIsNone(depthWith([(0, 0.0), (5000, 0.0)]))  # nothing shown: nothing to read
        OMEZarrLogic.stopStreaming(storePath, wait=True)

    def test_RequestPacing(self):
        self.delayDisplay("Requests to one server stay under its request-rate limit")
        import time

        saved = HTTP_WINDOW_REQUESTS, HTTP_WINDOW_S
        globals()["HTTP_WINDOW_REQUESTS"], globals()["HTTP_WINDOW_S"] = 5, 0.5
        try:
            gate = HttpGate(16)
            begin = time.monotonic()
            for _ in range(5):
                gate.release(gate.acquire())
            self.assertLess(time.monotonic() - begin, 0.25)  # within the window's allowance: no wait
            started = gate.acquire()  # the sixth waits until the first leaves the window
            gate.release(started)
            self.assertGreaterEqual(started - begin, 0.5 - 0.01)
            self.assertLess(started - begin, 1.0)
        finally:
            globals()["HTTP_WINDOW_REQUESTS"], globals()["HTTP_WINDOW_S"] = saved

    def test_StreamedVolumeRendering(self):
        self.delayDisplay("Streamed volume rendering holds only what the 3D view shows, at its resolution")
        import time

        mrHead, storePath = self.writeMRHeadStore(chunks=32)
        full = slicer.util.arrayFromVolume(mrHead)
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        Settings.set(Settings.STREAM, True)
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        budget = OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4  # level 1: the streamer keeps running
        Settings.set(Settings.STREAM_3D, True)  # set before loading: the store is rendered as it loads
        slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": budget})
        streamer = OMEZarrLogic.streamer(storePath)
        node3D = streamer.volume3D
        self.assertIsNotNone(node3D)
        self.assertIs(OMEZarrLogic.startVolumeRendering(storePath), node3D)
        self.assertIn(node3D.GetAttribute("OMEZarr.Level"), ("2", "1"))  # coarsest, until level 1 has streamed

        # Whenever the rendered image changes, its dimensions and voxels must agree: volume rendering
        # renders synchronously on the change and uploads that buffer to the GPU.
        mismatches = []

        def checkImage(caller=None, event=None):
            image = node3D.GetImageData()
            if image is not None and image.GetPointData().GetScalars() is not None:
                if image.GetNumberOfPoints() != image.GetPointData().GetScalars().GetNumberOfTuples():
                    mismatches.append(image.GetDimensions())

        imageObserver = node3D.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, checkImage)
        # Level 1 fits the GPU: once streamed it is the 3D fallback, sharing the volume node's voxels.
        self.assertTrue(self.waitFor(lambda: streamer.complete and streamer.contextLevel == 1, 20.0))
        self.assertEqual(node3D.GetAttribute("OMEZarr.Level"), "1")
        self.assertIs(node3D.GetImageData(), streamer.targetImageData)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node3D), np.asarray(multiscales.images[1].data))
        display = node3D.GetDisplayNode()
        self.assertTrue(display.IsA("vtkMRMLVolumeRenderingDisplayNode") and display.GetVisibility())

        widget = slicer.app.layoutManager().threeDWidget(0)
        cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(widget.mrmlViewNode())
        bounds = [0.0] * 6
        mrHead.GetRASBounds(bounds)
        center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])

        def lookFrom(distance):
            cameraNode.SetFocalPoint(*center)
            cameraNode.SetPosition(*(center + [0.0, -distance, 0.0]))
            cameraNode.SetViewUp(0.0, 0.0, 1.0)
            slicer.app.processEvents()
            deadline = time.time() + 2 * STREAM_3D_SETTLE_MS / 1000.0
            self.waitFor(lambda: time.time() > deadline, 5.0)  # let the camera settle

        def region3D():
            level, region = streamer.shown3D
            return int(level), region

        # Far away, the fallback level is detailed enough: nothing finer is read.
        lookFrom(20000.0)
        self.assertIsNone(streamer.request3D)
        self.assertEqual(node3D.GetAttribute("OMEZarr.Level"), "1")
        streamer.updateLevelLabels()
        self.assertIn("level 1", widget.threeDView().cornerAnnotation().GetText(Streamer.LEVEL_CORNER))
        redLabel = slicer.app.layoutManager().sliceWidget("Red").sliceView().cornerAnnotation()
        self.assertIn("OME-Zarr level", redLabel.GetText(Streamer.LEVEL_CORNER))

        # Close up, the view needs level 0, but only the part of the volume in front of the camera.
        lookFrom(60.0)
        self.assertTrue(self.waitFor(lambda: streamer.request3D is not None and streamer.request3D["shown"], 20.0))
        level, region = region3D()
        self.assertEqual(level, 0)
        self.assertEqual(node3D.GetAttribute("OMEZarr.Level"), "0")
        size = int(np.prod([stop - start for start, stop in region]))
        self.assertLess(size, full.size // 2)
        # In depth too: from the nearest surface the opacity shows, not through the whole head.
        low, high = streamer.regionRasBounds(level, region)
        self.assertLess(high[1] - low[1], 0.5 * (bounds[3] - bounds[2]))
        (z0, z1), (y0, y1), (x0, x1) = region
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node3D), full[z0:z1, y0:y1, x0:x1])

        # A small camera move copies what the previous texture already holds instead of reading it again.
        reused = streamer.reusedChunks
        cameraNode.SetFocalPoint(*(center + [15.0, 0.0, 0.0]))
        cameraNode.SetPosition(*(center + [15.0, -60.0, 0.0]))
        self.assertTrue(
            self.waitFor(
                lambda: streamer.request3D is not None
                and streamer.request3D["shown"]
                and streamer.shown3D[1] != region,
                20.0,
            )
        )
        self.assertGreater(streamer.reusedChunks, reused)
        (z0, z1), (y0, y1), (x0, x1) = streamer.shown3D[1]
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node3D), full[z0:z1, y0:y1, x0:x1])

        # Cropping limits it further, along the viewing direction too.
        roi = slicer.modules.volumerendering.logic().CreateROINode(display)
        roi.SetXYZ(*center)
        roi.SetRadiusXYZ(10.0, 10.0, 10.0)
        display.SetCroppingEnabled(True)
        cell = 15.0  # the camera's region is found on a lattice of about this spacing (mm) for this volume

        def insideRoi():
            if streamer.request3D is None or not streamer.request3D["shown"]:
                return False
            low, high = streamer.regionRasBounds(*streamer.shown3D)
            return bool(np.all(low >= center - 10.0 - cell) and np.all(high <= center + 10.0 + cell))

        self.assertTrue(self.waitFor(insideRoi, 20.0))
        (z0, z1), (y0, y1), (x0, x1) = streamer.shown3D[1]
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node3D), full[z0:z1, y0:y1, x0:x1])

        node3D.RemoveObserver(imageObserver)
        self.assertEqual(mismatches, [])

        OMEZarrLogic.stopVolumeRendering(storePath)
        self.assertIsNone(node3D.GetScene())
        self.assertIsNone(streamer.volume3D)

    def test_StreamedVolumeRenderingGpuLimits(self):
        self.delayDisplay("The 3D fallback is the finest whole level the GPU holds; full resolution stays put")
        import time

        mrHead, storePath = self.writeMRHeadStore(chunks=32)
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        Settings.set(Settings.STREAM, True)
        Settings.set(Settings.STREAM_3D, True)
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        widget = slicer.app.layoutManager().threeDWidget(0)
        cameraNode = slicer.modules.cameras.logic().GetViewActiveCameraNode(widget.mrmlViewNode())
        bounds = [0.0] * 6
        mrHead.GetRASBounds(bounds)
        center = np.array([(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        detect = Streamer.detectGpuLimits

        # The module setting overrides the automatic budget; 0 means the cube of the largest texture side.
        slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": 1 << 30})
        streamer = OMEZarrLogic.streamer(storePath)
        Settings.set(Settings.STREAM_3D_MEMORY, 300)
        streamer.detectGpuLimits(widget)
        self.assertEqual(streamer.maxTextureBytes, 300 << 20)
        Settings.set(Settings.STREAM_3D_MEMORY, 0)
        streamer.detectGpuLimits(widget)
        self.assertGreater(streamer.maxTextureDim, 0)
        self.assertLessEqual(streamer.maxTextureBytes, streamer.maxTextureDim**3 * np.dtype(streamer.dtype).itemsize)
        OMEZarrLogic.stopStreaming(storePath)

        def gpu(maxDim):
            def limits(self, widget):
                self.maxTextureDim, self.maxTextureBytes = maxDim, 1 << 34
            Streamer.detectGpuLimits = limits

        def lookFrom(distance):
            cameraNode.SetFocalPoint(*center)
            cameraNode.SetPosition(*(center + [0.0, -distance, 0.0]))
            slicer.app.processEvents()
            deadline = time.time() + 2 * STREAM_3D_SETTLE_MS / 1000.0
            self.waitFor(lambda: time.time() > deadline, 5.0)

        try:
            # A GPU too small for level 0 (256 voxels a side): level 1 is read once as the fallback.
            gpu(140)
            slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": 1 << 30})
            streamer = OMEZarrLogic.streamer(storePath)
            self.assertEqual(streamer.target, 0)
            self.assertTrue(self.waitFor(lambda: streamer.contextLevel == 1 and streamer.shown3D[0] == 1, 20.0))
            np.testing.assert_array_equal(
                slicer.util.arrayFromVolume(streamer.volume3D), np.asarray(multiscales.images[1].data)
            )
            OMEZarrLogic.stopStreaming(storePath)

            # A GPU that holds level 0: once streamed it is rendered whole and no longer follows the camera.
            gpu(4096)
            slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": 1 << 30})
            streamer = OMEZarrLogic.streamer(storePath)
            self.assertTrue(self.waitFor(lambda: streamer.complete and streamer.contextLevel == 0, 20.0))
            node3D = streamer.volume3D
            self.assertIs(node3D.GetImageData(), streamer.targetImageData)
            for distance in (60.0, 20000.0, 30.0):
                lookFrom(distance)
                self.assertIsNone(streamer.request3D)
                self.assertIs(node3D.GetImageData(), streamer.targetImageData)
        finally:
            Streamer.detectGpuLimits = detect

    @staticmethod
    def serveDirectory(root, busyAnswers=0):
        """A local HTTP server for ``root`` with Range, ETag and If-None-Match; returns (base URL, server).
        ``busyAnswers``: answer that many chunk requests "503, Retry-After: 1" first.
        ``server.state["requests"]`` counts the GETs."""
        import http.server

        state = {"busy": busyAnswers, "requests": 0}
        lock = threading.Lock()

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                path = os.path.join(root, self.path.split("?")[0].lstrip("/"))
                self.send_response(200 if os.path.isfile(path) else 404)
                self.send_header("Content-Length", str(os.path.getsize(path)) if os.path.isfile(path) else "0")
                self.end_headers()

            def do_GET(self):
                path = os.path.join(root, self.path.split("?")[0].lstrip("/"))
                with lock:
                    state["requests"] += 1
                    busy = state["busy"] > 0 and "/c/" in self.path
                    state["busy"] -= busy
                if busy:
                    self.send_response(503)
                    self.send_header("Retry-After", "1")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if not os.path.isfile(path):
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                stat = os.stat(path)
                etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.end_headers()
                    return
                data = open(path, "rb").read()
                byteRange = self.headers.get("Range")
                status, total, contentRange = 200, len(data), None
                if byteRange:
                    first, last = byteRange.replace("bytes=", "").split("-")
                    start, end = (total - int(last), total - 1) if first == "" else (int(first), int(last or total - 1))
                    end = min(end, total - 1)
                    data, status, contentRange = data[start : end + 1], 206, f"bytes {start}-{end}/{total}"
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("ETag", etag)
                if contentRange:
                    self.send_header("Content-Range", contentRange)
                self.end_headers()
                self.wfile.write(data)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server.state = state
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_address[1]}", server

    def streamRedView(self, url, budget, reference):
        """Load ``url`` streamed, zoom the Red view to level 0, wait until the target level is in and
        the view shows its level-0 plane; check the voxels. Returns the streamer."""
        node = slicer.util.loadNodeFromFile(url, "OMEZarr", {"maxBytes": budget})
        streamer = OMEZarrLogic.streamer(url)
        self.centerRedViewOn(reference, 40.0)
        self.assertTrue(self.waitFor(lambda: streamer.complete and streamer.views.get("Red", {}).get("shown", False), 60.0))
        request = streamer.views["Red"]
        self.assertEqual(request["level"], 0)
        full = slicer.util.arrayFromVolume(reference)
        (z0, z1), (y0, y1), (x0, x1) = request["region"]
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(streamer.overlays["Red"]), full[z0:z1, y0:y1, x0:x1])
        self.streamedNode = node
        return streamer

    def unloadStreamed(self, url):
        OMEZarrLogic.stopStreaming(url, wait=True)
        slicer.mrmlScene.RemoveNode(self.streamedNode)
        OMEZarrLogic.clearCache()  # a new session's readers: nothing read is remembered but the disk cache

    def test_StreamingShardedRemote(self):
        self.delayDisplay("Remote stores: shard indexes and byte ranges, the disk cache, a busy server")
        import ngff_zarr
        import SampleData

        mrHead = SampleData.SampleDataLogic().downloadMRHead()
        image = OMEZarrLogic.ngffImageFromVolumeNode(mrHead, name="MRHead")
        multiscales = ngff_zarr.to_multiscales(image, scale_factors=[2, 4], chunks=32)
        ngff_zarr.to_ome_zarr(os.path.join(self.tempDir, "sharded.ome.zarr"), multiscales, chunks_per_shard=2, overwrite=True)
        ngff_zarr.to_ome_zarr(os.path.join(self.tempDir, "plain.ome.zarr"), multiscales, overwrite=True)
        base, server = self.serveDirectory(self.tempDir)
        try:
            Settings.set(Settings.STREAM, True)
            Settings.set(Settings.STREAM_3D, False)  # the slice views alone read the same chunks every time
            budget = OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4  # level 1: level 0 only for the views
            url = f"{base}/plain.ome.zarr"
            streamer = self.streamRedView(url, budget, mrHead)
            self.assertFalse(streamer.levels[0].reader.sharded)  # plain chunks go through the same reader
            self.unloadStreamed(url)

            url = f"{base}/sharded.ome.zarr"
            streamer = self.streamRedView(url, budget, mrHead)
            reader = streamer.levels[0].reader
            self.assertTrue(reader.sharded)
            self.assertEqual(reader.perShard, (2, 2, 2))
            # One index per shard touched, then one range per non-empty chunk: never a whole shard.
            keys = {key for r in streamer.views.values() if r["level"] == 0 for key in r["keys"]}
            shards = {tuple(k // 2 for k in key) for key in keys}
            self.assertLessEqual(shards, set(reader.indexes))
            self.assertEqual(reader.indexReads, len(reader.indexes))  # each shard's index read once, then kept
            level0 = os.path.join(self.tempDir, "sharded.ome.zarr", "scale0", "MRHead", "c")
            shardFiles = [os.path.join(level0, *map(str, key)) for key in reader.indexes]
            shardBytes = sum(os.path.getsize(f) for f in shardFiles if os.path.isfile(f))
            self.assertLess(reader.bytesRead, shardBytes, f"{reader.bytesRead} bytes read of {shardBytes}")
            self.unloadStreamed(url)

            # Opened again: every chunk comes from the disk cache; the only requests check that the
            # shards are unchanged (304).
            streamer = self.streamRedView(url, budget, mrHead)
            readers = [grid.reader for grid in streamer.levels]
            self.assertEqual(sum(r.requests - r.indexReads for r in readers), 0)
            self.assertGreater(sum(r.cacheHits for r in readers), 0)
            self.unloadStreamed(url)

            # A store replaced on the server (new ETags) is read again, not served from the cache.
            for path in shardFiles:
                if os.path.isfile(path):
                    os.utime(path, (time.time() + 10, time.time() + 10))
            streamer = self.streamRedView(url, budget, mrHead)
            reader = streamer.levels[0].reader
            self.assertGreater(reader.requests, reader.indexReads)
            self.unloadStreamed(url)
        finally:
            OMEZarrLogic.stopStreaming()
            server.shutdown()

        # A busy server: its 503s are waited out (Retry-After), with fewer requests at a time.
        DiskCache.instance().clear()
        base, server = self.serveDirectory(self.tempDir, busyAnswers=20)
        try:
            url = f"{base}/sharded.ome.zarr"
            streamer = self.streamRedView(url, budget, mrHead)
            gate = streamer.client.gate
            self.assertEqual(gate.busyAnswers, 20)
            self.assertFalse(streamer.failed)
            self.unloadStreamed(url)
        finally:
            OMEZarrLogic.stopStreaming()
            server.shutdown()

    def test_MultiViewRefine(self):
        self.delayDisplay("Each slice view keeps its own refined block with the coarse window/level")
        mrHead, storePath = self.writeMRHeadStore()
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        budget = OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4
        coarse = OMEZarrLogic.loadImage(storePath, maxBytes=budget)[0]
        coarse.GetDisplayNode().SetAutoWindowLevel(False)
        coarse.GetDisplayNode().SetWindowLevel(123.0, 45.0)
        slicer.util.setSliceViewerLayers(background=coarse, fit=True)
        layoutManager = slicer.app.layoutManager()
        for viewName in ("Red", "Green"):
            layoutManager.sliceWidget(viewName).mrmlSliceNode().SetFieldOfView(40.0, 40.0, 1.0)
        red = OMEZarrLogic.refineView(storePath, "Red", maxBytes=budget)[0]
        green = OMEZarrLogic.refineView(storePath, "Green", maxBytes=budget)[0]
        self.assertEqual(len(OMEZarrLogic.refinedNodes(storePath)), 2)
        self.assertEqual(
            layoutManager.sliceWidget("Red").sliceLogic().GetForegroundLayer().GetVolumeNode().GetID(), red.GetID()
        )
        self.assertEqual(
            layoutManager.sliceWidget("Green").sliceLogic().GetForegroundLayer().GetVolumeNode().GetID(), green.GetID()
        )
        self.assertIsNone(layoutManager.sliceWidget("Yellow").sliceLogic().GetForegroundLayer().GetVolumeNode())
        for node in (red, green):
            self.assertAlmostEqual(node.GetDisplayNode().GetWindow(), 123.0)
            self.assertAlmostEqual(node.GetDisplayNode().GetLevel(), 45.0)
        # Refining Green again leaves Red's block alone.
        OMEZarrLogic.refineView(storePath, "Green", maxBytes=budget)
        self.assertEqual([n.GetID() for n in OMEZarrLogic.refinedNodes(storePath, "Red")], [red.GetID()])

    def test_LabelsAsSegmentation(self):
        self.delayDisplay("Labels load as a Segmentation when the setting is on")
        data, storePath = self.writeMicroscopyStore("segmented")
        _labelData, labelPath = self.writeLabelGroup(storePath, "cells", data.shape[1:])
        Settings.set(Settings.LABELS_AS_SEGMENTATION, True)
        labelMapsBefore = len(slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode"))
        nodes = OMEZarrLogic.loadImage(storePath, level=0, channels=[0])
        Settings.set(Settings.LABELS_AS_SEGMENTATION, False)
        self.assertEqual(len(nodes), 2)
        segmentation = nodes[1]
        self.assertTrue(segmentation.IsA("vtkMRMLSegmentationNode"))
        self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")), labelMapsBefore)
        segments = segmentation.GetSegmentation()
        self.assertEqual(segments.GetNumberOfSegments(), 2)
        self.assertEqual(sorted(segments.GetNthSegment(i).GetName() for i in range(2)), ["cytoplasm", "nucleus"])
        self.assertTrue(samePath(segmentation.GetAttribute("OMEZarr.Path"), labelPath))
        exported = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "check")
        slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
            segmentation, exported, slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY
        )
        self.assertEqual(set(np.unique(slicer.util.arrayFromVolume(exported)).tolist()), {0, 1, 2})
        slicer.mrmlScene.RemoveNode(exported)

    def test_Bioformats2raw(self):
        self.delayDisplay("A bioformats2raw container loads every image series")
        import ngff_zarr

        root = os.path.join(self.tempDir, "converted.zarr")
        os.makedirs(root)
        with open(os.path.join(root, ".zgroup"), "w", encoding="utf-8") as fp:
            json.dump({"zarr_format": 2}, fp)
        with open(os.path.join(root, ".zattrs"), "w", encoding="utf-8") as fp:
            json.dump({"bioformats2raw.layout": 3}, fp)
        arrays = []
        for index in range(2):
            rng = np.random.default_rng(index)
            array = rng.integers(0, 255, size=(4, 8 + index, 12), dtype=np.uint8)
            arrays.append(array)
            image = ngff_zarr.to_ngff_image(array, dims=("z", "y", "x"), scale={"z": 1.0, "y": 1.0, "x": 1.0})
            ngff_zarr.to_ome_zarr(
                os.path.join(root, str(index)), ngff_zarr.to_multiscales(image, scale_factors=[2]), version="0.4"
            )
        OMEZarrLogic.clearCache()

        self.assertEqual(os.path.normpath(omeZarrRootFromPath(root)), os.path.normpath(root))
        self.assertTrue(isBioformats2rawRoot(root))
        self.assertEqual(len(bioformats2rawSeries(root)), 2)
        self.assertEqual(str(slicer.app.coreIOManager().fileType(root)), "OMEZarr")
        nodes = OMEZarrLogic.loadImage(root, level=0)
        self.assertEqual([n.GetName() for n in nodes], ["converted_0", "converted_1"])
        for node, array in zip(nodes, arrays, strict=True):
            np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), array)
        first = slicer.util.loadNodeFromFile(root, "OMEZarr", {"level": 0})
        self.assertIsNotNone(first)

    def test_LabelMapDetection(self):
        self.delayDisplay("A plain integer mask store loads as a label map")
        import ngff_zarr

        mask = np.zeros((12, 30, 40), dtype=np.uint8)
        mask[3:9, 8:20, 10:30] = 1
        image = ngff_zarr.to_ngff_image(mask, dims=("z", "y", "x"), scale={"z": 2.0, "y": 0.25, "x": 0.25})
        storePath = os.path.join(self.tempDir, "Mask.ome.zarr")
        ngff_zarr.to_ome_zarr(storePath, ngff_zarr.to_multiscales(image, scale_factors=[2]))
        OMEZarrLogic.clearCache()
        node = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"level": 0})
        self.assertTrue(node.IsA("vtkMRMLLabelMapVolumeNode"))
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), mask)
        budgeted = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"maxBytes": mask.nbytes // 2})
        self.assertTrue(budgeted.IsA("vtkMRMLLabelMapVolumeNode"))
        self.assertEqual(budgeted.GetAttribute("OMEZarr.Level"), "1")
        # Explicit override, and the setting.
        node = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"level": 0, "asLabelMap": False})
        self.assertFalse(node.IsA("vtkMRMLLabelMapVolumeNode"))
        Settings.set(Settings.DETECT_LABEL_MAPS, False)
        node = OMEZarrLogic.loadImage(storePath, level=0)[0]
        self.assertFalse(node.IsA("vtkMRMLLabelMapVolumeNode"))
        Settings.set(Settings.DETECT_LABEL_MAPS, True)
        # A microscopy intensity image is not mistaken for labels.
        data, cellsPath = self.writeMicroscopyStore("intensity")
        self.assertFalse(OMEZarrLogic.looksLikeLabelMap(OMEZarrLogic.openMultiscales(cellsPath)))

    def test_RefineSkipsWhenNotFiner(self):
        self.delayDisplay("Refinement is refused when no finer level fits the budget")
        mrHead, storePath = self.writeMRHeadStore()
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        budget = OMEZarrLogic.volumeBytes(multiscales.images[0]) // 4
        coarse = OMEZarrLogic.loadImage(storePath, maxBytes=budget)[0]
        slicer.util.setSliceViewerLayers(background=coarse, fit=True)
        before = len(OMEZarrLogic.refinedNodes(storePath))
        with self.assertRaises(ValueError):
            OMEZarrLogic.refineView(storePath, "Red", maxBytes=budget // 64)
        self.assertEqual(len(OMEZarrLogic.refinedNodes(storePath)), before)

    def test_StorageOptions(self):
        self.delayDisplay("Remote storage options: configured JSON, anonymous S3 unless credentials are given")
        self.assertIsNone(OMEZarrLogic.storageOptions("/local/store.ome.zarr"))
        names = ("AWS_ACCESS_KEY_ID", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        saved = {name: os.environ.pop(name, None) for name in names}
        try:
            # A ~/.aws/credentials file does not count: obstore does not read it.
            self.assertEqual(OMEZarrLogic.storageOptions("s3://bucket/store.ome.zarr"), {"anon": True})
            self.assertIsNone(OMEZarrLogic.storageOptions("https://host/store.ome.zarr"))
            os.environ["AWS_ACCESS_KEY_ID"] = "AKIATEST"
            self.assertIsNone(OMEZarrLogic.storageOptions("s3://bucket/store.ome.zarr"))
            del os.environ["AWS_ACCESS_KEY_ID"]
            Settings.set(Settings.STORAGE_OPTIONS, '{"access_key_id": "AKIATEST", "secret_access_key": "x"}')
            self.assertNotIn("anon", OMEZarrLogic.storageOptions("s3://bucket/store.ome.zarr"))
            Settings.set(Settings.STORAGE_OPTIONS, '{"region": "us-west-2", "anon": false}')
            options = OMEZarrLogic.storageOptions("s3://bucket/store.ome.zarr")
            self.assertEqual(options["region"], "us-west-2")
            self.assertFalse(options["anon"])
            # The configured options apply to every remote store.
            self.assertEqual(OMEZarrLogic.storageOptions("https://host/store.ome.zarr")["region"], "us-west-2")
        finally:
            Settings.set(Settings.STORAGE_OPTIONS, "")
            for name, value in saved.items():
                if value is not None:
                    os.environ[name] = value

    def test_SegmentationWriter(self):
        self.delayDisplay("A segmentation is written as a label store with segment names and colours")
        import SampleData

        mrHead = SampleData.SampleDataLogic().downloadMRHead()
        labelArray = np.zeros(slicer.util.arrayFromVolume(mrHead).shape, dtype=np.uint8)
        labelArray[40:80, 100:150, 100:160] = 1
        labelArray[20:30, 50:70, 60:90] = 2
        labelNode = slicer.util.addVolumeFromArray(
            labelArray, ijkToRAS=self.ijkToRasArray(mrHead), name="seed", nodeClassName="vtkMRMLLabelMapVolumeNode"
        )
        segmentation = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "brain")
        segmentation.SetReferenceImageGeometryParameterFromVolumeNode(mrHead)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelNode, segmentation)
        slicer.mrmlScene.RemoveNode(labelNode)
        segments = segmentation.GetSegmentation()
        segments.GetNthSegment(0).SetName("cortex")
        segments.GetNthSegment(0).SetColor(0.9, 0.1, 0.1)
        segments.GetNthSegment(1).SetName("ventricle")
        segments.GetNthSegment(1).SetColor(0.1, 0.1, 0.9)

        storePath = os.path.join(self.tempDir, "brain.ome.zarr")
        labelMapsBefore = len(slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode"))
        self.assertTrue(self.saveAsOmeZarr(segmentation, storePath))
        self.assertTrue(isLabelStore(storePath))
        # The temporary export is removed again.
        self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")), labelMapsBefore)
        imageLabel = readStoreAttributes(storePath)["image-label"]
        self.assertEqual([p["name"] for p in imageLabel["properties"]], ["cortex", "ventricle"])
        self.assertEqual(imageLabel["colors"][1]["rgba"][:3], [26, 26, 230])

        OMEZarrLogic.clearCache()
        loaded = slicer.util.loadNodeFromFile(storePath, "OMEZarr")
        self.assertTrue(loaded.IsA("vtkMRMLLabelMapVolumeNode"))
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(loaded), labelArray)
        self.assertEqual(loaded.GetDisplayNode().GetColorNode().GetColorName(2), "ventricle")

    def test_WidgetInspectsByItself(self):
        self.delayDisplay("The module panel lists the levels without clicking Inspect")
        if slicer.util.mainWindow() is None:
            return
        mrHead, storePath = self.writeMRHeadStore()
        slicer.util.selectModule("OMEZarr")
        widget = slicer.modules.OMEZarrWidget
        widget.pathEdit.currentPath = self.tempDir  # not a store: quiet, empty
        self.assertEqual(widget.levelTable.rowCount, 0)
        self.assertFalse(widget.loadButton.enabled)
        widget.pathEdit.currentPath = storePath
        self.assertEqual(widget.levelTable.rowCount, 3)
        self.assertEqual(widget.timeIndexSpinBox.minimum, 0)
        self.assertFalse(widget.timeIndexSpinBox.isVisibleTo(widget.parent))
        self.assertTrue(widget.loadButton.enabled)
        self.assertEqual(widget.selectedLevel(), 0)  # the default budget fits the full resolution
        self.assertTrue(widget.levelTable.item(0, 1).font().bold())
        self.assertFalse(widget.levelTable.item(1, 1).font().bold())
        self.assertEqual(widget.levelTable.item(0, 1).text(), "256 × 256 × 130")
        # Entering the module with an OME-Zarr volume displayed fills the path.
        widget.pathEdit.currentPath = self.tempDir
        node = OMEZarrLogic.loadImage(storePath, level=2)[0]
        slicer.util.setSliceViewerLayers(background=node)
        widget.enter()
        self.assertTrue(samePath(widget.pathEdit.currentPath, storePath))
        self.assertEqual(widget.levelTable.rowCount, 3)
        self.assertEqual(widget.levelTable.item(2, 0).text(), "2 ✓")
        # A region of interest is created already placed, then loaded.
        slicer.app.layoutManager().sliceWidget("Red").mrmlSliceNode().SetFieldOfView(60.0, 60.0, 1.0)
        widget.onCreateRoi()
        roiNode = widget.roiSelector.currentNode()
        self.assertIsNotNone(roiNode)
        self.assertTrue(widget.loadRegionButton.enabled)
        self.assertGreater(min(roiNode.GetSize()), 0.0)

        def regionNodes():
            nodes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
            return [n for n in nodes if n.GetAttribute("OMEZarr.Region") and not n.GetAttribute("OMEZarr.Refined")]

        widget.levelTable.selectRow(0)
        regionsBefore = len(regionNodes())
        widget.onLoadRegion()
        self.assertEqual(len(regionNodes()), regionsBefore + 1)
        # A refusal is reported in the panel, not in a popup.
        widget.viewSelector.setCurrentIndex(widget.viewSelector.findData("Red"))
        self.assertEqual(widget.viewSelector.currentText, "Red (Axial)")
        Settings.set(Settings.MAX_BYTES, 1024)
        widget.onRefine()
        Settings.set(Settings.MAX_BYTES, 0)
        self.assertIn("Red:", widget.statusLabel.text)

    def test_ReadsKeepTheApplicationResponsive(self):
        self.delayDisplay("Qt events are processed while a volume is read")
        mrHead, storePath = self.writeMRHeadStore()
        ticks = []
        timer = qt.QTimer()
        timer.setInterval(1)
        timer.timeout.connect(lambda: ticks.append(1))
        timer.start()
        node = OMEZarrLogic.loadImage(storePath, level=0)[0]
        timer.stop()
        self.assertGreater(len(ticks), 0)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(node), slicer.util.arrayFromVolume(mrHead))

    def test_Settings(self):
        self.delayDisplay("Display units follow the store when enabled")
        data, storePath = self.writeMicroscopyStore("units")
        unitNode = slicer.mrmlScene.GetNodeByID("vtkMRMLUnitNodeApplicationLength")
        OMEZarrLogic.loadImage(storePath, level=0, channels=[0])
        self.assertEqual(unitNode.GetSuffix(), "mm")
        Settings.set(Settings.DISPLAY_UNITS, True)
        OMEZarrLogic.loadImage(storePath, level=0, channels=[0])
        self.assertEqual(unitNode.GetSuffix(), "µm")
        self.assertAlmostEqual(unitNode.GetDisplayCoefficient(), 1000.0)
        OMEZarrLogic.resetDisplayUnits()
        self.assertEqual(unitNode.GetSuffix(), "mm")
        Settings.set(Settings.DISPLAY_UNITS, False)

        Settings.set(Settings.MAX_BYTES, 0)
        self.assertGreater(OMEZarrLogic.maxBytesFromSettings(), FALLBACK_MAX_BYTES // 16)
        Settings.set(Settings.MAX_BYTES, 12345)
        self.assertEqual(OMEZarrLogic.maxBytesFromSettings(), 12345)

    def test_CancelledLoadLeavesNothing(self):
        self.delayDisplay("A cancelled load removes the nodes it added")
        import SampleData

        mrHead = SampleData.SampleDataLogic().downloadMRHead()
        storePath = os.path.join(self.tempDir, "cancelled.ome.zarr")
        self.assertTrue(self.saveAsOmeZarr(mrHead, storePath))
        nodeCount = slicer.mrmlScene.GetNumberOfNodes()
        with self.assertRaises(InterruptedError):
            OMEZarrLogic.loadImage(storePath, level=0, progress=lambda done, total, text=None: False)
        self.assertEqual(slicer.mrmlScene.GetNumberOfNodes(), nodeCount)

    def test_ObliqueRoundTrip(self):
        self.delayDisplay("An oblique volume keeps its rotation (OME-Zarr 0.6 affine)")
        import SampleData

        mrHead = SampleData.SampleDataLogic().downloadMRHead()
        oblique = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, mrHead, "oblique")
        transform = vtk.vtkTransform()
        transform.RotateX(17)
        transform.RotateZ(-33)
        rotation = vtk.vtkMatrix4x4()
        transform.GetMatrix(rotation)
        directions = np.zeros((3, 3))
        mrHead.GetIJKToRASDirections(directions)
        rotated = slicer.util.arrayFromVTKMatrix(rotation)[:3, :3] @ directions
        oblique.SetIJKToRASDirections(rotated.tolist())

        storePath = os.path.join(self.tempDir, "oblique.ome.zarr")
        self.assertTrue(self.saveAsOmeZarr(oblique, storePath))
        attributes = readStoreAttributes(storePath)
        self.assertEqual(attributes["version"], "0.6")
        [transformation] = attributes["multiscales"][0]["coordinateTransformations"]
        self.assertEqual(transformation["type"], "affine")

        OMEZarrLogic.clearCache()
        loaded = slicer.util.loadNodeFromFile(storePath, "OMEZarr", {"level": 0})
        self.assertEqual(loaded.GetAttribute("OMEZarr.OrientationSource"), "rfc4+affine")
        np.testing.assert_allclose(self.ijkToRasArray(loaded), self.ijkToRasArray(oblique), atol=1e-6)
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(loaded), slicer.util.arrayFromVolume(oblique))

        # Coarser levels keep the same direction.
        multiscales = OMEZarrLogic.openMultiscales(storePath)
        self.assertGreater(len(multiscales.images), 1)
        coarse, _source = OMEZarrLogic.ijkToRasMatrix(multiscales.images[1])
        fine = self.ijkToRasArray(oblique)
        np.testing.assert_allclose(
            coarse[:3, :3] / np.linalg.norm(coarse[:3, :3], axis=0),
            fine[:3, :3] / np.linalg.norm(fine[:3, :3], axis=0),
            atol=1e-6,
        )

        # An axis-aligned volume is still written as 0.5, without an affine.
        alignedPath = os.path.join(self.tempDir, "aligned.ome.zarr")
        self.assertTrue(self.saveAsOmeZarr(mrHead, alignedPath))
        attributes = readStoreAttributes(alignedPath)
        self.assertEqual(attributes["version"], "0.5")
        self.assertNotIn("coordinateTransformations", attributes["multiscales"][0])

    def test_RemoteStore(self):
        self.delayDisplay("Remote HTTPS store (IDR)")
        url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"
        node = slicer.util.loadNodeFromFile(url, "OMEZarr", {"level": 2})
        self.assertIsNotNone(node)
        self.assertEqual(node.GetAttribute("OMEZarr.Level"), "2")
        self.assertEqual(slicer.util.arrayFromVolume(node).shape, (236, 68, 67))
