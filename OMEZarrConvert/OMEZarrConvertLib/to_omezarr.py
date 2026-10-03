#!/usr/bin/env python3
"""Convert a microCT volume into a multiscale OME-Zarr store, reading it in slabs so that a volume
larger than the memory converts. Inputs:

* a folder of 2D slices (TIFF, PNG, BMP or JPEG; an NRecon *_Rec folder is read with its log),
* one multi-page TIFF file,
* an NRRD file (.nrrd, or a .nhdr header with its data file; raw, gzip or bzip2 data),
* an existing OME-Zarr store, whose coarser levels are rebuilt from its level 0.

    to_omezarr.py <input> <out.ome.zarr> [--levels N] [--voxel-size UM] [--shard 128]

Slices and TIFF files carry no geometry: give --voxel-size in micrometres (an NRecon log gives
it), the origin is 0 and the axes are taken as identity LPS. An NRRD keeps its voxel size, origin
and orientation. OME-Zarr 0.5 stores no rotation, so an oblique NRRD is refused, and the axes are
reordered and flipped so that x, y and z run toward left, posterior and superior: the store's
translation is then the same point for readers that apply the anatomical orientation and readers
that do not. Flips and swaps within a slab cost nothing; when z itself has to be reversed in a
compressed file, or comes from another axis of the file, level 0 is first copied in the file's own
order into a temporary store beside the output (as much disk as the compressed level 0).

The store is OME-Zarr 0.5 (Zarr v3): one array per level at scale{k}/<name>, 128^3 chunks, zstd,
millimetre units. --shard N groups the chunks into N^3 shards (N a multiple of 128); the default,
128, writes every chunk as its own file. Without --levels, levels are added until the coarsest is
at most 512 voxels on its longest side.

Every level is the block mean of the SOURCE voxels it covers (2^k per axis), summed exactly and
rounded once (half to even) for integer data. Level k is not computed from level k-1: rounding
once keeps every coarse voxel within 0.5 gray level of the true mean, which matters for 8-bit
data, and costs no extra reading because all levels are built from the source slab in memory.

Memory: about two slabs of the source, as deep as the shard (lcm with 2^(levels-1)). The source
is read once.
"""
import argparse
import bz2
import collections
import configparser
import glob
import gzip
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

CHUNK = 128
COARSEST_SIDE = 512  # automatic levels: the coarsest level is at most this many voxels on its longest side
ORIENTATION = {"x": "right-to-left", "y": "anterior-to-posterior", "z": "inferior-to-superior"}
SLICE_EXTENSIONS = (".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg")
SAMPLE_SLICES = 6  # level-0 slices compared with the source after writing


class ConversionError(Exception):
    """A problem with the input or the output path, reported without a traceback."""


def identity_axes():
    return [{"name": a, "type": "space", "unit": "millimeter",
             "orientation": {"type": "anatomical", "value": ORIENTATION[a]}} for a in "zyx"]


def auto_levels(shape):
    """Levels, full resolution included, until the coarsest is at most COARSEST_SIDE on its longest side."""
    levels = 1
    while max(shape) // 2 ** (levels - 1) > COARSEST_SIDE and min(shape) // 2**levels >= 1:
        levels += 1
    return levels


def slab_rows(shard, levels):
    """Source slices per slab: whole shards per write, and a multiple of every 2^k so that no
    coarse block straddles two slabs."""
    return math.lcm(shard, 2 ** (levels - 1))


def estimated_memory(shape, dtype, shard, levels):
    """Bytes the conversion needs at its peak: the slab, the coarse means and the temporaries
    (the 38 GB dry skull, 3882 slices in 512-slice slabs, peaked near 2.1 slabs)."""
    rows = min(slab_rows(shard, levels), shape[0])
    return int(2.2 * rows * shape[1] * shape[2] * np.dtype(dtype).itemsize)


def sample_slices(depth, samples=SAMPLE_SLICES, seed=0):
    rng = np.random.default_rng(seed)
    return sorted(int(z) for z in rng.choice(depth, min(samples, depth), replace=False))


#
# Slices
#


def read_slice(path):
    """One 2D slice as a numpy array; colour images whose channels are all equal are taken as grey."""
    if path.lower().endswith((".tif", ".tiff")):
        import tifffile

        image = tifffile.imread(path)
    else:
        try:
            import SimpleITK as sitk  # in Slicer's Python

            image = sitk.GetArrayFromImage(sitk.ReadImage(path))
        except ImportError:
            from PIL import Image

            image = np.asarray(Image.open(path))
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        if all(np.array_equal(image[..., 0], image[..., c]) for c in range(1, 3)):
            image = image[..., 0]
    if image.ndim != 2:
        raise ConversionError(f"{os.path.basename(path)} is not a greyscale 2D image (shape {image.shape})")
    return image


def read_nrecon_log(folder):
    """(slice files, voxel size µm, acquisition attributes, name, expected slice shape) from an NRecon *_rec.log, or None."""
    logs = glob.glob(os.path.join(folder, "*_rec.log"))
    if not logs:
        return None
    parser = configparser.ConfigParser(strict=False)
    parser.optionxform = str
    parser.read(logs[0], encoding="latin-1")
    rec, acq, system = parser["Reconstruction"], parser["Acquisition"], parser["System"]
    prefix = parser["File name convention"]["Filename Prefix"].strip()
    digits = int(parser["File name convention"].get("Filename Index Length", "8"))
    first, last = int(rec["First Section"]), int(rec["Last Section"])
    step = int(rec.get("Section to Section Step", "1"))
    extension = {"TIF": "tif", "TIFF": "tif", "BMP": "bmp", "PNG": "png", "JPG": "jpg"}.get(
        rec.get("Result File Type", "TIF").strip().upper(), "tif")
    files = [os.path.join(folder, f"{prefix}{i:0{digits}d}.{extension}") for i in range(first, last + 1, step)]
    voxel_um = float(rec["Pixel Size (um)"])
    attributes = {
        "source_log": os.path.basename(logs[0]),
        "scanner": system.get("Scanner"),
        "voxel_size_um": voxel_um,
        "source_kV": acq.get("Source Voltage (kV)", "").strip(),
        "source_uA": acq.get("Source Current (uA)", "").strip(),
        "filter": acq.get("Filter"),
        "exposure_ms": acq.get("Exposure (ms)"),
        "rotation_step_deg": acq.get("Rotation Step (deg)"),
        "study_date": acq.get("Study Date and Time"),
        "reconstruction": f'{rec.get("Reconstruction Program")} {rec.get("Program Version", "").replace("Version: ", "")}'.strip(),
        "sections": f"{first}-{last} (step {step})",
    }
    width, height = rec.get("Result Image Width (pixels)"), rec.get("Result Image Height (pixels)")
    expected = (int(height), int(width)) if width and height else None
    name = prefix.removesuffix("__rec").removesuffix("_rec").rstrip("_")
    return files, voxel_um, attributes, name, expected


def slice_files(folder):
    """The slice images of a folder, in name order, of its most common image type."""
    files = [f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(SLICE_EXTENSIONS)]
    if not files:
        raise ConversionError(f"no TIFF, PNG, BMP or JPEG slices in {folder}")
    kinds = collections.Counter(os.path.splitext(f)[1].lower().replace(".tiff", ".tif").replace(".jpeg", ".jpg") for f in files)
    kind = kinds.most_common(1)[0][0]
    same = (kind, ".tiff") if kind == ".tif" else (kind, ".jpeg") if kind == ".jpg" else (kind,)
    return sorted(f for f in files if f.lower().endswith(same))


class StackSource:
    """Slabs of a folder of 2D slices (NRecon log read when present)."""

    def __init__(self, folder, voxel_um=None, threads=8):
        nrecon = read_nrecon_log(folder)
        if nrecon:
            files, log_voxel_um, attributes, name, expected = nrecon
            voxel_um = voxel_um or log_voxel_um
        else:
            if voxel_um is None:
                raise ConversionError("the folder has no NRecon *_rec.log: give the voxel size in micrometres")
            files = slice_files(folder)
            attributes, name, expected = {"voxel_size_um": voxel_um}, os.path.basename(os.path.normpath(folder)), None
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            raise ConversionError(f"{len(missing)} slices missing, e.g. {missing[0]}")
        first = read_slice(files[0])
        if expected and expected != tuple(first.shape):
            raise ConversionError(f"slices are {first.shape[0]}x{first.shape[1]}, the log says {expected[0]}x{expected[1]}")
        self.files, self.name, self.dtype = files, name, first.dtype
        self.shape = (len(files), *first.shape)
        self.spacing = [voxel_um / 1000.0] * 3  # mm, z y x
        self.origin = [0.0] * 3
        self.axes = identity_axes()
        self.attributes = {"acquisition": attributes}
        self.pool = ThreadPoolExecutor(threads)
        self.description = f"{len(files)} slices, {voxel_um} µm isotropic"

    def slab(self, z0, z1):
        slab = np.empty((z1 - z0, *self.shape[1:]), dtype=self.dtype)

        def load(i):
            image = read_slice(self.files[z0 + i])
            if image.shape != self.shape[1:] or image.dtype != self.dtype:
                raise ConversionError(f"{os.path.basename(self.files[z0 + i])} is {image.shape} {image.dtype}, "
                                      f"the first slice {self.shape[1:]} {self.dtype}")
            slab[i] = image

        list(self.pool.map(load, range(z1 - z0)))
        return slab

    def reference(self, z):
        return read_slice(self.files[z])


class TiffFileSource:
    """Slabs of one multi-page TIFF (one page per slice, or one contiguous ImageJ stack)."""

    def __init__(self, path, voxel_um):
        import tifffile

        if voxel_um is None:
            raise ConversionError("a TIFF file carries no voxel size here: give it in micrometres")
        self.tif = tifffile.TiffFile(path)
        series = self.tif.series[0]
        shape = tuple(n for n in series.shape if n != 1) if len(series.shape) > 3 else tuple(series.shape)
        if len(shape) != 3:
            raise ConversionError(f"{os.path.basename(path)} is not a greyscale 3D stack (shape {series.shape})")
        # Uncompressed and contiguous (e.g. an ImageJ stack with one page header): read slabs straight
        # from the file; otherwise page by page.
        self.raw = _RawReader(path, series.dataoffset) if series.dataoffset is not None else None
        self.file_dtype = series.dtype.newbyteorder(self.tif.byteorder)
        if self.raw is None and len(self.tif.pages) != shape[0]:
            raise ConversionError(f"{os.path.basename(path)}: {len(self.tif.pages)} pages for {shape[0]} slices, "
                                  "and the data is not contiguous")
        self.shape, self.dtype = shape, series.dtype.newbyteorder("=")
        self.name = re.sub(r"\.(ome\.)?tiff?$", "", os.path.basename(path), flags=re.I)
        self.spacing = [voxel_um / 1000.0] * 3
        self.origin = [0.0] * 3
        self.axes = identity_axes()
        self.attributes = {"source_file": os.path.basename(path), "voxel_size_um": voxel_um}
        self.description = f"{voxel_um} µm isotropic"

    def slab(self, z0, z1):
        if self.raw is not None:
            out = np.empty((z1 - z0, *self.shape[1:]), dtype=self.file_dtype)
            self.raw.read(z0 * self.shape[1] * self.shape[2] * self.file_dtype.itemsize, out)
            return out.astype(self.dtype, copy=False)
        return self.tif.asarray(key=range(z0, z1)).reshape(z1 - z0, *self.shape[1:])

    def reference(self, z):
        return self.slab(z, z + 1)[0]


#
# NRRD
#

NRRD_TYPES = {}
for _names, _code in (
    (("signed char", "int8", "int8_t", "char"), "i1"),
    (("uchar", "unsigned char", "uint8", "uint8_t"), "u1"),
    (("short", "short int", "signed short", "signed short int", "int16", "int16_t"), "i2"),
    (("ushort", "unsigned short", "unsigned short int", "uint16", "uint16_t"), "u2"),
    (("int", "signed int", "int32", "int32_t"), "i4"),
    (("uint", "unsigned int", "uint32", "uint32_t"), "u4"),
    (("longlong", "long long", "long long int", "signed long long", "signed long long int", "int64", "int64_t"), "i8"),
    (("ulonglong", "unsigned long long", "unsigned long long int", "uint64", "uint64_t"), "u8"),
    (("float",), "f4"),
    (("double",), "f8"),
):
    for _name in _names:
        NRRD_TYPES[_name] = _code

# Sign of each axis of a NRRD space in LPS. Spaces with no anatomical meaning are taken as LPS, like ITK does.
NRRD_SPACES = {
    "left-posterior-superior": (1, 1, 1), "lps": (1, 1, 1),
    "right-anterior-superior": (-1, -1, 1), "ras": (-1, -1, 1),
    "left-anterior-superior": (1, -1, 1), "las": (1, -1, 1),
}
NRRD_UNITS_MM = {"mm": 1.0, "millimeter": 1.0, "millimeters": 1.0, "um": 1e-3, "µm": 1e-3, "micron": 1e-3,
                 "microns": 1e-3, "micrometer": 1e-3, "micrometers": 1e-3, "cm": 10.0, "m": 1000.0}
OBLIQUE_TOLERANCE = 1e-6  # an axis is aligned when its direction cosine is within this of ±1


def read_nrrd_header(path):
    """(fields with lower-case keys, byte offset where the header ends)."""
    with open(path, "rb") as f:
        magic = f.readline()
        if not magic.startswith(b"NRRD000"):
            raise ConversionError(f"{os.path.basename(path)} is not a NRRD file")
        fields = {}
        while True:
            line = f.readline()
            if not line:
                break  # a detached header may end without a blank line
            text = line.decode("latin-1").rstrip("\r\n")
            if text == "":
                break
            if text.startswith("#") or ":=" in text:
                continue  # comments and key/value pairs
            key, separator, value = text.partition(":")
            if not separator:
                raise ConversionError(f"cannot read the NRRD header line {text!r}")
            fields[key.strip().lower().replace("datafile", "data file").replace("byteskip", "byte skip")
                   .replace("lineskip", "line skip")] = value.strip()
        return fields, f.tell()


def _vectors(text):
    """'(1,0,0) none (0,1,0)' -> [[1,0,0], None, [0,1,0]]."""
    out = []
    for match in re.finditer(r"\(([^)]*)\)|none", text):
        out.append([float(v) for v in match.group(1).split(",")] if match.group(1) is not None else None)
    return out


class _RawReader:
    """Reads of uncompressed data at any offset, into the caller's buffer. Not a memory map: the
    pages of a mapped file count toward the process's memory as they are read (5.9 GB peak for a
    5.3 GB file against 1.4 GB for the same data read from gzip)."""

    def __init__(self, path, start):
        self.path, self.start = path, start

    def read(self, offset, out):
        view = memoryview(out.reshape(-1).view(np.uint8))
        with open(self.path, "rb") as f:
            f.seek(self.start + offset)
            filled = 0
            while filled < len(view):
                n = f.readinto(view[filled:])
                if not n:
                    raise ConversionError("the NRRD data ends early")
                filled += n


class _CompressedReader:
    """Sequential reads of the decompressed data; a read behind the current position starts over."""

    def __init__(self, path, start, encoding, byte_skip):
        self.path, self.start, self.encoding, self.byte_skip = path, start, encoding, byte_skip
        self.stream = None
        self.position = 0

    def _open(self):
        if self.stream is not None:
            self.stream.close()
        raw = open(self.path, "rb")
        raw.seek(self.start)
        self.stream = gzip.GzipFile(fileobj=raw) if self.encoding == "gzip" else bz2.BZ2File(raw)
        self._skip(self.byte_skip)
        self.position = 0

    def _skip(self, count):
        while count > 0:
            step = self.stream.read(min(count, 64 << 20))
            if not step:
                raise ConversionError("the NRRD data ends early")
            count -= len(step)

    def read(self, offset, out):
        """Fill ``out`` (a contiguous array) with the bytes at ``offset`` of the decompressed data."""
        if self.stream is None or offset < self.position:
            self._open()
        self._skip(offset - self.position)
        view = memoryview(out.reshape(-1).view(np.uint8))
        filled = 0
        while filled < len(view):
            n = self.stream.readinto(view[filled:])
            if not n:
                raise ConversionError("the NRRD data ends early")
            filled += n
        self.position = offset + len(view)


class NrrdFile:
    """A 3D scalar NRRD in its own index order: ``shape`` (k, j, i), slowest axis first; ``directions``
    the LPS step in mm of each of those axes; ``origin`` the LPS position of voxel 0;
    ``random_access`` whether slabs can be read in any order (raw data) or only front to back."""

    def __init__(self, path):
        fields, header_end = read_nrrd_header(path)
        name = os.path.basename(path)
        if int(fields.get("dimension", "0")) != 3:
            raise ConversionError(f"{name} has {fields.get('dimension')} dimensions; only 3D scalar volumes convert")
        sizes = [int(v) for v in fields["sizes"].split()]
        kind = fields.get("type", "").lower()
        if kind not in NRRD_TYPES:
            raise ConversionError(f"{name}: NRRD type {fields.get('type')!r} is not handled")
        dtype = np.dtype(NRRD_TYPES[kind])
        if dtype.itemsize > 1:
            dtype = dtype.newbyteorder("<" if fields.get("endian", "little").lower() == "little" else ">")
        encoding = fields.get("encoding", "raw").lower()
        encoding = {"gz": "gzip", "bz2": "bzip2", "raw": "raw", "gzip": "gzip", "bzip2": "bzip2"}.get(encoding)
        if encoding is None:
            raise ConversionError(f"{name}: NRRD encoding {fields.get('encoding')!r} is not handled (raw, gzip, bzip2 are)")

        # Geometry: the step of each index axis (i, j, k) in LPS millimetres, and the origin.
        signs = np.array(NRRD_SPACES.get(fields.get("space", "").lower(), (1, 1, 1)), dtype=float)
        units = [u.strip('"').lower() for u in re.findall(r'"[^"]*"|\S+', fields.get("space units", ""))]
        unit = NRRD_UNITS_MM.get(units[0], None) if units else 1.0
        if unit is None or any(NRRD_UNITS_MM.get(u) != unit for u in units):
            raise ConversionError(f"{name}: space units {fields.get('space units')!r} are not handled")
        if "space directions" in fields:
            steps = _vectors(fields["space directions"])
            if len(steps) != 3 or any(s is None or len(s) != 3 for s in steps):
                raise ConversionError(f"{name}: space directions {fields['space directions']!r} are not three 3D vectors")
            steps = [np.array(s) * signs * unit for s in steps]
        elif "spacings" in fields:
            spacings = [float(v) for v in fields["spacings"].split()]
            steps = [np.eye(3)[a] * spacings[a] * unit for a in range(3)]
        else:
            steps = [np.eye(3)[a] * unit for a in range(3)]
        origin = np.zeros(3)
        if "space origin" in fields:
            origin = np.array(_vectors(fields["space origin"])[0]) * signs * unit

        # Where the data is.
        data_path, start = path, header_end
        if "data file" in fields:
            target = fields["data file"]
            if target.upper().startswith("LIST") or "%" in target or len(target.split()) > 1:
                raise ConversionError(f"{name}: a volume split over several data files is not handled")
            data_path = target if os.path.isabs(target) else os.path.join(os.path.dirname(path), target)
            start = 0
        if not os.path.exists(data_path):
            raise ConversionError(f"NRRD data file {data_path} not found")
        for _ in range(int(fields.get("line skip", "0"))):
            with open(data_path, "rb") as f:
                f.seek(start)
                f.readline()
                start = f.tell()
        byte_skip = int(fields.get("byte skip", "0"))
        nbytes = sizes[0] * sizes[1] * sizes[2] * dtype.itemsize

        self.name = re.sub(r"\.(nrrd|nhdr)$", "", name, flags=re.I)
        self.file = name
        self.encoding = encoding
        self.dtype = dtype.newbyteorder("=") if dtype.itemsize > 1 else dtype
        self.shape = (sizes[2], sizes[1], sizes[0])
        self.directions = [steps[2], steps[1], steps[0]]  # k, j, i
        self.origin = origin
        if encoding == "raw":
            if byte_skip == -1:
                start = os.path.getsize(data_path) - nbytes
            else:
                start += byte_skip
            if os.path.getsize(data_path) < start + nbytes:
                raise ConversionError(f"{name}: the data file is shorter than the header says")
            self.reader = _RawReader(data_path, start)
        else:
            if byte_skip < 0:
                raise ConversionError(f"{name}: byte skip {byte_skip} with {encoding} data")
            self.reader = _CompressedReader(data_path, start, encoding, byte_skip)
        self.file_dtype = dtype
        self.random_access = encoding == "raw"

    def block(self, k0, k1):
        """Slices k0..k1-1 along the slowest axis, native order."""
        out = np.empty((k1 - k0, *self.shape[1:]), dtype=self.file_dtype)
        self.reader.read(k0 * self.shape[1] * self.shape[2] * self.file_dtype.itemsize, out)
        return out.astype(self.dtype, copy=False)


def axis_plan(directions, name="the volume"):
    """For the output axes z, y, x (superior, posterior, left): (native axis, sign, spacing mm) each."""
    plan = {}
    for native, step in enumerate(directions):
        length = float(np.linalg.norm(step))
        if length == 0:
            raise ConversionError(f"{name} has a zero voxel size along one axis")
        unit = np.asarray(step) / length
        lps = int(np.argmax(np.abs(unit)))
        if abs(unit[lps]) < 1 - OBLIQUE_TOLERANCE:
            raise ConversionError(
                f"{name} is oblique: its axes are rotated against left/posterior/superior, and OME-Zarr 0.5 stores "
                "no rotation. Resample it to an axis-aligned grid first.")
        if lps in plan:
            raise ConversionError(f"two axes of {name} point the same way")
        plan[lps] = (native, 1 if unit[lps] > 0 else -1, length)
    return [plan[2], plan[1], plan[0]]


def oriented(block, plan, native_axis_order=(0, 1, 2)):
    """``block`` (axes in ``native_axis_order``) reordered to z, y, x with the negative axes flipped."""
    order = [native_axis_order.index(p[0]) for p in plan]
    block = block.transpose(order)
    return block[tuple(slice(None, None, -1) if p[1] < 0 else slice(None) for p in plan)]


class NrrdSource:
    """An NRRD as z, y, x running toward superior, posterior, left. Slabs come straight from the
    file when z is the file's slowest axis (reversed from the end of a raw file when it runs
    inferior); otherwise level 0 is first copied in the file's order to a temporary store
    (``native_copy``) and the slabs are cut from it."""

    def __init__(self, path):
        self.nrrd = NrrdFile(path)
        self.plan = axis_plan(self.nrrd.directions, self.nrrd.file)
        self.name, self.dtype = self.nrrd.name, self.nrrd.dtype
        self.shape = tuple(self.nrrd.shape[p[0]] for p in self.plan)
        self.spacing = [p[2] for p in self.plan]
        origin = np.array(self.nrrd.origin, dtype=float)
        for native, step in enumerate(self.nrrd.directions):
            if next(p for p in self.plan if p[0] == native)[1] < 0:
                origin = origin + np.asarray(step) * (self.nrrd.shape[native] - 1)  # voxel 0 is now the far end
        self.origin = [float(origin[2]), float(origin[1]), float(origin[0])]
        self.axes = identity_axes()
        self.attributes = {"source_file": self.nrrd.file}
        z_native, z_sign, _ = self.plan[0]
        self.needs_native_copy = z_native != 0 or (z_sign < 0 and not self.nrrd.random_access)
        self.store = None  # level 0 of the native copy, once made
        self.kept = {}
        self.wanted = set()
        flips = [("x", "left"), ("y", "posterior"), ("z", "superior")]
        changes = [f"{a} flipped toward {d}" for (a, d), p in zip(reversed(flips), self.plan) if p[1] < 0]
        if [p[0] for p in self.plan] != [0, 1, 2]:
            changes.insert(0, "axes reordered")
        self.description = (f"{self.nrrd.encoding} NRRD, spacing {[round(s * 1000, 4) for s in self.spacing]} µm (z, y, x)"
                            + (f", {', '.join(changes)}" if changes else ""))

    def native(self):
        """The file in its own order, for the temporary copy."""
        nrrd = self.nrrd

        class Native:
            name, dtype, shape = nrrd.name, nrrd.dtype, nrrd.shape
            spacing, origin, axes, attributes = [1.0] * 3, [0.0] * 3, identity_axes(), {}
            description = "copy in the file's order"

            @staticmethod
            def slab(z0, z1):
                return nrrd.block(z0, z1)

        return Native

    def remember(self, zs):
        """Keep these output slices as they pass, for the check after writing (a compressed file
        cannot be read at random)."""
        self.wanted = set(zs)

    def slab(self, z0, z1):
        z_native, z_sign, _ = self.plan[0]
        if self.store is not None:
            index = [slice(None)] * 3
            n = self.store.shape[z_native]
            index[z_native] = slice(z0, z1) if z_sign > 0 else slice(n - z1, n - z0)
            block = oriented(self.store[tuple(index)], self.plan)
        elif z_sign > 0:
            block = oriented(self.nrrd.block(z0, z1), self.plan)
        else:
            n = self.nrrd.shape[0]
            block = oriented(self.nrrd.block(n - z1, n - z0), self.plan)
        for z in self.wanted.intersection(range(z0, z1)):
            self.kept[z] = np.array(block[z - z0])
        return block

    def reference(self, z):
        if z in self.kept:
            return self.kept[z]
        return self.slab(z, z + 1)[0]


#
# OME-Zarr store (rebuild its coarser levels)
#


class StoreSource:
    """Level 0 of an existing OME-Zarr store, with its axes, transforms and other attributes."""

    def __init__(self, path):
        group = zarr.open_group(path, mode="r")
        attrs = dict(group.attrs)
        multiscales = attrs["ome"]["multiscales"][0]
        dataset = multiscales["datasets"][0]
        self.array = group[dataset["path"]]
        names = [a["name"] for a in multiscales["axes"]]
        if names != ["z", "y", "x"]:
            raise ConversionError(f"level 0 axes are {names}; only z, y, x stores are handled")
        transforms = {t["type"]: t for t in dataset["coordinateTransformations"]}
        self.spacing = [float(v) for v in transforms["scale"]["scale"]]
        self.origin = [float(v) for v in transforms.get("translation", {"translation": [0.0] * 3})["translation"]]
        self.axes = multiscales["axes"]
        self.name = multiscales.get("name") or dataset["path"].split("/")[-1]
        self.shape, self.dtype = tuple(self.array.shape), self.array.dtype
        self.attributes = {k: v for k, v in attrs.items() if k != "ome"}
        self.description = f"level 0 of {os.path.basename(os.path.normpath(path))}, spacing {self.spacing} mm"

    def slab(self, z0, z1):
        return self.array[z0:z1]

    def reference(self, z):
        return self.array[z]


def open_source(path, voxel_um=None, threads=8):
    """The source for an input path: slice folder, OME-Zarr store, NRRD or TIFF file."""
    path = os.path.normpath(path)
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "zarr.json")):
            return StoreSource(path)
        return StackSource(path, voxel_um, threads)
    if not os.path.exists(path):
        raise ConversionError(f"{path} does not exist")
    lower = path.lower()
    if lower.endswith((".nrrd", ".nhdr")):
        return NrrdSource(path)
    if lower.endswith((".tif", ".tiff")):
        return TiffFileSource(path, voxel_um)
    raise ConversionError(f"{os.path.basename(path)}: not a slice folder, OME-Zarr store, NRRD or TIFF file")


#
# Writing
#


def check_output(out, source_path=None):
    """Refuse an output that is the input, or an existing folder that is not an OME-Zarr store
    (writing replaces the folder's contents)."""
    if source_path and os.path.exists(out) and os.path.samefile(source_path, out):
        raise ConversionError("write to a new store: the input is read while the output is written")
    if os.path.isfile(out):
        raise ConversionError(f"{out} is a file")
    if os.path.isdir(out) and os.listdir(out) and not os.path.exists(os.path.join(out, "zarr.json")):
        raise ConversionError(f"{out} exists and is not an OME-Zarr store: choose a new folder")


def convert(source, out, levels=None, shard=CHUNK, log=print, progress=None):
    """Write ``source`` to ``out`` with ``levels`` levels (automatic when None), chunks grouped in
    ``shard``^3 shards (``shard`` == CHUNK: no shards, one file per chunk). ``progress(done, total)``
    is called after each slab, in source slices."""
    if shard < CHUNK or shard % CHUNK:
        raise ValueError(f"shard size {shard} is not a multiple of the {CHUNK}-voxel chunk")
    check_output(out)
    levels = levels or auto_levels(source.shape)
    name, dtype = source.name, np.dtype(source.dtype)
    log(f"{name}: {source.shape[2]}x{source.shape[1]}x{source.shape[0]} {dtype}, {source.description}, {levels} levels")
    shapes = [tuple(n // 2**k for n in source.shape) for k in range(levels)]
    root = zarr.open_group(out, mode="w", zarr_format=3)
    arrays = []
    for k, shape in enumerate(shapes):
        group = root.create_group(f"scale{k}")
        group.attrs["_ARRAY_DIMENSIONS"] = ["z", "y", "x"]
        layout = {"chunks": (CHUNK,) * 3, "shards": (shard,) * 3} if shard > CHUNK else {"chunks": (CHUNK,) * 3}
        arrays.append(
            group.create_array(
                name, shape=shape, dtype=dtype, compressors=ZstdCodec(level=3), fill_value=0,
                dimension_names=("z", "y", "x"), config={"write_empty_chunks": False}, **layout,
            )
        )
    datasets = []
    for k in range(levels):
        factor = 2**k
        datasets.append({
            "path": f"scale{k}/{name}",
            "coordinateTransformations": [
                {"type": "scale", "scale": [s * factor for s in source.spacing]},
                # a level-k voxel is the mean of a factor^3 block: its centre sits (factor-1)/2 voxels in
                {"type": "translation", "translation": [o + s * (factor - 1) / 2.0 for o, s in zip(source.origin, source.spacing)]},
            ],
        })
    root.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": name,
            "axes": source.axes,
            "datasets": datasets,
            "type": "local_mean",
            "metadata": {
                "description": "Each level is the mean of the 2^k x 2^k x 2^k level-0 voxels it covers, rounded once (half to even)",
                "method": "to_omezarr.block_mean",
            },
        }],
    }
    for key, value in source.attributes.items():
        root.attrs[key] = value

    pending = [[] for _ in range(levels)]  # coarse slices waiting to fill a row of shards
    written = [0] * levels
    rows = slab_rows(shard, levels)

    def flush(k, final=False):
        data = np.concatenate(pending[k]) if pending[k] else None
        while data is not None and (len(data) >= rows or (final and len(data))):
            n = min(rows, len(data))
            arrays[k][written[k] : written[k] + n] = data[:n]
            written[k] += n
            data = data[n:] if len(data) > n else None
        pending[k] = [data] if data is not None else []

    started = time.time()
    depth = source.shape[0]
    for z0 in range(0, depth, rows):
        z1 = min(z0 + rows, depth)
        slab = np.asarray(source.slab(z0, z1))
        arrays[0][z0:z1] = slab
        written[0] += z1 - z0
        for k in range(1, levels):
            pending[k].append(block_mean(slab, 2**k, dtype))
            flush(k)
        del slab
        log(f"  slices {z0}-{z1 - 1} done, {time.time() - started:.0f} s")
        if progress:
            progress(z1, depth)
    for k in range(1, levels):
        flush(k, final=True)
    assert written == [s[0] for s in shapes], (written, shapes)
    zarr.consolidate_metadata(out)
    log(f"levels {shapes} written in {time.time() - started:.0f} s")
    return levels


def block_mean(slab, factor, dtype):
    """Mean of each factor^3 block of ``slab`` (z, y, x), trailing partial blocks dropped.
    Integers are summed exactly and rounded once, half to even."""
    z, y, x = (n // factor * factor for n in slab.shape)
    out = np.empty((z // factor, y // factor, x // factor), dtype)
    integer = np.issubdtype(dtype, np.integer)
    accumulator = np.int64 if np.issubdtype(dtype, np.signedinteger) else np.uint64 if integer else np.float64
    rows = max(factor, 32 // factor * factor)  # source slices per step: bounds the temporaries
    for z0 in range(0, z, rows):
        part = slab[z0 : min(z0 + rows, z), :y, :x]
        sums = part.reshape(part.shape[0] // factor, factor, y // factor, factor, x // factor, factor).sum(
            axis=(1, 3, 5), dtype=accumulator
        )
        mean = sums / float(factor**3)
        out[z0 // factor : z0 // factor + mean.shape[0]] = np.rint(mean) if integer else mean
    return out


def verify(out, source, levels, log=print, samples=SAMPLE_SLICES, seed=0):
    """Level 0 against the source; every coarse level against the exact block mean of level 0."""
    rng = np.random.default_rng(seed + 1)
    name, depth = source.name, source.shape[0]
    group = zarr.open_group(out, mode="r")
    level0 = group[f"scale0/{name}"]
    mismatched = 0
    zs = sample_slices(depth, samples, seed)
    for z in zs:
        mismatched += int((level0[z] != np.asarray(source.reference(z))).sum())
    log(f"level 0 vs source, {len(zs)} slices: {mismatched} voxels differ")
    ok = mismatched == 0
    for k in range(1, levels):
        factor, array = 2**k, group[f"scale{k}/{name}"]
        worst = 0.0
        for _ in range(3):
            n = min(8, *array.shape)
            cz, cy, cx = (int(rng.integers(0, s - n + 1)) for s in array.shape)
            block = level0[cz * factor : (cz + n) * factor, cy * factor : (cy + n) * factor, cx * factor : (cx + n) * factor]
            mean = block.astype(np.float64).reshape(n, factor, n, factor, n, factor).mean(axis=(1, 3, 5))
            stored = array[cz : cz + n, cy : cy + n, cx : cx + n].astype(np.float64)
            worst = max(worst, float(np.abs(stored - mean).max()))
        log(f"level {k} {array.shape}: max |stored - true {factor}^3 mean| = {worst:.3f} gray levels")
        ok &= worst <= 0.5 + 1e-9
    return ok


def run(input_path, out, levels=None, voxel_um=None, shard=CHUNK, threads=8, check=True, log=print, progress=None):
    """Convert ``input_path`` to ``out``: the whole job, including the temporary native copy an
    NRRD may need and the check against the source. Returns True when the check passes (or is skipped).
    ``progress(done, total, step)`` reports slices of the current step ("copy", "convert")."""
    out = os.path.normpath(out)
    check_output(out, input_path)
    source = open_source(input_path, voxel_um, threads)
    if check and hasattr(source, "remember"):
        source.remember(sample_slices(source.shape[0]))
    native = None
    try:
        if getattr(source, "needs_native_copy", False):
            native = out.rstrip(os.sep) + ".native-copy"
            if os.path.exists(native):
                shutil.rmtree(native)
            log(f"z of {source.nrrd.file} needs reordering: copying level 0 in the file's order to {native} first")
            convert(source.native(), native, levels=1, shard=CHUNK, log=log,
                    progress=progress and (lambda d, t: progress(d, t, "copy")))
            group = zarr.open_group(native, mode="r")
            source.store = group[f"scale0/{source.name}"]
        levels = convert(source, out, levels, shard, log, progress and (lambda d, t: progress(d, t, "convert")))
        return verify(out, source, levels, log) if check else True
    finally:
        if native and os.path.exists(native):
            shutil.rmtree(native)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", help="folder of 2D slices, multi-page TIFF, .nrrd/.nhdr, or an .ome.zarr store")
    parser.add_argument("out", help="output .ome.zarr directory (an existing store there is replaced)")
    parser.add_argument("--levels", type=int, help=f"resolution levels including full resolution "
                                                   f"(default: until the coarsest is at most {COARSEST_SIDE} voxels)")
    parser.add_argument("--voxel-size", type=float, help="voxel size in micrometres (slices and TIFF; overrides an NRecon log)")
    parser.add_argument("--threads", type=int, default=8, help="threads reading slices (default 8)")
    parser.add_argument("--shard", type=int, default=CHUNK,
                        help=f"shard size in voxels per side, a multiple of {CHUNK} (default {CHUNK}: no shards, one file per chunk)")
    parser.add_argument("--no-verify", action="store_true", help="skip the check against the source")
    parser.add_argument("--progress", action="store_true", help="print 'PROGRESS <step> <done> <total>' lines")
    args = parser.parse_args()
    if args.shard < CHUNK or args.shard % CHUNK:
        parser.error(f"--shard must be a multiple of {CHUNK}")
    if args.levels is not None and args.levels < 1:
        parser.error("--levels must be at least 1")

    def progress(done, total, step):
        print(f"PROGRESS {step} {done} {total}", flush=True)

    def log(*parts):
        print(*parts, flush=True)

    try:
        ok = run(args.input, args.out, args.levels, args.voxel_size, args.shard, args.threads,
                 check=not args.no_verify, log=log, progress=progress if args.progress else None)
    except ConversionError as e:
        print(f"ERROR {e}", flush=True)
        sys.exit(2)
    if not ok:
        print("ERROR the written store does not match the source (VERIFY FAILED)", flush=True)
        sys.exit(1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
