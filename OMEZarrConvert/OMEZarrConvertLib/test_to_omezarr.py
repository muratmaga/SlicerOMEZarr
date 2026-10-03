"""Checks of the converter without Slicer: python -m pytest test_to_omezarr.py (or run this file)."""
import bz2
import gzip
import os
import tempfile

import numpy as np
import tifffile
import zarr

import to_omezarr as conv

QUIET = dict(log=lambda *a: None)


def write_nrrd(path, volume, directions, origin, space="left-posterior-superior", encoding="raw", detached=False,
               endian="little", byte_skip=None):
    """``volume`` in NRRD order (k, j, i); ``directions`` the i, j, k steps in ``space``."""
    kinds = {np.dtype("uint8"): "uchar", np.dtype("int16"): "short", np.dtype("uint16"): "ushort", np.dtype("float32"): "float"}
    data = volume.astype(volume.dtype.newbyteorder("<" if endian == "little" else ">")).tobytes()
    if encoding == "gzip":
        data = gzip.compress(data)
    elif encoding == "bzip2":
        data = bz2.compress(data)
    vectors = " ".join("(" + ",".join(repr(float(v)) for v in d) + ")" for d in directions)
    header = [
        "NRRD0004",
        f"type: {kinds[volume.dtype]}",
        "dimension: 3",
        f"space: {space}",
        f"sizes: {volume.shape[2]} {volume.shape[1]} {volume.shape[0]}",
        f"space directions: {vectors}",
        "kinds: domain domain domain",
        f"endian: {endian}",
        f"encoding: {encoding}",
        f"space origin: ({','.join(repr(float(v)) for v in origin)})",
        'space units: "mm" "mm" "mm"',
    ]
    if byte_skip is not None:
        header.append(f"byte skip: {byte_skip}")
    if detached:
        data_name = os.path.basename(path).replace(".nhdr", ".raw" + (".gz" if encoding == "gzip" else ""))
        header.append(f"data file: {data_name}")
        with open(os.path.join(os.path.dirname(path), data_name), "wb") as f:
            f.write(data)
        with open(path, "w") as f:
            f.write("\n".join(header) + "\n")
    else:
        with open(path, "wb") as f:
            f.write(("\n".join(header) + "\n\n").encode())
            f.write(data)


def lps_points(shape, origin, steps):
    """LPS position of every voxel of an array of ``shape`` (axes a0, a1, a2) given the step of each axis."""
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    return np.asarray(origin) + idx @ np.asarray(steps)


def assert_same_physical_voxels(out, name, volume, directions, origin, space_signs=(1, 1, 1)):
    """Every level-0 voxel of the store has the value of the source voxel at the same LPS point,
    the store's geometry read as the Slicer module reads it (direction from orientation, translation = origin)."""
    signs = np.asarray(space_signs, dtype=float)
    steps_native = [np.asarray(directions[2]) * signs, np.asarray(directions[1]) * signs, np.asarray(directions[0]) * signs]
    source = {tuple(np.round(p, 6)): v for p, v in zip(lps_points(volume.shape, np.asarray(origin) * signs, steps_native), volume.ravel())}
    group = zarr.open_group(out, mode="r")
    ms = group.attrs["ome"]["multiscales"][0]
    assert [a["orientation"]["value"] for a in ms["axes"]] == ["inferior-to-superior", "anterior-to-posterior", "right-to-left"]
    t = {x["type"]: x for x in ms["datasets"][0]["coordinateTransformations"]}
    sz, sy, sx = t["scale"]["scale"]
    tz, ty, tx = t["translation"]["translation"]
    stored = group[f"scale0/{name}"][:]
    steps_out = [np.array([0, 0, sz]), np.array([0, sy, 0]), np.array([sx, 0, 0])]
    points = lps_points(stored.shape, [tx, ty, tz], steps_out)
    for p, v in zip(points, stored.ravel()):
        assert source[tuple(np.round(p, 6))] == v, p


def check_levels(out, name, levels):
    group = zarr.open_group(out, mode="r")
    level0 = group[f"scale0/{name}"][:]
    for k in range(1, levels):
        f = 2**k
        z, y, x = (n // f for n in level0.shape)
        exact = level0[: z * f, : y * f, : x * f].astype(np.float64).reshape(z, f, y, f, x, f).mean(axis=(1, 3, 5))
        np.testing.assert_array_equal(group[f"scale{k}/{name}"][:], np.rint(exact).astype(level0.dtype))


def volume_of(dtype, shape, seed=0):
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, endpoint=True, dtype=dtype)


# Slicer saves a RAS-identity volume as LPS with x and y running right and anterior.
SLICER_DIRECTIONS = [(-0.05, 0, 0), (0, -0.05, 0), (0, 0, 0.05)]


def run_nrrd(tmp, volume, directions, origin, levels=3, shard=conv.CHUNK, space_signs=(1, 1, 1), **kwargs):
    path = os.path.join(tmp, "specimen.nhdr" if kwargs.get("detached") else "specimen.nrrd")
    write_nrrd(path, volume, directions, origin, **kwargs)
    out = os.path.join(tmp, "specimen.ome.zarr")
    assert conv.run(path, out, levels=levels, shard=shard, **QUIET)
    assert_same_physical_voxels(out, "specimen", volume, directions, origin, space_signs)
    check_levels(out, "specimen", levels)
    assert not os.path.exists(out + ".native-copy")
    return out


def test_nrrd_raw_slicer_orientation():
    with tempfile.TemporaryDirectory() as tmp:
        run_nrrd(tmp, volume_of(np.uint16, (40, 37, 30)), SLICER_DIRECTIONS, (3.0, -2.0, 7.5))


def test_nrrd_gzip_big_endian_signed():
    with tempfile.TemporaryDirectory() as tmp:
        run_nrrd(tmp, volume_of(np.int16, (33, 20, 26)), SLICER_DIRECTIONS, (0.0, 0.0, 0.0), encoding="gzip", endian="big")


def test_nrrd_ras_space_detached_gzip():
    with tempfile.TemporaryDirectory() as tmp:
        # RAS space, identity directions: x and y run toward right and anterior in LPS terms.
        run_nrrd(tmp, volume_of(np.uint8, (24, 30, 18)), [(0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)], (1.0, 2.0, 3.0),
                 space="right-anterior-superior", space_signs=(-1, -1, 1), encoding="gzip", detached=True)


def test_nrrd_z_inferior_raw_and_compressed():
    directions = [(0.05, 0, 0), (0, 0.05, 0), (0, 0, -0.05)]
    for encoding in ("raw", "bzip2"):  # raw reads slabs from the end; bzip2 goes through the native copy
        with tempfile.TemporaryDirectory() as tmp:
            run_nrrd(tmp, volume_of(np.uint8, (300, 20, 22)), directions, (0.0, 0.0, 10.0), encoding=encoding)


def test_nrrd_coronal_permuted_axes_and_shards():
    # The file's slowest axis runs posterior; its rows run superior: z comes from j, y from k.
    directions = [(-0.02, 0, 0), (0, 0, -0.02), (0, 0.02, 0)]
    with tempfile.TemporaryDirectory() as tmp:
        run_nrrd(tmp, volume_of(np.uint16, (21, 260, 19)), directions, (5.0, 1.0, 2.0), shard=256)


def test_nrrd_byte_skip_minus_one():
    with tempfile.TemporaryDirectory() as tmp:
        run_nrrd(tmp, volume_of(np.uint8, (10, 12, 14)), SLICER_DIRECTIONS, (0, 0, 0), byte_skip=-1, levels=2)


def test_nrrd_oblique_refused():
    c, s = np.cos(0.1), np.sin(0.1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "oblique.nrrd")
        write_nrrd(path, volume_of(np.uint8, (8, 8, 8)), [(c, s, 0), (-s, c, 0), (0, 0, 1)], (0, 0, 0))
        try:
            conv.run(path, os.path.join(tmp, "o.ome.zarr"), **QUIET)
        except conv.ConversionError as e:
            assert "oblique" in str(e)
        else:
            raise AssertionError("an oblique NRRD was accepted")


def test_output_folder_guard():
    with tempfile.TemporaryDirectory() as tmp:
        write_nrrd(os.path.join(tmp, "a.nrrd"), volume_of(np.uint8, (8, 8, 8)), SLICER_DIRECTIONS, (0, 0, 0))
        precious = os.path.join(tmp, "precious")
        os.mkdir(precious)
        open(os.path.join(precious, "notes.txt"), "w").close()
        try:
            conv.run(os.path.join(tmp, "a.nrrd"), precious, **QUIET)
        except conv.ConversionError:
            assert os.path.exists(os.path.join(precious, "notes.txt"))
        else:
            raise AssertionError("a folder that is not a store was overwritten")
        store = os.path.join(tmp, "a.ome.zarr")  # an existing store is replaced
        assert conv.run(os.path.join(tmp, "a.nrrd"), store, levels=1, **QUIET)
        assert conv.run(os.path.join(tmp, "a.nrrd"), store, levels=2, **QUIET)


def test_slice_folder_and_store_rebuild():
    volume = volume_of(np.uint8, (523, 37, 41))  # crosses 128-slice slabs; odd sizes drop trailing partial blocks
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "slices")
        os.mkdir(folder)
        for i, image in enumerate(volume):
            tifffile.imwrite(os.path.join(folder, f"s{i:04d}.tif"), image)
        out = os.path.join(tmp, "slices.ome.zarr")
        assert conv.run(folder, out, voxel_um=10.0, **QUIET)
        group = zarr.open_group(out, mode="r")
        np.testing.assert_array_equal(group["scale0/slices"][:], volume)
        levels = len(group.attrs["ome"]["multiscales"][0]["datasets"])
        assert levels == conv.auto_levels(volume.shape) == 2
        check_levels(out, "slices", levels)
        again = os.path.join(tmp, "again.ome.zarr")
        assert conv.run(out, again, levels=levels, shard=384, **QUIET)
        rebuilt = zarr.open_group(again, mode="r")
        assert rebuilt["scale0/slices"].shards == (384,) * 3
        np.testing.assert_array_equal(rebuilt["scale1/slices"][:], group["scale1/slices"][:])


def test_multipage_tiff():
    volume = volume_of(np.uint16, (20, 30, 25))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stack.tif")
        tifffile.imwrite(path, volume)
        out = os.path.join(tmp, "stack.ome.zarr")
        assert conv.run(path, out, levels=2, voxel_um=5.0, **QUIET)
        np.testing.assert_array_equal(zarr.open_group(out, mode="r")["scale0/stack"][:], volume)
        compressed = os.path.join(tmp, "packed.tif")  # page by page when the data is not contiguous
        tifffile.imwrite(compressed, volume, compression="zlib")
        assert conv.run(compressed, os.path.join(tmp, "packed.ome.zarr"), levels=2, voxel_um=5.0, **QUIET)


def test_auto_levels():
    assert conv.auto_levels((3882, 2596, 1908)) == 4
    assert conv.auto_levels((8000, 8000, 8000)) == 5
    assert conv.auto_levels((300, 200, 100)) == 1


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
    print("all passed")
