# `scene/colmap_loader.py`

## Purpose

Parses COLMAP's binary and text reconstruction formats — camera intrinsics,
image poses, and the sparse 3-D point cloud. 294 lines of format handling.

Used indirectly by our pipeline: every render loads camera poses through this file.
We never call it directly and never modified it.

## Structure

### Lines ~20–40 — Data structures

```python
CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id",
                                             "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error",
                                             "image_ids", "point2D_idxs"])
```

Mirrors COLMAP's own schema. `qvec` is a rotation quaternion, `tvec` a translation —
together the camera pose that eventually becomes `R` and `T` in a `Camera` object.

### Rotation conversion

```python
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        ...])
```

Quaternion to rotation matrix, in COLMAP's `(w, x, y, z)` convention. The same
mathematics as `build_rotation` in [general_utils.md](general_utils.md), written for
NumPy and a single quaternion rather than batched CUDA tensors.

This is the first step in the chain that produces the view matrix used by the
rasteriser.

### Binary readers

```python
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
```

The primitive used throughout: read a fixed number of bytes and unpack according to a
`struct` format string. `"<"` specifies little-endian, matching COLMAP's format.

```python
def read_extrinsics_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            ...
```

Reads `images.bin`: for each registered image, an ID, a quaternion, a translation, a
camera ID, a name, and its 2-D feature observations.

**This is where our 301 camera poses come from.** The "Reading camera 301/301" line in
every Colab log is this loop.

```python
def read_intrinsics_binary(path_to_model_file):
    ...
```

Reads `cameras.bin`: image dimensions and intrinsic parameters (focal lengths,
principal point) per camera model. These become the field-of-view values via
`focal2fov`.

```python
def read_points3D_binary(path_to_model_file):
    ...
```

Reads `points3D.bin`: the sparse point cloud with positions, colours and reprojection
errors. Used only to initialise training — irrelevant to us, since we load a trained
checkpoint.

This is nonetheless the source of the log line **"Converting point3d.bin to .ply, will
happen only the first time you open the scene"**, which appeared in our first Colab
session and not in subsequent ones.

### Text readers

`read_extrinsics_text`, `read_intrinsics_text`, `read_points3D_text` parse COLMAP's
plain-text equivalents. Our dataset ships binary, so these are unused.

## Why it matters

Three practical points for the dissertation.

**1. Camera poses are deterministic and shared.** Every render — baseline and all 12
compressed configurations — loads identical poses through this file. Camera geometry
is therefore not a variable in any comparison.

**2. It explains the per-render startup cost.** Parsing 301 camera records plus the
sparse point cloud, then loading and resizing 301 images, is the several-second
"Reading camera 301/301 / Loading Training Cameras / Loading Test Cameras" pause
visible before each render. Multiplied by 12 configurations, this accounts for a
noticeable share of the ~40-minute evaluation run — a useful detail when explaining
where the wall time went.

**3. It defines the dataset contract.** The scene directory must contain a `sparse/`
subdirectory in COLMAP format, which is what `scene/__init__.py` probes for when
choosing its loader. That requirement is why our `fix_cfg_args.py` must point
`source_path` at the unpacked `tandt/train` directory specifically.
