# CNN training data generation

`experimental/cnn_training` is a simulator-agnostic data-generation package
for CNN and vision-policy experiments. It turns numeric voxel worlds into
deterministic first-person image frames with aligned geometric supervision and
token features.

The current package provides scene generation, rendering, token derivation,
and sample persistence. It does not contain a CNN architecture or optimizer,
capture frames from Hytale, load an asset catalog, or connect to the native
server bridge.

## What it provides

- `worldgen.terrain` creates numeric voxel worlds from caller-supplied opaque
  class IDs. Random worlds support `patchwork`, `clutter`, `chaotic`, and
  `mixed` modes. Structured worlds support `layered`, `terraced`, `rifted`, and
  `frontier` terrain with separate rock, soil, surface, trunk, and foliage
  label groups.
- `worldgen.viewpoints.camera` creates deterministic first-person camera poses.
  `worldgen.viewpoints.plan` plans chunk-aligned capture grids with a halo and
  a safe camera path.
- `worldgen.viewpoints.raycast` renders a voxel field with PyTorch. A frame
  includes RGBA pixels plus class IDs, hit distances, face IDs, voxel
  coordinates, surface UVs, albedo, light, and fog channels.
- `worldgen.viewpoints.tokens` derives dense per-pixel features and sparse
  visible-voxel features from a rendered frame.
- `pipeline.py` combines world generation, rendering, and tokenization into
  `CnnSample` objects. `generate_batch` keeps samples independently
  seed-addressable for worker sharding.
- `dataset.py` saves and loads the frame, supervision, token, camera, and
  world-coordinate arrays as pickle-free compressed NPZ files.

The `worldgen.render_distance` modules are compatibility imports for older
callers. The shared implementations live under `worldgen.viewpoints`.

## Minimal example

Run this from the repository root:

```python
import numpy as np

from experimental.cnn_training import generate_sample, write_sample_npz

class_indices = np.arange(32, dtype=np.int16)
face_colors = np.zeros((32, 6, 3), dtype=np.uint8)
for label in range(len(class_indices)):
    face_colors[label] = (label * 17 % 255, label * 29 % 255, label * 43 % 255)

sample = generate_sample(
    class_indices,
    face_colors,
    seed=7,
    render_distance_chunks=1,
    height_blocks=48,
    side=64,
    device="cpu",  # use "cuda" for an available CUDA device
)

print(sample.frame.pixels_rgba_hwc.shape)
print(sample.tokens.dense_features_hwc.shape)
print(sample.tokens.visible_features_nf.shape)
write_sample_npz(sample, "output/sample.npz")
```

Install the runtime dependencies in the environment used for the experiment:

```text
python -m pip install numpy torch
```

The package is currently used directly from the repository root; it is not
included in the root distribution's published package list.

## Data contract

Class IDs are opaque integers. The caller supplies `face_colors_rgb` with
shape `[class_count, 6, 3]` and `uint8` values; optional textures use
`[class_count, 6, texture_size, texture_size, 4]`. The renderer does not
interpret those IDs as Hytale block or asset IDs.

`RaycastFrame` keeps image-space and world-space supervision aligned. Missed
pixels use the explicit empty-surface values, while `BlockTokenBatch` exposes
named dense and sparse feature channels suitable for a downstream CNN,
router, or teacher-label pipeline.

## Current limitations

- This is a data-generation layer, not a complete vision-training pipeline.
- There is no bundled CNN, training loop, dataset split/manifest, or evaluation
  harness in this directory.
- Terrain is generated from numeric rules. It is not a reconstruction of
  Hytale terrain, materials, lighting, animation, or native framebuffer output.
- Appearance is caller-provided flat color or texture data. Asset and semantic
  mapping must be implemented by an adapter outside this package.
- Rendering uses PyTorch and returns NumPy arrays. Selecting CUDA accelerates
  ray marching, but this package does not provide an end-to-end device-resident
  training path.
- The privileged class, distance, face, UV, voxel, albedo, light, and fog
  channels are supervision outputs; they are not a substitute for testing a
  policy from RGB alone.

The intended next layer is an experiment-specific dataset/training adapter
that chooses labels, appearance, augmentations, splits, and a CNN architecture
without adding simulator-specific dependencies to this package.
