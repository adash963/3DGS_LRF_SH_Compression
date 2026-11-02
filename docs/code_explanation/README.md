# Code Explanation

Line-by-line explanations of every Python file the thesis pipeline uses, written for a
reader who knows Python but not this codebase or 3D Gaussian Splatting.

## How these documents are organised

Each source file gets one markdown file. Within it:

- **Purpose** — what the file is for and who calls it
- **Line-by-line** — walkthrough in source order, with line numbers matching the
  file at the commit these were written against
- **Why it matters** — how the file supports the dissertation's claims (project
  code) or what role it plays in rendering (upstream code)

Line numbers drift if the source is edited. If a number looks wrong, trust the
quoted code, not the number.

## Two codebases live here

| Area | Origin | Lines | Meaning |
|---|---|---|---|
| `project/` | Written for this dissertation | 1,945 | The compression work being assessed |
| `upstream/` | graphdeco-inria 3DGS, unmodified | 3,760 | The baseline renderer this work sits on top of |

The distinction matters academically: the dissertation claims authorship of the
`project/` code only. The upstream code is the *unmodified* baseline — leaving it
untouched is what makes the comparison fair, since baseline and compressed models
render through exactly the same rasteriser.

## Project code (`project/`)

### Core package — `src/sh_compression/`

| Document | Source | Role |
|---|---|---|
| [sh_layout.md](project/sh_layout.md) | `sh_layout.py` | The canonical SH coefficient layout and its PLY mapping |
| [ply_io.md](project/ply_io.md) | `ply_io.py` | Reading/writing 3DGS point clouds without disturbing non-SH fields |
| [factorization.md](project/factorization.md) | `factorization.py` | Truncated SVD, rank analysis, synthetic test data |
| [quantization.md](project/quantization.md) | `quantization.py` | Scalar and K-means vector quantisation |
| [serialization.md](project/serialization.md) | `serialization.py` | The versioned `.npz` compressed asset format |
| [extraction.md](project/extraction.md) | `extraction.py` | High-level pipeline operations |
| [metrics.md](project/metrics.md) | `metrics.py` | PSNR, SSIM, LPIPS over render directories |
| [storage.md](project/storage.md) | `storage.py` | Byte accounting and compression ratios |
| [init.md](project/init.md) | `__init__.py` | Package version and format version constant |

### Command-line scripts — `scripts/`

| Document | Source | Role |
|---|---|---|
| [inspect_model.md](project/inspect_model.md) | `inspect_model.py` | Report a model's structure and SH share of bytes |
| [extract_sh.md](project/extract_sh.md) | `extract_sh.py` | Dump the SH matrix for offline analysis |
| [analyse_rank.md](project/analyse_rank.md) | `analyse_rank.py` | Singular spectrum, rank sweep, figures |
| [compress_model.md](project/compress_model.md) | `compress_model.py` | Build one compressed asset and report its bytes |
| [reconstruct_model.md](project/reconstruct_model.md) | `reconstruct_model.py` | Turn an asset back into a renderable PLY |
| [render_model.md](project/render_model.md) | `render_model.py` | Render any variant PLY via a staged model directory |
| [evaluate_model.md](project/evaluate_model.md) | `evaluate_model.py` | Compare two render directories |
| [run_ablation.md](project/run_ablation.md) | `run_ablation.py` | The storage/coefficient experiment matrix |
| [run_render_eval.md](project/run_render_eval.md) | `run_render_eval.py` | The full GPU evaluation driver |
| [generate_report_assets.md](project/generate_report_assets.md) | `generate_report_assets.py` | Results table and working plots, from result files only |
| [fix_cfg_args.md](project/fix_cfg_args.md) | `fix_cfg_args.py` | Repair the checkpoint's Windows-authored config |
| [export_crops.md](project/export_crops.md) | `export_crops.py` | Move image crops out of Colab without credentials |
| [compare_sh_baselines.md](project/compare_sh_baselines.md) | `compare_sh_baselines.py` | Same-checkpoint comparison with peer-style SH treatments |

### Figure scripts — `scripts/figures/`

| Document | Source | Role |
|---|---|---|
| [make_analysis_figures.md](project/make_analysis_figures.md) | `make_analysis_figures.py` | Thesis Figures 4.1–4.3 and 5.1 |
| [make_charts.md](project/make_charts.md) | `make_charts.py` | Thesis Figures 5.2 and 5.3 |
| [make_diagrams.md](project/make_diagrams.md) | `make_diagrams.py` | Thesis Figures 3.1–3.3 |

### Tests — `tests/`

| Document | Source | Role |
|---|---|---|
| [conftest.md](project/conftest.md) | `conftest.py` | Synthetic model fixtures |
| [test_sh_layout.md](project/test_sh_layout.md) | `test_sh_layout.py` | Proves no channel/coefficient permutation |
| [test_ply_roundtrip.md](project/test_ply_roundtrip.md) | `test_ply_roundtrip.py` | Proves round-trip losslessness |
| [test_factorization.md](project/test_factorization.md) | `test_factorization.py` | Proves the SVD mathematics |
| [test_quantization.md](project/test_quantization.md) | `test_quantization.py` | Proves quantiser correctness and determinism |
| [test_serialization.md](project/test_serialization.md) | `test_serialization.py` | Proves the asset format survives save/load |
| [test_metrics.md](project/test_metrics.md) | `test_metrics.py` | Proves the metrics behave sanely |
| [test_full_pipeline.md](project/test_full_pipeline.md) | `test_full_pipeline.py` | End-to-end integration |

## Upstream code (`upstream/`)

Documented because understanding the baseline is necessary to defend the work —
particularly `gaussian_model.py`, which defines the SH storage layout this entire
project depends on.

| Document | Source | Role |
|---|---|---|
| [gaussian_model.md](upstream/gaussian_model.md) | `scene/gaussian_model.py` | **The most important upstream file** — defines SH storage |
| [sh_utils.md](upstream/sh_utils.md) | `utils/sh_utils.py` | SH basis evaluation — what the coefficients *mean* |
| [gaussian_renderer.md](upstream/gaussian_renderer.md) | `gaussian_renderer/__init__.py` | The render call into the CUDA rasteriser |
| [render.md](upstream/render.md) | `render.py` | The rendering entry point used for all evaluation |
| [scene_init.md](upstream/scene_init.md) | `scene/__init__.py` | Scene loading, train/test split |
| [arguments.md](upstream/arguments.md) | `arguments/__init__.py` | Config system — source of the `cfg_args` bugs |
| [cameras.md](upstream/cameras.md) | `scene/cameras.py` | Camera model and view matrices |
| [dataset_readers.md](upstream/dataset_readers.md) | `scene/dataset_readers.py` | COLMAP/Blender ingestion |
| [colmap_loader.md](upstream/colmap_loader.md) | `scene/colmap_loader.py` | COLMAP binary format parsing |
| [train.md](upstream/train.md) | `train.py` | Optimisation loop (not used — we use a checkpoint) |
| [metrics_upstream.md](upstream/metrics_upstream.md) | `metrics.py` | Upstream's own evaluation script |
| [general_utils.md](upstream/general_utils.md) | `utils/general_utils.py` | Activations, learning-rate schedule, seeding |
| [graphics_utils.md](upstream/graphics_utils.md) | `utils/graphics_utils.py` | Projection matrices, point transforms |
| [camera_utils.md](upstream/camera_utils.md) | `utils/camera_utils.py` | Camera construction and resolution handling |
| [loss_utils.md](upstream/loss_utils.md) | `utils/loss_utils.py` | L1 and SSIM losses |
| [image_utils.md](upstream/image_utils.md) | `utils/image_utils.py` | MSE and PSNR |
| [system_utils.md](upstream/system_utils.md) | `utils/system_utils.py` | Directory and checkpoint-search helpers |

Not documented, because the thesis pipeline does not rely on them: `full_eval.py`
(paper-reproduction harness), `convert.py` (images to a COLMAP dataset),
`utils/read_write_model.py` (COLMAP's reference I/O), `utils/make_depth_scale.py`
(depth-regularisation scales), `gaussian_renderer/network_gui.py` (live-viewer socket)
and `lpipsPyTorch/` (the bundled LPIPS; the thesis uses the `lpips` package).

## Reading order for someone new to the project

1. [gaussian_model.md](upstream/gaussian_model.md) — where SH coefficients live
2. [sh_layout.md](project/sh_layout.md) — how this project addresses them
3. [factorization.md](project/factorization.md) — the compression mathematics
4. [quantization.md](project/quantization.md) — the second compression stage
5. [serialization.md](project/serialization.md) — what actually gets stored
6. [run_render_eval.md](project/run_render_eval.md) — how a result is produced
