# `scripts/figures/make_diagrams.py`

## Purpose

Draws the three method diagrams of Chapter 3 with matplotlib into
`results/figures/thesis/`:

- **Figure 3.1** — the end-to-end experimental pipeline (`fig3_1_pipeline_workflow`);
- **Figure 3.2** — the architecture of the compressed SH representation
  (`fig3_2_model_architecture`);
- **Figure 3.3** — the model process workflow for one scene (`fig3_3_process_workflow`).

The diagrams read no data. Every number on them (205.7 MB, 13.7 MB, 15.0×, 3.6×, 4.43×,
effective rank 12, 22 and 12 configurations) is typed into the labels, so they have to be
edited by hand if the results ever change.

## Line-by-line

### Lines 1–24 — Header, imports, paths, fonts

Paths resolve from the file's location; output goes to `results/figures/thesis/`. SVG
text is kept as text (`svg.fonttype: none`) so the diagrams stay editable.

### Lines 25–36 — Colour roles

```python
ROLE = {
    "thesis":   ("#2a78d6", "#eaf2fc"),   # new code (blue)
    "upstream": ("#7a7974", "#f2f1ee"),   # unchanged reference 3DGS (grey)
    "artifact": ("#eb6834", "#fdefe8"),   # files on disk (orange)
    "eval":     ("#1baf7a", "#e6f6ef"),   # evaluation (aqua)
    "decision": ("#6250d6", "#efedfb"),   # decisions (violet)
    ...
}
```

One colour per role, used consistently in all three diagrams. The thesis captions rely
on this ("grey boxes belong to the unchanged 3DGS pipeline, blue boxes are thesis
code, …").

### Lines 38–124 — Drawing helpers

- `new` — a figure whose axes are in inches, so boxes are placed in physical units;
- `box`, `diamond`, `group` — rounded process boxes (first line bold), decision diamonds,
  and dashed group outlines with a label;
- `arrow`, `poly_arrow` — straight or bent arrows with an optional italic label;
- `legend`, `hlegend` — vertical and horizontal role legends;
- `save` — PNG and SVG at 300 dpi.

### Lines 129–222 — Figure 3.2 (`model_architecture`)

Follows one checkpoint at tensor level: the 62 float32 fields (248 B per Gaussian), split
into the 48 SH floats (192 B, 205.7 MB, 77.4 %) and the 14 non-SH floats (56 B), which
bypass the compressor. The encoder group shows the canonical layout `S [N,3,16] → X
[N,48]`, the truncated SVD and the three ways of quantising U with their bytes per
Gaussian; the orange box is the `.npz` file whose size is the rate. The decoder group
dequantises, reconstructs and hands a standard PLY to the unchanged rasteriser. The
side panel gives the measured numbers for rank 16 with 8-bit codes.

### Lines 226–307 — Figure 3.1 (`pipeline_workflow`)

A grid of three phases (training, unchanged; baseline render and offline compression;
evaluation and reporting) against three rows (Colab T4 GPU, any CPU, files written).
Box numbers are the step numbers of thesis Table 3.7; solid arrows are control flow and
dotted arrows the files each step writes. It shows that only 12 of the 22 configurations
go back to the GPU, and that the rate is always the `.npz` size.

### Lines 311–365 — Figure 3.3 (`process_workflow`)

One scene through the pipeline as a flowchart with three decisions: is the input a valid
degree-3 model (otherwise rejected by `ply_io` / `sh_layout`), which quantisation mode,
and is the result within the tolerance of Section 3.4.2 (ΔPSNR against GT ≤ 1 dB and SSIM
against the baseline ≥ 0.99). The loop on the left sends a failed configuration back with
a larger r, more bits or a larger K.

### Lines 367–370 — Entry point

Draws the three diagrams.

## Running it

    python scripts/figures/make_diagrams.py

## Why it matters

The diagrams are how a reader first meets the method, so they must agree with the code.
Each box names the module or script that implements it (`sh_layout.py`,
`factorization.py`, `serialization.py`, `extraction.py`, `run_ablation.py`, …), which
lets a reader go straight from a figure to the source.
