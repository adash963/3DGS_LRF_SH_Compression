# Data

Nothing in this directory is committed. Colab downloads into `/content/work/`:

- `GaussianViewTest.zip` — pretrained T&T `train` model (iteration 30000)
  https://huggingface.co/camenduru/gaussian-splatting/resolve/main/GaussianViewTest.zip
- `tandt_db.zip` — Tanks&Temples + Deep Blending COLMAP datasets (650 MB, official mirror)
  https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip

For local (CPU) work, download the model zip and point scripts at
`GaussianViewTest/model/point_cloud/iteration_30000/point_cloud.ply`.
