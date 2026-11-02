# Data

Nothing in this directory is committed. Colab downloads into `/content/work/`:

| Archive | Contents | Size (bytes) | SHA-256 |
|---|---|---|---|
| [`GaussianViewTest.zip`](https://huggingface.co/camenduru/gaussian-splatting/resolve/main/GaussianViewTest.zip) | Pretrained T&T `train` model (iteration 30000) | 487,321,454 | `04543c347c164deaec3f0f4b585a7e9bb15b08a01607fdb4bfda88e8109d9238` |
| [`tandt_db.zip`](https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip) | Tanks and Temples + Deep Blending images and COLMAP poses | 682,628,995 | `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1` |

These are the values of thesis Appendix B.2; the notebook (cell 4) stops if a download
differs. To check by hand, run `sha256sum` on Linux or `shasum -a 256` on macOS.

For local (CPU) work, download the model zip and point scripts at
`GaussianViewTest/model/point_cloud/iteration_30000/point_cloud.ply`.
