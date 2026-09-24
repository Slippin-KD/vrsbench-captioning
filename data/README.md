# Data layout

The dataset files are deliberately ignored by Git.

Download these files from `xiang709/VRSBench` into `data/downloads/`:

- `Images_train.zip`
- `Annotations_train.zip`
- `Images_val.zip`
- `VRSBench_EVAL_Cap.json`

The scripts read image files directly from the ZIP archives. Extract only
`Annotations_train.zip`; it is small and supplies the training captions.

Run the manifest scripts after downloading:

```bash
python scripts/prepare_train_manifest.py
python scripts/prepare_validation_manifest.py
```
