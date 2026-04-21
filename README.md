# hackenza---AuraSense
NATIVE / NON NATIVE audio classification for hackenza 2026 by Renan.

## Classification pipeline

This repository now includes a small reusable pipeline in
`classification_pipeline.py` for NATIVE vs NON_NATIVE classification.

Example:

```python
from classification_pipeline import AudioFeatures, NativeNonNativePipeline

pipeline = NativeNonNativePipeline().fit(
    [
        (AudioFeatures(155, 0.93, 0.12, 0.02), "NATIVE"),
        (AudioFeatures(108, 0.72, 0.28, 0.09), "NON_NATIVE"),
    ]
)

label = pipeline.predict(AudioFeatures(150, 0.90, 0.14, 0.03))
```
