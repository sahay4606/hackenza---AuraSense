from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Literal, Tuple

Label = Literal["NATIVE", "NON_NATIVE"]


@dataclass(frozen=True)
class AudioFeatures:
    speaking_rate_wpm: float
    pronunciation_score: float
    pause_ratio: float
    filler_word_ratio: float


class NativeNonNativePipeline:
    """
    A minimal classification pipeline for NATIVE/NON_NATIVE prediction.
    It learns per-label feature centroids and predicts by nearest centroid.
    """

    def __init__(self) -> None:
        self._centroids: Dict[Label, AudioFeatures] = {}
        self._feature_scales = AudioFeatures(1.0, 1.0, 1.0, 1.0)

    def fit(self, dataset: Iterable[Tuple[AudioFeatures, Label]]) -> "NativeNonNativePipeline":
        buckets: Dict[Label, List[AudioFeatures]] = {"NATIVE": [], "NON_NATIVE": []}
        for features, label in dataset:
            buckets[label].append(features)

        if not buckets["NATIVE"] or not buckets["NON_NATIVE"]:
            raise ValueError("Training data must include both NATIVE and NON_NATIVE labels.")

        all_samples = buckets["NATIVE"] + buckets["NON_NATIVE"]
        self._feature_scales = self._compute_scales(all_samples)
        self._centroids = {
            label: self._mean_features(samples)
            for label, samples in buckets.items()
        }
        return self

    def predict(self, features: AudioFeatures) -> Label:
        if not self._centroids:
            raise RuntimeError("Pipeline is not fitted. Call fit() before predict().")

        native_distance = self._distance(features, self._centroids["NATIVE"])
        non_native_distance = self._distance(features, self._centroids["NON_NATIVE"])
        return "NATIVE" if native_distance <= non_native_distance else "NON_NATIVE"

    def predict_batch(self, samples: Iterable[AudioFeatures]) -> List[Label]:
        return [self.predict(sample) for sample in samples]

    def explain(self, features: AudioFeatures) -> Dict[str, float]:
        if not self._centroids:
            raise RuntimeError("Pipeline is not fitted. Call fit() before explain().")
        return {
            "distance_to_native": self._distance(features, self._centroids["NATIVE"]),
            "distance_to_non_native": self._distance(features, self._centroids["NON_NATIVE"]),
        }

    @staticmethod
    def _mean_features(samples: List[AudioFeatures]) -> AudioFeatures:
        if not samples:
            raise ValueError("Cannot calculate mean features from an empty sample set.")
        count = len(samples)
        return AudioFeatures(
            speaking_rate_wpm=sum(s.speaking_rate_wpm for s in samples) / count,
            pronunciation_score=sum(s.pronunciation_score for s in samples) / count,
            pause_ratio=sum(s.pause_ratio for s in samples) / count,
            filler_word_ratio=sum(s.filler_word_ratio for s in samples) / count,
        )

    @staticmethod
    def _compute_scales(samples: List[AudioFeatures]) -> AudioFeatures:
        def scale(values: List[float]) -> float:
            value_range = max(values) - min(values)
            return value_range if value_range > 0 else 1.0

        return AudioFeatures(
            speaking_rate_wpm=scale([s.speaking_rate_wpm for s in samples]),
            pronunciation_score=scale([s.pronunciation_score for s in samples]),
            pause_ratio=scale([s.pause_ratio for s in samples]),
            filler_word_ratio=scale([s.filler_word_ratio for s in samples]),
        )

    def _distance(self, a: AudioFeatures, b: AudioFeatures) -> float:
        return sqrt(
            ((a.speaking_rate_wpm - b.speaking_rate_wpm) / self._feature_scales.speaking_rate_wpm) ** 2
            + ((a.pronunciation_score - b.pronunciation_score) / self._feature_scales.pronunciation_score) ** 2
            + ((a.pause_ratio - b.pause_ratio) / self._feature_scales.pause_ratio) ** 2
            + ((a.filler_word_ratio - b.filler_word_ratio) / self._feature_scales.filler_word_ratio) ** 2
        )
