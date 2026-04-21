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

    def fit(self, dataset: Iterable[Tuple[AudioFeatures, Label]]) -> "NativeNonNativePipeline":
        buckets: Dict[Label, List[AudioFeatures]] = {"NATIVE": [], "NON_NATIVE": []}
        for features, label in dataset:
            buckets[label].append(features)

        if not buckets["NATIVE"] or not buckets["NON_NATIVE"]:
            raise ValueError("Training data must include both NATIVE and NON_NATIVE labels.")

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
        count = len(samples)
        return AudioFeatures(
            speaking_rate_wpm=sum(s.speaking_rate_wpm for s in samples) / count,
            pronunciation_score=sum(s.pronunciation_score for s in samples) / count,
            pause_ratio=sum(s.pause_ratio for s in samples) / count,
            filler_word_ratio=sum(s.filler_word_ratio for s in samples) / count,
        )

    @staticmethod
    def _distance(a: AudioFeatures, b: AudioFeatures) -> float:
        return sqrt(
            (a.speaking_rate_wpm - b.speaking_rate_wpm) ** 2
            + (a.pronunciation_score - b.pronunciation_score) ** 2
            + (a.pause_ratio - b.pause_ratio) ** 2
            + (a.filler_word_ratio - b.filler_word_ratio) ** 2
        )
