import unittest

from classification_pipeline import AudioFeatures, NativeNonNativePipeline


class NativeNonNativePipelineTests(unittest.TestCase):
    def test_predicts_native_for_native_like_features(self):
        pipeline = NativeNonNativePipeline().fit(
            [
                (AudioFeatures(155, 0.93, 0.12, 0.02), "NATIVE"),
                (AudioFeatures(160, 0.91, 0.10, 0.03), "NATIVE"),
                (AudioFeatures(108, 0.72, 0.28, 0.09), "NON_NATIVE"),
                (AudioFeatures(102, 0.70, 0.31, 0.11), "NON_NATIVE"),
            ]
        )

        prediction = pipeline.predict(AudioFeatures(158, 0.92, 0.11, 0.02))
        self.assertEqual(prediction, "NATIVE")

    def test_raises_if_fit_missing_class(self):
        pipeline = NativeNonNativePipeline()
        with self.assertRaises(ValueError):
            pipeline.fit([(AudioFeatures(158, 0.92, 0.11, 0.02), "NATIVE")])

    def test_raises_when_predict_before_fit(self):
        pipeline = NativeNonNativePipeline()
        with self.assertRaises(RuntimeError):
            pipeline.predict(AudioFeatures(158, 0.92, 0.11, 0.02))

    def test_distance_scales_features_to_avoid_speaking_rate_domination(self):
        pipeline = NativeNonNativePipeline().fit(
            [
                (AudioFeatures(150, 0.95, 0.10, 0.02), "NATIVE"),
                (AudioFeatures(155, 0.94, 0.11, 0.02), "NATIVE"),
                (AudioFeatures(95, 0.65, 0.30, 0.10), "NON_NATIVE"),
                (AudioFeatures(100, 0.64, 0.32, 0.12), "NON_NATIVE"),
            ]
        )

        prediction = pipeline.predict(AudioFeatures(120, 0.93, 0.12, 0.03))
        self.assertEqual(prediction, "NATIVE")

    def test_mean_features_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            NativeNonNativePipeline._mean_features([])


if __name__ == "__main__":
    unittest.main()
