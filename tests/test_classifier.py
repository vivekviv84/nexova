import unittest
import numpy as np
import sys
import os

# Add project root to path so we can import nexova_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexova_core.models import GridSenseClassifier

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = GridSenseClassifier()
        self.classifier.train_on_real_data(max_per_class=10)

    def test_normal_waveform(self):
        # Create a pure sine wave (normal)
        t = np.linspace(0, 0.2, 1000)
        samples = np.sin(2 * np.pi * 50 * t)
        
        cls, conf, _, method = self.classifier.predict(samples)
        
        # In the demo engine, pure sine is classified as normal
        self.assertEqual(cls, "normal")
        self.assertGreaterEqual(conf, 0.8)

    def test_empty_data(self):
        # Test with empty samples - should handle gracefully
        samples = np.array([])
        cls, conf, _, _ = self.classifier.predict(samples)
        self.assertEqual(cls, "normal")
        self.assertEqual(conf, 0.5)

if __name__ == "__main__":
    unittest.main()
