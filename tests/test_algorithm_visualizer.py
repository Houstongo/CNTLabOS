import unittest

import numpy as np

from backend.core.algorithm_visualizer import AlgorithmVisualizer


class AlgorithmVisualizerTests(unittest.TestCase):
    def test_visualize_extraction_generates_steps_for_valid_grayscale_image(self):
        img = np.tile(np.arange(0, 64, dtype=np.uint8), (64, 1))

        visualizer = AlgorithmVisualizer(magnification=20000)

        steps = visualizer.visualize_extraction(img)

        self.assertGreater(len(steps), 0)


if __name__ == "__main__":
    unittest.main()
