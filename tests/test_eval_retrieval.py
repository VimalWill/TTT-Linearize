import math
import unittest

import torch

from eval import ar_masks


class RetrievalBucketTests(unittest.TestCase):
    def test_chunk_start_uses_prediction_query_position(self):
        # The bigram ending at target 4 repeats the bigram ending at 1.
        # Its prediction query is position 3, so both occurrences are in
        # chunk 0.  Comparing target positions would incorrectly call this
        # cross-chunk.
        ids = torch.tensor([7, 8, 9, 7, 8])
        masks = ar_masks(ids, [4, math.inf], chunk=4)
        self.assertTrue(bool(masks['ar_same_chunk'][4]))
        self.assertFalse(bool(masks['ar_x0_4'][4]))

    def test_cross_chunk_repeat_stays_cross_chunk(self):
        ids = torch.tensor([7, 8, 9, 10, 11, 7, 8])
        masks = ar_masks(ids, [4, math.inf], chunk=4)
        self.assertTrue(bool(masks['ar_x4+'][6]))
        self.assertFalse(bool(masks['ar_same_chunk'][6]))


if __name__ == '__main__':
    unittest.main()
