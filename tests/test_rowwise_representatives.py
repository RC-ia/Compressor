import unittest

import numpy as np

from compare_representatives_vs_q4 import (
    assign_rowwise_indices,
    fit_rowwise_codebooks,
    pack_nibbles,
    rowwise_representative_reconstruction,
    unpack_nibbles,
)


class RowwiseRepresentativeTests(unittest.TestCase):
    def test_rowwise_codebooks_pack_and_reconstruct(self) -> None:
        base = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
        matrix = np.stack(
            [
                base,
                base * 0.5 + 2.0,
                base * -1.5 - 0.25,
                np.square(base) * 3.0,
            ],
            axis=0,
        )

        codebooks = fit_rowwise_codebooks(matrix, group_count=4, chunk_rows=2, max_iter=12)
        codebooks_fp16 = codebooks.astype(np.float16)
        codebooks_stored = codebooks_fp16.astype(np.float32)
        indices = assign_rowwise_indices(matrix, codebooks_stored, chunk_rows=2)

        self.assertEqual(codebooks.shape, (4, 4))
        self.assertEqual(indices.shape, matrix.shape)
        self.assertTrue(np.all(np.isfinite(codebooks)))
        self.assertTrue(np.all(codebooks[:, 1:] >= codebooks[:, :-1]))

        packed = pack_nibbles(indices.reshape(-1))
        decoded_indices = unpack_nibbles(packed, matrix.size)
        np.testing.assert_array_equal(decoded_indices, indices.reshape(-1))

        reconstructed = rowwise_representative_reconstruction(
            packed, codebooks_stored, matrix.shape[1], 0, matrix.size
        )
        expected = codebooks_stored[
            np.repeat(np.arange(matrix.shape[0]), matrix.shape[1]), decoded_indices
        ]
        np.testing.assert_array_equal(reconstructed, expected)
        self.assertEqual(packed.nbytes, matrix.size // 2)
        self.assertEqual(codebooks_fp16.nbytes, matrix.shape[0] * 4 * 2)


if __name__ == "__main__":
    unittest.main()
