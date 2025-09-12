import polars as pl

from polars.testing import assert_frame_equal

from utils.merging import multi_merge


class TestMultiMerge:
    def test_multi_merge_single_merge(self):
        base = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "val": ["a", "b", "c"],
            }
        )

        extra = pl.DataFrame(
            {
                "id": [1, 2],
                "extra": [10, 20],
            }
        )

        result = multi_merge(base, [extra], on="id", how="left")

        expected = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "val": ["a", "b", "c"],
                "extra": [10, 20, None],
            }
        )

        assert_frame_equal(
            result,
            expected,
            check_column_order=False,
            check_row_order=False,
        )

    def test_multi_merge_multiple_merges(self):
        base = pl.DataFrame({"id": [1, 2]})
        df1 = pl.DataFrame({"id": [1, 2], "a": ["x", "y"]})
        df2 = pl.DataFrame({"id": [1, 2], "b": [100, 200]})

        result = multi_merge(base, [df1, df2], on="id", how="left")

        expected = pl.DataFrame(
            {
                "id": [1, 2],
                "a": ["x", "y"],
                "b": [100, 200],
            }
        )

        assert_frame_equal(
            result,
            expected,
            check_column_order=False,
            check_row_order=False,
        )
