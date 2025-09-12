import polars as pl

from polars.testing import assert_frame_equal

from utils.aggregation import (
    calculate_unique_patient_counts,
    create_df_with_one_row_per_variant,
    get_rows_for_cancer_types,
    get_truncating_variants,
    get_inframe_deletions,
)


class TestCalculateUniquePatientCounts:
    def test_calculate_unique_patient_counts_basic(self):
        df = pl.DataFrame(
            {
                "PATIENT_ID": [1, 2, 3, 1, 4],
                "CANCER_TYPE": ["A", "A", "B", "B", "C"],
            }
        )

        total, per_cancer, haemonc, solid = calculate_unique_patient_counts(df)

        assert total == 4
        assert per_cancer == {"A": 2, "B": 2, "C": 1}
        assert haemonc is None
        assert solid is None

    def test_calculate_unique_patient_counts_with_haemonc(self):
        df = pl.DataFrame(
            {
                "PATIENT_ID": [
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                    "patient_6",
                ],
                "CANCER_TYPE": ["AML", "CML", "Lung", "Breast", "AML", "CML"],
            }
        )
        haemonc_list = ["AML", "CML"]
        solid_list = ["Lung", "Breast"]

        total, per_cancer, haemonc, solid = calculate_unique_patient_counts(
            df, haemonc_list, solid_list
        )

        assert total == 6
        assert per_cancer == {"AML": 2, "CML": 2, "Lung": 1, "Breast": 1}
        assert haemonc == 4
        assert solid == 2

    def test_calculate_unique_patient_counts_empty_df(self):
        df = pl.DataFrame({"PATIENT_ID": [], "CANCER_TYPE": []})
        total, per_cancer, haemonc, solid = calculate_unique_patient_counts(df)
        assert total == 0
        assert per_cancer == {}
        assert haemonc is None
        assert solid is None


class TestCreateDfWithOneRowPerVariant:
    def test_create_df_with_one_row_per_variant_merges_columns(self):
        df = pl.DataFrame(
            {
                "grch38_description": ["var1", "var1", "var2"],
                "gene": ["TP53", "TP53", "BRCA1"],
                "effect": ["missense", "nonsense", "frameshift"],
            }
        )

        result = create_df_with_one_row_per_variant(df, ["gene", "effect"])

        expected = pl.DataFrame(
            {
                "grch38_description": ["var1", "var2"],
                "gene": ["TP53", "BRCA1"],
                "effect": ["missense&nonsense", "frameshift"],
            }
        )

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestGetTruncatingVariants:
    def test_get_truncating_variants_filters_correctly(self):
        df = pl.DataFrame(
            {
                "Variant_Classification": [
                    "Frame_Shift_Del",
                    "Frame_Shift_Ins",
                    "Nonsense_Mutation",
                    "Missense_Mutation",
                ],
                "HGVSp": [
                    "p.Trp23Ter",
                    "p.Arg50Ter",
                    "p.StopTer",
                    "p.Arg100Gly",
                ],
            }
        )

        result = get_truncating_variants(df)

        expected = pl.DataFrame(
            {
                "Variant_Classification": [
                    "Frame_Shift_Del",
                    "Frame_Shift_Ins",
                    "Nonsense_Mutation",
                ],
                "HGVSp": ["p.Trp23Ter", "p.Arg50Ter", "p.StopTer"],
            }
        )

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestGetInframeDeletions:
    def test_get_inframe_deletions_filters_correctly(self):
        df = pl.DataFrame(
            {
                "Variant_Classification": [
                    "In_Frame_Del",
                    "In_Frame_Del",
                    "Missense_Mutation",
                ],
                "HGVSc": ["c.123del", None, "c.456A>T"],
            }
        )
        result = get_inframe_deletions(df)

        expected = pl.DataFrame(
            {
                "Variant_Classification": ["In_Frame_Del"],
                "HGVSc": ["c.123del"],
            }
        )
        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestGetHaemoncCancerRows:
    def test_get_haemonc_cancer_rows_filters_correctly(self):
        df = pl.DataFrame(
            {
                "PATIENT_ID": [1, 2, 3, 4],
                "CANCER_TYPE": ["AML", "CML", "Breast", "Lung"],
            }
        )
        result = get_rows_for_cancer_types(df, ["AML", "CML"])

        expected = pl.DataFrame(
            {
                "PATIENT_ID": [1, 2],
                "CANCER_TYPE": ["AML", "CML"],
            }
        )
        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )
