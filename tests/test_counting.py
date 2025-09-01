import numpy as np
import os
import pandas as pd
import sys

from pandas.testing import assert_frame_equal
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.counting


class TestCountSameNucleotideChangeAllCancers:
    def test_count_same_nucleotide_change_all_cancers_multiple_patients(self):
        df = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_100_A_T",
                    "2_200_G_C",
                    "2_200_G_C",
                    "3_300_T_G",
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                ],
            }
        )

        result = utils.counting.count_same_nucleotide_change(
            df, 100, "All_Cancers"
        )

        expected = pd.DataFrame(
            {
                "grch38_description": ["1_100_A_T", "2_200_G_C", "3_300_T_G"],
                "SameNucleotideChange.All_Cancers_Count_N_100": [1, 2, 1],
            }
        )

        assert_frame_equal(result, expected)

    def test_count_same_nucleotide_change_all_cancers_single_patient(self):
        df = pd.DataFrame(
            {
                "grch38_description": ["1_100_A_T", "1_100_A_T"],
                "PATIENT_ID": ["patient_1", "patient_1"],
            }
        )

        result = utils.counting.count_same_nucleotide_change(
            df, 100, "All_Cancers"
        )

        expected = pd.DataFrame(
            {
                "grch38_description": ["1_100_A_T"],
                "SameNucleotideChange.All_Cancers_Count_N_100": [1],
            }
        )

        assert_frame_equal(result, expected)


class CountSameNucleotideChangePerCancerType:
    def test_count_same_nucleotide_change_per_cancer_type(self):
        df = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_100_A_T",
                    "2_200_G_C",
                    "2_200_G_C",
                    "3_300_T_G",
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                ],
                "CANCER_TYPE": [
                    "T-Lymphoblastic Leukemia/Lymphoma",
                    "T-Lymphoblastic Leukemia/Lymphoma",
                    "Lymphatic Cancer",
                    "Blood Cancer",
                    "Mastocytosis",
                ],
            }
        )

        unique_patients_per_cancer = {
            "T-Lymphoblastic Leukemia/Lymphoma": 10,
            "Lymphatic Cancer": 2,
            "Blood Cancer": 1,
            "Mastocytosis": 1,
            "Another cancer": 12,
        }

        result = utils.counting.count_same_nucleotide_change_per_cancer_type(
            df, unique_patients_per_cancer
        )

        expected = pd.DataFrame(
            {
                "grch38_description": ["1_100_A_T", "2_200_G_C", "3_300_T_G"],
                "SameNucleotideChange.T-Lymphoblastic Leukemia/Lymphoma_Count_N_10": [
                    1,
                    0,
                    0,
                ],
                "SameNucleotideChange.Lymphatic Cancer_Count_N_2": [1, 0, 0],
                "SameNucleotideChange.Blood Cancer_Count_N_1": [1, 0, 0],
                "SameNucleotideChange.Mastocytosis_Count_N_1": [1, 0, 0],
                "SameNucleotideChange.Another cancer_Count_N_12": [0, 0, 0],
            }
        )

        assert_frame_equal(result, expected)


class TestCountNucleotideChangeHaemoncCancers:
    def test_count_same_nucleotide_change_haemonc_cancers(self):
        haemonc_rows = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_100_A_T",
                    "1_200_G_C",
                    "1_200_G_C",
                    "4_101_T_G",
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                ],
                "CANCER_TYPE": [
                    "Leukemia",
                    "Mature T and NK Neoplasms",
                    "Myelodysplastic Syndromes",
                    "Myelodysplastic Syndromes",
                    "Hodgkin Lymphoma",
                ],
            }
        )

        all_rows = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_100_A_T",
                    "1_200_G_C",
                    "1_200_G_C",
                    "4_101_T_G",
                    "20_1_T_A",
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                ],
                "CANCER_TYPE": [
                    "Leukemia",
                    "Mature T and NK Neoplasms",
                    "Myelodysplastic Syndromes",
                    "Myelodysplastic Syndromes",
                    "Hodgkin Lymphoma",
                    "Stomach Cancer",
                ],
            }
        )

        result = utils.counting.count_same_nucleotide_change(
            haemonc_rows, 1000, "Haemonc_Cancers", all_rows
        )

        expected = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_200_G_C",
                    "4_101_T_G",
                    "20_1_T_A",
                ],
                "SameNucleotideChange.Haemonc_Cancers_Count_N_1000": [
                    1,
                    2,
                    1,
                    0,
                ],
            }
        )

        assert_frame_equal(result, expected)


class TestCountAminoAcidChangeAllCancers:
    def test_count_amino_acid_change_all_cancers_when_amino_acid_same(self):
        df = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_100_A_T",
                    "1_200_G_C",
                    "1_200_G_C",
                    "4_101_T_G",
                    "10_100_A_C",
                    "10_101_G_T",
                    "11_500_A_C",
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_1",
                    "patient_7",
                    "patient_8",
                ],
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE3",
                    "GENE3",
                    "GENE4",
                ],
                "HGVSp": [
                    "p.Ala100Thr",
                    "p.Ala100Thr",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Arg400Gln",
                    "p.Arg400Gln",
                    np.nan,
                ],
            }
        )

        result = utils.counting.count_amino_acid_change(
            df, 16000, "All_Cancers"
        )

        expected = pd.DataFrame(
            {
                "Hugo_Symbol": ["GENE1", "GENE1", "GENE2", "GENE3"],
                "HGVSp": [
                    "p.Ala100Thr",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Arg400Gln",
                ],
                "SameAminoAcidChange.All_Cancers_Count_N_16000": [1, 2, 1, 2],
            }
        )

        assert_frame_equal(result, expected)


class TestCountAminoAcidChangePerCancerType:
    def test_count_amino_acid_change_per_cancer_type(self):
        df = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_100_A_T",
                    "1_200_G_C",
                    "1_200_G_C",
                    "4_101_T_G",
                    "10_100_A_C",
                    "10_101_G_T",
                    "11_500_A_C",
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_1",
                    "patient_7",
                    "patient_8",
                ],
                "CANCER_TYPE": [
                    "Cancer 1",
                    "Cancer 1",
                    "Cancer 2",
                    "Cancer 3",
                    "Cancer 4",
                    "Cancer 1",
                    "Cancer 2",
                    "Cancer 3",
                ],
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE3",
                    "GENE3",
                    "GENE4",
                ],
                "HGVSp": [
                    "p.Ala100Thr",
                    "p.Ala100Thr",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Arg400Gln",
                    "p.Arg400Gln",
                    np.nan,
                ],
            }
        )

        unique_patients_per_cancer = {
            "Cancer 1": 500,
            "Cancer 2": 4,
            "Cancer 3": 3,
            "Cancer 4": 50,
        }

        result = utils.counting.count_amino_acid_change_per_cancer_type(
            df, unique_patients_per_cancer
        )

        expected = pd.DataFrame(
            {
                "Hugo_Symbol": ["GENE1", "GENE1", "GENE2", "GENE3"],
                "HGVSp": [
                    "p.Ala100Thr",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Arg400Gln",
                ],
                "SameAminoAcidChange.Cancer 1_Count_N_500": [
                    1,
                    0,
                    0,
                    1,
                ],
                "SameAminoAcidChange.Cancer 2_Count_N_4": [0, 1, 0, 1],
                "SameAminoAcidChange.Cancer 3_Count_N_3": [0, 1, 0, 0],
                "SameAminoAcidChange.Cancer 4_Count_N_50": [0, 0, 1, 0],
            }
        )

        assert_frame_equal(result, expected)


class TestCountAminoAcidChangeHaemoncCancers:
    def test_count_amino_acid_change_haemonc_cancers(self):
        haemonc_rows = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_100_A_T",
                    "1_200_G_C",
                    "1_200_G_C",
                    "4_101_T_G",
                ],
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE2",
                    "GENE3",
                ],
                "HGVSp": [
                    "p.Ala100Thr",
                    "p.Ala100Thr",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                ],
                "CANCER_TYPE": [
                    "Leukemia",
                    "Mature T and NK Neoplasms",
                    "Myelodysplastic Syndromes",
                    "Myelodysplastic Syndromes",
                    "Hodgkin Lymphoma",
                ],
            }
        )

        all_variants = pd.DataFrame(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_100_A_T",
                    "1_200_G_C",
                    "1_200_G_C",
                    "4_101_T_G",
                    "20_500_A_C",
                ],
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE2",
                    "GENE3",
                    "GENE4",
                ],
                "HGVSp": [
                    "p.Ala100Thr",
                    "p.Ala100Thr",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Lys512His",
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                ],
                "CANCER_TYPE": [
                    "Leukemia",
                    "Mature T and NK Neoplasms",
                    "Myelodysplastic Syndromes",
                    "Myelodysplastic Syndromes",
                    "Hodgkin Lymphoma",
                    "Stomach Cancer",
                ],
            }
        )

        result = utils.counting.count_amino_acid_change(
            haemonc_rows, 1000, "Haemonc_Cancers", all_variants
        )

        expected = pd.DataFrame(
            {
                "Hugo_Symbol": ["GENE1", "GENE2", "GENE3", "GENE4"],
                "HGVSp": [
                    "p.Ala100Thr",
                    "p.Gly200Cys",
                    "p.Gly200Cys",
                    "p.Lys512His",
                ],
                "SameAminoAcidChange.Haemonc_Cancers_Count_N_1000": [
                    1,
                    2,
                    1,
                    0,
                ],
            }
        )

        assert_frame_equal(result, expected)


class TestExtractPositionFromCDS:
    @pytest.mark.parametrize(
        "hgvsc_value, expected",
        [
            (np.nan, None),
            ([], None),
            ("ENST00000269305.4:c.637C>T", 637),
            ("ENST00000278616.4:c.1027_1030del", 1027),
        ],
    )
    def test_extract_position_from_cds_return_none_when_input_not_str(
        self, hgvsc_value, expected
    ):
        result = utils.counting.extract_position_from_cds(hgvsc_value)
        assert result == expected


class TestCountPatientsWithVariantAtSamePositionOrDownstream:
    def test_count_patients_with_variant_at_same_position_or_downstream(self):
        df = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "CDS_position": [
                    480,
                    481,
                    481,
                    483,
                    484,
                    485,
                    511,
                    512,
                    513,
                    514,
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                    "patient_5",
                    "patient_6",
                    "patient_7",
                    "patient_8",
                ],
            }
        )

        result = utils.counting.count_patients_with_variant_at_same_position_or_downstream(
            df
        )

        expected = pd.DataFrame(
            {
                "CDS_position": [
                    480,
                    481,
                    483,
                    484,
                    485,
                    511,
                    512,
                    513,
                    514,
                ],
                "downstream_patient_count": [8, 8, 6, 5, 4, 4, 3, 2, 1],
            }
        )

        pd.testing.assert_frame_equal(result, expected)


class TestCountFrameshiftTruncatingAndNonsenseInCancers:
    def test_count_frameshift_truncating_and_nonsense(self):
        df = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "CDS_position": [
                    480,
                    481,
                    481,
                    483,
                    484,
                    485,
                    511,
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                    "patient_5",
                ],
            }
        )

        result = utils.counting.count_frameshift_truncating_and_nonsense(
            df, "All_Cancers", 500
        )

        expected = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "CDS_position": [480, 481, 483, 484, 485, 511],
                "SameOrDownstreamTruncatingVariantsPerCDS.All_Cancers_Count_N_500": [
                    5,
                    5,
                    3,
                    2,
                    1,
                    1,
                ],
            }
        )

        pd.testing.assert_frame_equal(result, expected)


class TestCountFrameshiftTruncatingAndNonsensePerCancerType:
    def test_count_frameshift_truncating_and_nonsense_per_cancer_type(self):
        df = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "CDS_position": [
                    480,
                    481,
                    481,
                    483,
                    484,
                    485,
                    511,
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                    "patient_5",
                ],
                "CANCER_TYPE": [
                    "Cancer 1",
                    "Cancer 2",
                    "Cancer 3",
                    "Cancer 4",
                    "Cancer 1",
                    "Cancer 4",
                    "Cancer 2",
                ],
            }
        )

        unique_patients_per_cancer = {
            "Cancer 1": 500,
            "Cancer 2": 4,
            "Cancer 3": 3,
            "Cancer 4": 50,
            "Cancer 5": 120,
        }

        result = utils.counting.count_frameshift_truncating_and_nonsense_per_cancer_type(
            df, unique_patients_per_cancer
        )

        expected = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "CDS_position": [480, 481, 483, 484, 485, 511],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 1_Count_N_500": [
                    2,
                    0,
                    0,
                    1,
                    0,
                    0,
                ],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 2_Count_N_4": [
                    0,
                    2,
                    0,
                    0,
                    0,
                    1,
                ],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 3_Count_N_3": [
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 4_Count_N_50": [
                    0,
                    0,
                    2,
                    0,
                    1,
                    0,
                ],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 5_Count_N_120": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            }
        )

        pd.testing.assert_frame_equal(result, expected)


class TestExtractHGVScDeletionPositions:
    @pytest.mark.parametrize(
        "hgvsc_value, expected",
        [
            ("ENST00000269305.4:c.480_485del", (480, 485)),
            ("ENST00000296930.5:c.511_524+1del", (511, 524)),
            ("ENST00000269305.4:c.480del", (480, 480)),
        ],
    )
    def test_extract_hgvsc_deletion_positions(self, hgvsc_value, expected):
        result = utils.counting.extract_hgvsc_deletion_positions(hgvsc_value)
        assert result == expected


class TestCountPatientsWithNestedDeletions:
    def test_count_patients_with_nested_deletions(self):
        df = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "del_start": [
                    480,
                    481,
                    481,
                    483,
                    484,
                    485,
                ],
                "del_end": [
                    485,
                    482,
                    482,
                    484,
                    485,
                    485,
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                ],
            }
        )

        result = utils.counting.count_patients_with_nested_deletions(df)

        expected = pd.DataFrame(
            {
                "del_start": [480, 481, 483, 484, 485],
                "del_end": [485, 482, 484, 485, 485],
                "nested_patient_count": [5, 2, 1, 2, 1],
            }
        )

        pd.testing.assert_frame_equal(result, expected)


class TestCountNestedInframeDeletionsAllCancers:
    def test_count_nested_inframe_deletions_all_cancers(self):
        df = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE3",
                    "GENE3",
                ],
                "del_start": [
                    480,
                    481,
                    481,
                    483,
                    484,
                    485,
                ],
                "del_end": [
                    485,
                    482,
                    482,
                    484,
                    485,
                    485,
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                ],
            }
        )

        result = utils.counting.count_nested_inframe_deletions(
            df, "All_Cancers", 5000
        )

        expected = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE3",
                    "GENE3",
                ],
                "del_start": [480, 481, 483, 484, 485],
                "del_end": [485, 482, 484, 485, 485],
                "NestedInframeDeletionsPerCDS.All_Cancers_Count_N_5000": [
                    2,
                    2,
                    1,
                    2,
                    1,
                ],
            }
        )

        pd.testing.assert_frame_equal(result, expected)


class TestCountNestedInframeDeletionsPerCancerType:
    def test_count_nested_inframe_deletions_per_cancer_type(self):
        df = pd.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "del_start": [
                    480,
                    481,
                    481,
                    483,
                    484,
                    485,
                ],
                "del_end": [
                    485,
                    482,
                    482,
                    484,
                    485,
                    485,
                ],
                "PATIENT_ID": [
                    "patient_1",
                    "patient_1",
                    "patient_2",
                    "patient_3",
                    "patient_4",
                    "patient_5",
                ],
                "CANCER_TYPE": [
                    "Cancer 1",
                    "Cancer 1",
                    "Cancer 2",
                    "Cancer 3",
                    "Cancer 1",
                    "Cancer 4",
                ],
            }
        )
        per_patient_cancer_total = {
            "Cancer 1": 500,
            "Cancer 2": 4,
            "Cancer 3": 3,
            "Cancer 4": 50,
            "Cancer 5": 120,
        }
        result = utils.counting.count_nested_inframe_deletions_per_cancer_type(
            df, per_patient_cancer_total
        )

        expected = pd.DataFrame(
            {
                "Hugo_Symbol": ["GENE1", "GENE1", "GENE1", "GENE1", "GENE1"],
                "del_start": [480, 481, 483, 484, 485],
                "del_end": [485, 482, 484, 485, 485],
                "NestedInframeDeletionsPerCDS.Cancer 1_Count_N_500": [
                    2,
                    1,
                    0,
                    1,
                    0,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 2_Count_N_4": [
                    0,
                    1,
                    0,
                    0,
                    0,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 3_Count_N_3": [
                    0,
                    0,
                    1,
                    0,
                    0,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 4_Count_N_50": [
                    0,
                    0,
                    0,
                    0,
                    1,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 5_Count_N_120": [
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            }
        )

        pd.testing.assert_frame_equal(result, expected)
