import os
import pandas as pd
import polars as pl
import sys

from pandas.testing import assert_frame_equal as assert_frame_equal_pd
from polars.testing import assert_frame_equal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.counting


class TestCountSameNucleotideChangeAllCancers:
    def test_count_same_nucleotide_change_all_cancers_multiple_patients(self):
        df = pl.from_dict(
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

        expected = pl.from_dict(
            {
                "grch38_description": ["1_100_A_T", "2_200_G_C", "3_300_T_G"],
                "SameNucleotideChange.All_Cancers_Count_N_100": [1, 2, 1],
            }
        )

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )

    def test_count_same_nucleotide_change_all_cancers_single_patient(self):
        df = pl.from_dict(
            {
                "grch38_description": ["1_100_A_T", "1_100_A_T"],
                "PATIENT_ID": ["patient_1", "patient_1"],
            }
        )

        result = utils.counting.count_same_nucleotide_change(
            df, 100, "All_Cancers"
        )

        expected = pl.from_dict(
            {
                "grch38_description": ["1_100_A_T"],
                "SameNucleotideChange.All_Cancers_Count_N_100": [1],
            }
        )

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestCountSameNucleotideChangePerCancerType:
    def test_count_same_nucleotide_change_per_cancer_type(self):
        df = pl.from_dict(
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
        }

        result = utils.counting.count_same_nucleotide_change_per_cancer_type(
            df, unique_patients_per_cancer
        )

        expected = pl.from_dict(
            {
                "grch38_description": ["1_100_A_T", "2_200_G_C", "3_300_T_G"],
                "SameNucleotideChange.T-Lymphoblastic Leukemia/Lymphoma_Count_N_10": [
                    1,
                    0,
                    0,
                ],
                "SameNucleotideChange.Lymphatic Cancer_Count_N_2": [0, 1, 0],
                "SameNucleotideChange.Blood Cancer_Count_N_1": [0, 1, 0],
                "SameNucleotideChange.Mastocytosis_Count_N_1": [0, 0, 1],
            }
        )

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestCountNucleotideChangeHaemoncCancers:
    def test_count_same_nucleotide_change_haemonc_cancers(self):
        haemonc_rows = pl.from_dict(
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

        all_rows = pl.from_dict(
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

        expected = pl.from_dict(
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

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestCountAminoAcidChangeAllCancers:
    def test_count_amino_acid_change_all_cancers_when_amino_acid_same(self):
        df = pl.from_dict(
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
                    None,
                ],
            }
        )

        result = utils.counting.count_amino_acid_change(
            df, 16000, "All_Cancers"
        )

        expected = pl.from_dict(
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

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestCountAminoAcidChangePerCancerType:
    def test_count_amino_acid_change_per_cancer_type(self):
        df = pl.from_dict(
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
                    None,
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

        expected = pl.from_dict(
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

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestCountAminoAcidChangeHaemoncCancers:
    def test_count_amino_acid_change_haemonc_cancers(self):
        haemonc_rows = pl.from_dict(
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

        all_variants = pl.from_dict(
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

        expected = pl.from_dict(
            {
                "grch38_description": [
                    "1_100_A_T",
                    "1_200_G_C",
                    "4_101_T_G",
                    "20_500_A_C",
                ],
                "SameAminoAcidChange.Haemonc_Cancers_Count_N_1000": [
                    1,
                    2,
                    1,
                    0,
                ],
            }
        )

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestExtractPositionFromHGVSc:
    def test_extract_position_from_hgvsc(self):
        df = pl.from_dict(
            {
                "HGVSc": [
                    "ENST00000269305.4:c.637C>T",
                    "ENST00000278616.4:c.1027_1030del",
                    "ENST00000269305.4:c.637del",
                ]
            }
        )

        result = utils.counting.extract_position_from_hgvsc(df)

        expected = pl.from_dict(
            {
                "HGVSc": [
                    "ENST00000269305.4:c.637C>T",
                    "ENST00000278616.4:c.1027_1030del",
                    "ENST00000269305.4:c.637del",
                ],
                "CDS_position": [637, 1027, 637],
            }
        )

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestCountFrameshiftTruncatingAndNonsenseInCancers:
    def test_count_frameshift_truncating_and_nonsense(self):
        df = pl.from_dict(
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
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript2",
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
                    "patient_1",
                ],
            }
        )

        result = utils.counting.count_frameshift_truncating_and_nonsense(
            df, "All_Cancers", 500
        )

        expected = pl.from_dict(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript2",
                ],
                "CDS_position": [480, 481, 483, 484, 485, 511],
                "SameOrDownstreamTruncatingVariantsPerCDS.All_Cancers_Count_N_500": [
                    4,
                    4,
                    2,
                    1,
                    2,
                    1,
                ],
            }
        )

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestCountFrameshiftTruncatingAndNonsensePerCancerType:
    def test_count_frameshift_truncating_and_nonsense_per_cancer(self):
        df = pl.DataFrame(
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
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript2",
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

        expected = pl.DataFrame(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript2",
                ],
                "CDS_position": [480, 481, 483, 484, 485, 511],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 1_Count_N_500": [
                    2,
                    1,
                    1,
                    1,
                    0,
                    0,
                ],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 2_Count_N_4": [
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                ],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 3_Count_N_3": [
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                "SameOrDownstreamTruncatingVariantsPerCDS.Cancer 4_Count_N_50": [
                    1,
                    1,
                    1,
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

        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestAddDeletionPositions:
    def test_add_deletion_positions_hgvsc(self):
        df = pl.from_dict(
            {
                "HGVSc": [
                    "ENST00000269305.4:c.480_485del",
                    "ENST00000296930.5:c.511_524+1del",
                    "ENST00000269305.4:c.480del",
                ]
            }
        )
        result = utils.counting.add_deletion_positions(df, source="HGVSc")
        expected = pl.from_dict(
            {
                "HGVSc": [
                    "ENST00000269305.4:c.480_485del",
                    "ENST00000296930.5:c.511_524+1del",
                    "ENST00000269305.4:c.480del",
                ],
                "del_start": [480, 511, 480],
                "del_end": [485, 524, 480],
            }
        )
        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )

    def test_add_deletion_positions_hgvsp(self):
        df = pl.from_dict(
            {
                "HGVSp": [
                    "p.Asp359_Thr364delinsGlyArgAla",
                    "p.Gln367_Gln379del",
                    "p.Leu370del",
                ]
            }
        )
        result = utils.counting.add_deletion_positions(df, source="HGVSp")
        expected = pl.from_dict(
            {
                "HGVSp": [
                    "p.Asp359_Thr364delinsGlyArgAla",
                    "p.Gln367_Gln379del",
                    "p.Leu370del",
                ],
                "del_start": [359, 367, 370],
                "del_end": [364, 379, 370],
            }
        )
        assert_frame_equal(
            result, expected, check_column_order=False, check_row_order=False
        )


class TestCountNestedInframeDeletionsAllCancers:
    def test_count_nested_inframe_deletions_all_cancers_one_transcript_per_gene(
        self,
    ):
        df = pl.from_dict(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE3",
                    "GENE3",
                ],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript3",
                    "Transcript3",
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
            df, "All_Cancers", 5000, position_method="CDS"
        )

        expected = pl.from_dict(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE3",
                    "GENE3",
                ],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript3",
                    "Transcript3",
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

        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_count_nested_inframe_deletions_all_cancers_multiple_transcripts_per_gene(
        self,
    ):
        df = pl.from_dict(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE3",
                    "GENE3",
                ],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript3",
                    "Transcript4",
                    "Transcript4",
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
            df, "All_Cancers", 5000, position_method="CDS"
        )

        expected = pl.from_dict(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE2",
                    "GENE3",
                    "GENE3",
                ],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript3",
                    "Transcript4",
                    "Transcript4",
                ],
                "del_start": [480, 481, 481, 483, 484, 485],
                "del_end": [485, 482, 482, 484, 485, 485],
                "NestedInframeDeletionsPerCDS.All_Cancers_Count_N_5000": [
                    1,
                    1,
                    1,
                    1,
                    2,
                    1,
                ],
            }
        )

        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )


class TestCountNestedInframeDeletionsPerCancerType:
    def test_count_nested_inframe_deletions_per_cancer_type_one_transcript_per_gene(
        self,
    ):
        df = pl.from_dict(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
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
            df, per_patient_cancer_total, position_method="CDS"
        )

        expected = pl.from_dict(
            {
                "Hugo_Symbol": ["GENE1", "GENE1", "GENE1", "GENE1", "GENE1"],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                ],
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
                    1,
                    1,
                    0,
                    0,
                    0,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 3_Count_N_3": [
                    1,
                    0,
                    1,
                    0,
                    0,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 4_Count_N_50": [
                    1,
                    0,
                    0,
                    1,
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

        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )

    def test_count_nested_inframe_deletions_per_cancer_type_multiple_transcripts_per_gene(
        self,
    ):
        df = pl.from_dict(
            {
                "Hugo_Symbol": [
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                    "GENE1",
                ],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript2",
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
            df, per_patient_cancer_total, position_method="CDS"
        )

        expected = pl.from_dict(
            {
                "Hugo_Symbol": ["GENE1", "GENE1", "GENE1", "GENE1", "GENE1"],
                "Transcript_ID": [
                    "Transcript1",
                    "Transcript1",
                    "Transcript1",
                    "Transcript2",
                    "Transcript2",
                ],
                "del_start": [480, 481, 483, 484, 485],
                "del_end": [485, 482, 484, 485, 485],
                "NestedInframeDeletionsPerCDS.Cancer 1_Count_N_500": [
                    1,
                    1,
                    0,
                    1,
                    0,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 2_Count_N_4": [
                    1,
                    1,
                    0,
                    0,
                    0,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 3_Count_N_3": [
                    1,
                    0,
                    1,
                    0,
                    0,
                ],
                "NestedInframeDeletionsPerCDS.Cancer 4_Count_N_50": [
                    0,
                    0,
                    0,
                    1,
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

        assert_frame_equal(
            result, expected, check_row_order=False, check_column_order=False
        )
