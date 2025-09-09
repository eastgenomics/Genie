import numpy as np
import os
import pandas as pd
import pytest
import sys
import unittest

from unittest.mock import patch
from pandas.testing import assert_frame_equal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modify_normalisation_dups
from utils.consequence_priorities import effect_priority, effect_map


class DummyRecord:
    def __init__(self, chrom, pos, ref, alts, info):
        self.chrom = chrom
        self.pos = pos
        self.ref = ref
        self.alts = alts
        self.info = info


class TestReadAnnotatedVcfToDf:
    """
    Test reading annotated VCF files into a DataFrame
    """

    @pytest.fixture
    def mock_vcf_records(self):
        return [
            DummyRecord(
                chrom="1",
                pos=12345,
                ref="A",
                alts=("T",),
                info={
                    "Genie_description": "1_12345_A_T",
                    "Transcript_ID": "ENST0001",
                    "CSQ_Consequence": ("missense_variant",),
                    "CSQ_Feature": ("ENST0001",),
                    "CSQ_HGVSc": ("ENST0001.1:c.123A>T",),
                    "CSQ_HGVSp": ("ENSP0001.1:p.Lys41Asn",),
                },
            ),
            DummyRecord(
                chrom="2",
                pos=54321,
                ref="G",
                alts=("C",),
                info={
                    "Genie_description": "3_54321_G_C",
                    "Transcript_ID": None,
                    "CSQ_Consequence": ".",
                    "CSQ_Feature": ".",
                    "CSQ_HGVSc": ".",
                    "CSQ_HGVSp": ".",
                },
            ),
            DummyRecord(
                chrom="3",
                pos=999,
                ref="G",
                alts=("A",),
                info={
                    "Genie_description": "3_999_G_A",
                    "Transcript_ID": "ENST0002",
                    "CSQ_Consequence": ("synonymous_variant",),
                    "CSQ_Feature": ("ENST0002",),
                    "CSQ_HGVSc": ("ENST0002.3:c.999G>A",),
                    "CSQ_HGVSp": ("ENSP0002.3:p.Val333%3DAsp",),
                },
            ),
        ]

    @patch("modify_normalisation_dups.pysam.VariantFile")
    def test_read_annotated_vcf_to_df(
        self, mock_variant_file, mock_vcf_records
    ):
        mock_variant_file.return_value.__iter__.return_value = mock_vcf_records
        df = modify_normalisation_dups.read_annotated_vcf_to_df("dummy.vcf")
        assert df.to_dict(orient="records") == [
            {
                "grch37_norm": "1_12345_A_T",
                "chrom_grch37": "1",
                "pos_grch37": 12345,
                "ref_grch37": "A",
                "alt_grch37": "T",
                "Genie_description": "1_12345_A_T",
                "Transcript_ID": "ENST0001",
                "VEP_Consequence": "missense_variant",
                "VEP_Feature": "ENST0001",
                "VEP_Feature_Version": "ENST0001.1",
                "VEP_HGVSc": "ENST0001.1:c.123A>T",
                "VEP_HGVSp": "ENSP0001.1:p.Lys41Asn",
                "VEP_p": "p.Lys41Asn",
            },
            {
                "grch37_norm": "2_54321_G_C",
                "chrom_grch37": "2",
                "pos_grch37": 54321,
                "ref_grch37": "G",
                "alt_grch37": "C",
                "Genie_description": "3_54321_G_C",
                "Transcript_ID": None,
                "VEP_Consequence": np.nan,
                "VEP_Feature": np.nan,
                "VEP_Feature_Version": np.nan,
                "VEP_HGVSc": np.nan,
                "VEP_HGVSp": np.nan,
                "VEP_p": np.nan,
            },
            {
                "grch37_norm": "3_999_G_A",
                "chrom_grch37": "3",
                "pos_grch37": 999,
                "ref_grch37": "G",
                "alt_grch37": "A",
                "Genie_description": "3_999_G_A",
                "Transcript_ID": "ENST0002",
                "VEP_Consequence": "synonymous_variant",
                "VEP_Feature": "ENST0002",
                "VEP_Feature_Version": "ENST0002.3",
                "VEP_HGVSc": "ENST0002.3:c.999G>A",
                "VEP_HGVSp": "ENSP0002.3:p.Val333%3DAsp",
                "VEP_p": "p.Val333=Asp",
            },
        ]


class TestGetNormalisationDuplicates(unittest.TestCase):
    """
    Test getting duplicate Genie_description values for the same variant
    caused by normalisation
    """

    def test_get_normalisation_duplicates_when_no_duplicates(self):
        df = pd.DataFrame(
            [
                {"grch37_norm": "1_100_A_T", "Genie_description": "1_100_A_T"},
                {"grch37_norm": "2_200_G_C", "Genie_description": "2_200_G_C"},
            ]
        )
        result = modify_normalisation_dups.get_normalisation_duplicates(
            df, "grch37_norm"
        )
        assert result.empty

    def test_get_normalisation_duplicates_when_single_duplicate(self):
        df = pd.DataFrame(
            [
                {"grch37_norm": "1_100_A_T", "Genie_description": "1_100_A_T"},
                {
                    "grch37_norm": "1_100_A_T",
                    "Genie_description": "1_100_A_T_alt",
                },
                {"grch37_norm": "2_200_G_C", "Genie_description": "2_200_G_C"},
            ]
        )
        result = modify_normalisation_dups.get_normalisation_duplicates(
            df, "grch37_norm"
        )
        # Should return only the conflicting rows
        expected_norms = ["1_100_A_T"]
        assert set(result["grch37_norm"]) == set(expected_norms)
        assert set(result["Genie_description"]) == {
            "1_100_A_T",
            "1_100_A_T_alt",
        }

    def test_get_normalisation_duplicates_when_multiple_duplicates(self):
        df = pd.DataFrame(
            [
                {"grch37_norm": "1_100_A_T", "Genie_description": "1_100_A_T"},
                {
                    "grch37_norm": "1_100_A_T",
                    "Genie_description": "1_100_A_T_alt",
                },
                {"grch37_norm": "2_200_G_C", "Genie_description": "2_200_G_C"},
                {
                    "grch37_norm": "2_200_G_C",
                    "Genie_description": "2_200_G_C_alt",
                },
                {"grch37_norm": "3_300_T_G", "Genie_description": "3_300_T_G"},
            ]
        )
        result = modify_normalisation_dups.get_normalisation_duplicates(
            df, "grch37_norm"
        )
        expected_norms = ["1_100_A_T", "2_200_G_C"]
        assert set(result["grch37_norm"]) == set(expected_norms)

    def test_get_normalisation_duplicates_when_identical_descriptions(
        self,
    ):
        df = pd.DataFrame(
            [
                {"grch37_norm": "1_100_A_T", "Genie_description": "1_100_A_T"},
                {"grch37_norm": "1_100_A_T", "Genie_description": "1_100_A_T"},
            ]
        )
        result = modify_normalisation_dups.get_normalisation_duplicates(
            df, "grch37_norm"
        )
        # No conflict since all descriptions are identical
        assert result.empty

    def test_get_normalisation_duplicates_when_empty_dataframe(self):
        df = pd.DataFrame(columns=["grch37_norm", "Genie_description"])
        result = modify_normalisation_dups.get_normalisation_duplicates(
            df, "grch37_norm"
        )
        assert result.empty


class TestConvertDuplicatesToOneRowPerGRCh38:
    """
    Test aggregating duplicates to get their Genie annotations
    """

    def test_convert_duplicates_to_one_variant_per_row_when_conflicting_annotations(
        self,
    ):
        df = pd.DataFrame(
            [
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": "1_99_A_T",
                    "VEP_Consequence": "missense_variant",
                },
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": "1_100_A_T",
                    "VEP_Consequence": "frameshift_variant",
                },
            ]
        )
        annotations_list = ["VEP_Consequence"]
        result = modify_normalisation_dups.convert_duplicates_to_one_variant_per_row(
            df, annotations_list, "grch37_norm", "grch38_description"
        )

        assert result.to_dict(orient="records") == [
            {
                "grch37_norm": "1_100_A_T",
                "grch38_description": "1_900_A_T",
                "Genie_description": ["1_99_A_T", "1_100_A_T"],
                "VEP_Consequence": ["missense_variant", "frameshift_variant"],
            },
        ]

    def test_convert_duplicates_to_one_variant_per_row_when_same_annotation(
        self,
    ):
        df = pd.DataFrame(
            [
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": "1_99_A_T",
                    "VEP_Consequence": "missense_variant",
                },
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": "1_100_A_T",
                    "VEP_Consequence": "missense_variant",
                },
            ]
        )

        annotations_list = ["VEP_Consequence"]
        result = modify_normalisation_dups.convert_duplicates_to_one_variant_per_row(
            df, annotations_list, "grch37_norm", "grch38_description"
        )

        assert result.to_dict(orient="records") == [
            {
                "grch37_norm": "1_100_A_T",
                "grch38_description": "1_900_A_T",
                "Genie_description": ["1_99_A_T", "1_100_A_T"],
                "VEP_Consequence": ["missense_variant"],
            },
        ]

    def test_convert_duplicates_to_one_variant_per_row_when_includes_nan(
        self,
    ):
        df = pd.DataFrame(
            [
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": "1_99_A_T",
                    "VEP_Consequence": "missense_variant",
                },
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": "1_100_A_T",
                    "VEP_Consequence": np.nan,
                },
            ]
        )

        annotations_list = ["VEP_Consequence"]
        result = modify_normalisation_dups.convert_duplicates_to_one_variant_per_row(
            df, annotations_list, "grch37_norm", "grch38_description"
        )

        assert result.to_dict(orient="records") == [
            {
                "grch37_norm": "1_100_A_T",
                "grch38_description": "1_900_A_T",
                "Genie_description": ["1_99_A_T", "1_100_A_T"],
                "VEP_Consequence": ["missense_variant", np.nan],
            },
        ]


class CheckAnnotationsForVariants(unittest.TestCase):
    """
    Test checking specific annotations in Genie data for normalisation
    duplicates
    """

    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        """
        Capture stdout to provide it to tests
        """
        self.capsys = capsys

    def test_check_annotations_for_variants_when_same(self):
        df = pd.DataFrame(
            [
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": ["1_99_A_T", "1_100_A_T"],
                    "Transcript_ID": ["ENSTXXX"],
                },
            ]
        )
        annotations_to_check = ["Transcript_ID"]
        result = modify_normalisation_dups.check_annotations_for_variants(
            df, annotations_to_check
        )

        stdout = self.capsys.readouterr().out
        assert "No rows with different Transcript_ID found." in stdout

    def test_check_annotations_for_variants_when_different(self):
        df = pd.DataFrame(
            [
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": ["1_99_A_T", "1_100_A_T"],
                    "Transcript_ID": ["ENSTXXX", "ENSTYYY"],
                    "Hugo_Symbol": ["GENE1", "GENE2"],
                }
            ]
        )
        annotations_to_check = ["Transcript_ID", "Hugo_Symbol"]
        result = modify_normalisation_dups.check_annotations_for_variants(
            df, annotations_to_check
        )

        expected_df_transcript = pd.DataFrame(
            [
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": ["1_99_A_T", "1_100_A_T"],
                    "Transcript_ID": ["ENSTXXX", "ENSTYYY"],
                    "Hugo_Symbol": ["GENE1", "GENE2"],
                    "same_Transcript_ID": False,
                },
            ]
        )

        expected_df_symbol = pd.DataFrame(
            [
                {
                    "grch37_norm": "1_100_A_T",
                    "grch38_description": "1_900_A_T",
                    "Genie_description": ["1_99_A_T", "1_100_A_T"],
                    "Transcript_ID": ["ENSTXXX", "ENSTYYY"],
                    "Hugo_Symbol": ["GENE1", "GENE2"],
                    "same_Hugo_Symbol": False,
                },
            ]
        )

        assert set(result.keys()) == {"Transcript_ID", "Hugo_Symbol"}

        assert_frame_equal(
            result["Transcript_ID"].reset_index(drop=True),
            expected_df_transcript.reset_index(drop=True),
        )
        assert_frame_equal(
            result["Hugo_Symbol"].reset_index(drop=True),
            expected_df_symbol.reset_index(drop=True),
        )


class TestUniqueWithNaN(unittest.TestCase):
    """
    Test getting unique values (including NaN) from a list of values
    """

    def test_unique_with_nan_when_nan_in_list(self):
        input_list = [1, 2, 2, 3, np.nan, 4, 4, np.nan]
        expected_output = [1, 2, 3, 4, np.nan]
        assert (
            modify_normalisation_dups.unique_with_nan(input_list)
            == expected_output
        )

    def test_unique_with_nan_when_no_nan_in_list(self):
        input_list = [1, 2, 2, 3, 4, 4]
        expected_output = [1, 2, 3, 4]
        assert (
            modify_normalisation_dups.unique_with_nan(input_list)
            == expected_output
        )

    def test_unique_with_nan_when_list_is_empty(self):
        input_list = []
        expected_output = []
        assert (
            modify_normalisation_dups.unique_with_nan(input_list)
            == expected_output
        )


class TestGetMostSevereConsequence(unittest.TestCase):
    """
    Test getting the most severe consequence from a string of consequences
    """

    def test_get_most_severe_consequence_when_multiple(self):
        input_data = (
            "protein_altering_variant$&synonymous_variant&splice_donor_variant"
        )
        expected_output = "splice_donor_variant"
        actual_output = modify_normalisation_dups.get_most_severe_consequence(
            input_data, effect_priority
        )
        self.assertEqual(actual_output, expected_output)

    def test_get_most_severe_consequence_when_single(self):
        input_data = "missense_variant"
        expected_output = "missense_variant"
        actual_output = modify_normalisation_dups.get_most_severe_consequence(
            input_data, effect_priority
        )
        self.assertEqual(actual_output, expected_output)

    def test_get_most_severe_consequence_when_empty(self):
        input_data = ""
        expected_output = "intergenic_variant"
        actual_output = modify_normalisation_dups.get_most_severe_consequence(
            input_data, effect_priority
        )
        self.assertEqual(actual_output, expected_output)


class TestSplitOutGRCh37ChromPosRefAlt:
    """
    Test splitting out a string into separate chrom, pos, ref and alt values
    """

    def test_split_out_grch37_chrom_pos_ref_alt(self):
        df = pd.DataFrame({"grch37_norm": ["1_12345_A_T"]})
        expected = pd.DataFrame(
            {
                "grch37_norm": ["1_12345_A_T"],
                "chrom": ["1"],
                "pos": ["12345"],
                "ref": ["A"],
                "alt": ["T"],
            }
        )
        result = modify_normalisation_dups.split_out_grch37_chrom_pos_ref_alt(
            df
        )
        pd.testing.assert_frame_equal(result, expected)


class TestClassifyVariantType(unittest.TestCase):
    """
    Test classifying a variant type by the length of ref and alt
    """

    def test_classify_variant_type_snv(self):
        ref = "A"
        alt = "G"
        expected_output = ("SNP", False)
        actual_output = modify_normalisation_dups.classify_variant_type(
            ref, alt
        )
        self.assertEqual(actual_output, expected_output)

    def test_classify_variant_type_insertion_not_in_frame(self):
        ref = "A"
        alt = "AGT"
        expected_output = ("INS", False)
        actual_output = modify_normalisation_dups.classify_variant_type(
            ref, alt
        )
        self.assertEqual(actual_output, expected_output)

    def test_classify_variant_type_insertion_in_frame(self):
        ref = "A"
        alt = "AGTA"
        expected_output = ("INS", True)
        actual_output = modify_normalisation_dups.classify_variant_type(
            ref, alt
        )
        self.assertEqual(actual_output, expected_output)

    def test_classify_variant_type_deletion_not_in_frame(self):
        ref = "AGT"
        alt = "A"
        expected_output = ("DEL", False)
        actual_output = modify_normalisation_dups.classify_variant_type(
            ref, alt
        )
        self.assertEqual(actual_output, expected_output)

    def test_classify_variant_type_deletion_in_frame(self):
        ref = "AGTG"
        alt = "A"
        expected_output = ("DEL", True)
        actual_output = modify_normalisation_dups.classify_variant_type(
            ref, alt
        )
        self.assertEqual(actual_output, expected_output)

    def test_classify_variant_type_dnp(self):
        ref = "AG"
        alt = "TC"
        expected_output = ("DNP", False)
        actual_output = modify_normalisation_dups.classify_variant_type(
            ref, alt
        )
        self.assertEqual(actual_output, expected_output)


class TestGetVariantClassification:
    @pytest.mark.parametrize(
        "effect,var_type,inframe,expected",
        [
            ("frameshift_variant", "DEL", False, "Frame_Shift_Del"),
            ("frameshift_variant", "INS", False, "Frame_Shift_Ins"),
            ("protein_altering_variant", "DEL", False, "Frame_Shift_Del"),
            ("protein_altering_variant", "INS", False, "Frame_Shift_Ins"),
            ("inframe_insertion", "INS", True, "In_Frame_Ins"),
            ("inframe_deletion", "DEL", True, "In_Frame_Del"),
            ("protein_altering_variant", "DEL", True, "In_Frame_Del"),
            ("", "", False, "Targeted_Region"),
            ("stop_gained", "INS", False, "Nonsense_Mutation"),
            ("start_lost", "SNP", False, "Translation_Start_Site"),
            ("missense_variant", "SNP", False, "Missense_Mutation"),
        ],
    )
    def test_get_variant_classification_multiple(
        self, effect, var_type, inframe, expected
    ):
        actual = modify_normalisation_dups.get_variant_classification(
            effect, effect_map, var_type, inframe
        )
        assert actual == expected
