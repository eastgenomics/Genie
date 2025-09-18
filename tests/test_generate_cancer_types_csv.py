import pytest

import additional_scripts.generate_cancer_types_csv as gct


class TestMapVCFCancerTypeToGenieDisplayName:
    def test_map_vcf_cancer_type_to_genie_display_name(self):
        all_cancer_types = [
            "Breast Invasive Ductal Carcinoma",
            "Non-Small Cell Lung Cancer",
            "Acute Myeloid Leukemia",
        ]

        result = gct.map_vcf_cancer_type_to_genie_display_name(
            all_cancer_types
        )

        expected = {
            "BreastInvasiveDuctalCarcinoma": (
                "Breast Invasive Ductal Carcinoma"
            ),
            "NonSmallCellLungCancer": "Non-Small Cell Lung Cancer",
            "AcuteMyeloidLeukemia": "Acute Myeloid Leukemia",
        }
        assert result == expected, f"Expected {expected}, but got {result}"


class TestAddDisplayNamesToCancerInfoWithMapping:
    def test_add_display_names_to_cancer_info_with_mapping(self):
        cancer_info = [
            {
                "vcf_name": "NonSmallCellLungCancer",
                "total_patient_count": 42,
            }
        ]
        display_map = {"NonSmallCellLungCancer": "Non-Small Cell Lung Cancer"}

        result = gct.add_display_names_to_cancer_info(cancer_info, display_map)

        expected = [
            {
                "vcf_name": "NonSmallCellLungCancer",
                "total_patient_count": 42,
                "display_name": "Non-Small Cell Lung Cancer",
            }
        ]
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_add_display_names_to_cancer_info_without_mapping(self, capsys):
        cancer_info = [
            {
                "vcf_name": "All_Cancers",
                "total_patient_count": 10,
            }
        ]
        display_map = {"NonSmallCellLungCancer": "Non-Small Cell Lung Cancer"}

        result = gct.add_display_names_to_cancer_info(cancer_info, display_map)
        expected = [
            {
                "vcf_name": "All_Cancers",
                "total_patient_count": 10,
                "display_name": "All Cancers",
            }
        ]
        assert result == expected, f"Expected {expected}, but got {result}"

        captured = capsys.readouterr()
        assert (
            "Warning: No mapping found for 'All_Cancers', using fallback"
            " display name 'All Cancers'"
            in captured.out
        )


class TestAddWhetherEachCancerTypeIsPartOfGroupedCancerType:
    def test_add_whether_each_cancer_type_is_part_of_grouped_cancer_type(self):
        cancer_info = [
            {"display_name": "Non-Small Cell Lung Cancer"},
            {"display_name": "Acute Myeloid Leukemia"},
            {"display_name": "Unknown Cancer"},
        ]

        haemonc = ["Acute Myeloid Leukemia"]
        solid = ["Non-Small Cell Lung Cancer"]

        result = (
            gct.add_whether_each_cancer_type_is_part_of_grouped_cancer_type(
                cancer_info, haemonc, solid
            )
        )

        expected = [
            {
                "display_name": "Non-Small Cell Lung Cancer",
                "is_haemonc": 0,
                "is_solid": 1,
            },
            {
                "display_name": "Acute Myeloid Leukemia",
                "is_haemonc": 1,
                "is_solid": 0,
            },
            {"display_name": "Unknown Cancer", "is_haemonc": 0, "is_solid": 0},
        ]
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_add_whether_each_cancer_type_is_part_of_grouped_cancer_type_error_raised(
        self, capsys
    ):
        cancer_info = [
            {"display_name": "Non-Small Cell Lung Cancer"},
            {"display_name": "Acute Myeloid Leukemia"},
        ]

        haemonc = ["Acute Myeloid Leukemia"]
        solid = ["Non-Small Cell Lung Cancer", "Acute Myeloid Leukemia"]

        with pytest.raises(ValueError):
            gct.add_whether_each_cancer_type_is_part_of_grouped_cancer_type(
                cancer_info, haemonc, solid
            )

            captured = capsys.readouterr()
            assert (
                "Overlap detected between haemonc and solid cancer types:"
                in captured.out
            )
