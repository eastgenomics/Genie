import argparse
import csv
import os
import pysam
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.file_io import read_txt_file_to_list


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns
    -------
    args : Namespace
        Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser(
        description=(
            "Information required to generate cancer types CSV file which is"
            " required as input to the Genie NHS website"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help=(
            "Path to Genie VCF file containing INFO fields with cancer type"
            " information and total patient counts per cancer type"
        ),
    )

    parser.add_argument(
        "--all_cancer_types",
        required=True,
        type=str,
        help="Path to file which lists all cancer types in the Genie data",
    )

    parser.add_argument(
        "--haemonc_cancer_types",
        required=True,
        type=str,
        help="Path to file which lists haemonc cancer types in the Genie data",
    )

    parser.add_argument(
        "--solid_cancer_types",
        required=True,
        type=str,
        help="Path to file which lists solid cancer types in the Genie data",
    )

    parser.add_argument(
        "--output", required=True, type=str, help="Name of output CSV file"
    )

    return parser.parse_args()


def read_vcf_cancer_info_fields_to_dict(vcf_file: str) -> list:
    """
    Read the cancer types and counts from the INFO field names in the VCF file.

    Parameters
    ----------
    vcf_file : str
        Name of VCF file to read in

    Returns
    -------
    cancer_info : list[dict]
        List of dicts, each containing info on a cancer type (vcf_name is how
        the cancer is represented in the INFO field name and patient count is
        the N in the INFO field name)
    """
    vcf = pysam.VariantFile(vcf_file)
    cancer_info = []

    for header, _ in vcf.header.info.items():
        match = re.match(r"SameNucleotideChange_(.+?)_Count_N_(\d+)", header)
        if match:
            cancer_type = match.group(1)
            patients = int(match.group(2))
            cancer_info.append(
                {"vcf_name": cancer_type, "total_patient_count": patients}
            )

    return cancer_info


def map_vcf_cancer_type_to_genie_display_name(all_cancer_types: list) -> dict:
    """
    Create a dictionary mapping the cancer type names from the VCF to display
    name (name in original Genie data).

    Example:
    {
        "BreastInvasiveDuctalCarcinoma": "Breast Invasive Ductal Carcinoma",
        "NonSmallCellLungCancer": "Non-Small Cell Lung Cancer",
        ...
    }

    Parameters
    ----------
    all_cancer_types : list
        List of all cancer types in the original Genie data

    Returns
    -------
    dict
        Mapping of vcf_name -> display_name (Genie name)
    """
    vcf_to_display = {}

    for display_name in all_cancer_types:
        # Normalise spaces
        cleaned_display = " ".join(display_name.split())
        # Remove punctuation: commas, slashes, dashes
        vcf_name = re.sub(r"[-/,]", "", cleaned_display)
        # Remove spaces to generate vcf_name
        vcf_name = vcf_name.replace(" ", "")
        vcf_to_display[vcf_name] = display_name

    return vcf_to_display


def add_display_names_to_cancer_info(
    cancer_info: list, display_name_map: dict
) -> list:
    """
    Add the original Genie name (display name) for each cancer type extracted
    from the VCF to the cancer info list of dicts.

    Parameters
    ----------
    cancer_info : list[dict]
        List of dictionaries with cancer type info from the VCF
    display_name_map : dict
        Mapping of vcf_name -> display_name

    Returns
    -------
    list[dict]
        Updated list with display names added
    """
    for info_field in cancer_info:
        vcf_name = info_field["vcf_name"]

        # If we have a mapping of the VCF name to the original Genie cancer
        # type name, use it. Otherwise as a fallback (e.g. All_Cancers),
        # replace underscores with spaces
        if vcf_name in display_name_map:
            display_name = display_name_map[vcf_name]
        else:
            display_name = vcf_name.replace("_", " ")
            print(
                f"Warning: No mapping found for '{vcf_name}', "
                f"using fallback display name '{display_name}'"
            )

        info_field["display_name"] = display_name

    return cancer_info


def add_whether_each_cancer_type_is_part_of_grouped_cancer_type(
    cancer_info: list, haemonc_cancer_types: list, solid_cancer_types: list
) -> dict:
    """
    Add whether a specific cancer type is present in haemonc_cancer_types
    or solid_cancer types input lists.

    Parameters
    ----------
    cancer_info : list
        Dictionary of cancer types, patient counts and cancer name as it
        appears in the original Genie data
    haemonc_cancer_types : list
        List of haemonc cancer types to check for
    solid_cancer_types : list
        List of solid cancer types to check for

    Returns
    -------
    cancer_info : dict
        Updated dictionary for whether each cancer type is haemonc or solid

    Raises
    ------
    ValueError
        If there is an overlap between haemonc_cancer_types and solid_cancer_types
    """
    overlap = set(haemonc_cancer_types) & set(solid_cancer_types)
    if overlap:
        raise ValueError(
            "Overlap detected between haemonc and solid cancer types:"
            f" {sorted(overlap)}"
        )

    for info_field in cancer_info:
        if info_field["display_name"] in haemonc_cancer_types:
            info_field.update(
                {
                    "is_haemonc": 1,
                    "is_solid": 0,
                }
            )
        elif info_field["display_name"] in solid_cancer_types:
            info_field.update(
                {
                    "is_haemonc": 0,
                    "is_solid": 1,
                }
            )
        else:
            info_field.update(
                {
                    "is_haemonc": 0,
                    "is_solid": 0,
                }
            )

    return cancer_info


def write_cancer_info_to_csv(cancer_info: list, output_file: str):
    """
    Write cancer info list of dicts to CSV in fixed column order.

    Parameters
    ----------
    cancer_info : list[dict]
        List of cancer info dicts containing:
        display_name, vcf_name, is_haemonc, is_solid, total_patient_count
    output_file : str
        Path to output CSV file
    """
    fieldnames = [
        "display_name",
        "vcf_name",
        "is_haemonc",
        "is_solid",
        "total_patient_count",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC
        )
        writer.writeheader()
        for row in cancer_info:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    cancer_info = read_vcf_cancer_info_fields_to_dict(args.input)
    all_cancer_types = read_txt_file_to_list(args.all_cancer_types)
    cancer_name_map = map_vcf_cancer_type_to_genie_display_name(
        all_cancer_types
    )
    haemonc_cancer_types = read_txt_file_to_list(args.haemonc_cancer_types)
    solid_cancer_types = read_txt_file_to_list(args.solid_cancer_types)

    cancer_info_display = add_display_names_to_cancer_info(
        cancer_info, cancer_name_map
    )
    cancer_info_final = (
        add_whether_each_cancer_type_is_part_of_grouped_cancer_type(
            cancer_info_display, haemonc_cancer_types, solid_cancer_types
        )
    )
    write_cancer_info_to_csv(cancer_info_final, args.output)


if __name__ == "__main__":
    main()
