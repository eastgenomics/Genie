import argparse
import numpy as np
import pandas as pd
import pysam

from typing import Optional

from utils.consequence_priorities import effect_priority, effect_map
from utils.dtypes import column_dtypes
from utils.file_io import read_in_to_df


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
            "Information required to fix annotations for duplicates in Genie"
            " data"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help=(
            "Path to MAF file which includes patient and sample information "
            "and GRCh38 liftover"
        ),
    )

    parser.add_argument(
        "--vep_vcf",
        required=True,
        type=str,
        help=(
            "Path to VCF of unique duplicates which have been annotated"
            " with VEP"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Name of output file with normalisation duplicates modified",
    )

    return parser.parse_args()


def read_annotated_vcf_to_df(vcf_file: str) -> pd.DataFrame:
    """
    Read in VCF of normalisation duplicates annotated by VEP to a dataframe.

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file to read in.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the relevant variants and INFO fields from the
        VCF file.
    """
    with pysam.VariantFile(vcf_file, "r") as vcf_in:
        # Parse VCF records and generate a dataframe
        records = []

        for record in vcf_in:
            row = {
                "CHROM": str(record.chrom),
                "POS": int(record.pos),
                "REF": str(record.ref),
                "ALT": ",".join(str(a) for a in record.alts),
                "Genie_description": record.info.get(
                    "Genie_description", None
                ),
                "Transcript_ID": record.info.get("Transcript_ID", None),
                "VEP_Consequence": record.info.get("CSQ_Consequence", None),
                "VEP_Feature": record.info.get("CSQ_Feature", None),
                "VEP_HGVSc": record.info.get("CSQ_HGVSc", None),
                "VEP_HGVSp": record.info.get("CSQ_HGVSp", None),
            }

            records.append(row)

    vcf_df = pd.DataFrame(records)

    vcf_df = vcf_df.rename(
        columns={
            "CHROM": "chrom_grch37",
            "POS": "pos_grch37",
            "REF": "ref_grch37",
            "ALT": "alt_grch37",
        }
    )

    # Create a unique GRCh37 identifier for the GRCh37 normalised variant
    vcf_df["grch37_norm"] = (
        vcf_df["chrom_grch37"].astype(str)
        + "_"
        + vcf_df["pos_grch37"].astype(str)
        + "_"
        + vcf_df["ref_grch37"].astype(str)
        + "_"
        + vcf_df["alt_grch37"].astype(str)
    )

    # Pysam gives tuples for the VEP annotations for some reason so just
    # get the value
    for col in [
        "VEP_Consequence",
        "VEP_Feature",
        "VEP_HGVSc",
        "VEP_HGVSp",
    ]:
        vcf_df[col] = vcf_df[col].apply(
            lambda x: x[0] if isinstance(x, tuple) and len(x) == 1 else x
        )

    # Extract the version of the Ensembl transcript from the HGVSc
    vcf_df["VEP_Feature_Version"] = (
        vcf_df["VEP_HGVSc"].astype(str).str.split(":").str[0]
    )

    # Replace '.' with NaNs to match the rest of the variants
    cols = [
        "VEP_Consequence",
        "VEP_Feature",
        "VEP_Feature_Version",
        "VEP_HGVSc",
        "VEP_HGVSp",
    ]
    vcf_df[cols] = vcf_df[cols].replace(".", np.nan)

    # Split out just the p. from the HGVSp and replace URL encoding for '='
    vcf_df["VEP_p"] = (
        vcf_df["VEP_HGVSp"]
        .astype(str)
        .str.split(":", n=1)
        .str[1]
        .str.replace("%3D", "=", regex=False)
    )

    # Reorder columns
    vcf_df = vcf_df[
        [
            "grch37_norm",
            "chrom_grch37",
            "pos_grch37",
            "ref_grch37",
            "alt_grch37",
            "Genie_description",
            "Transcript_ID",
            "VEP_Consequence",
            "VEP_Feature",
            "VEP_Feature_Version",
            "VEP_HGVSc",
            "VEP_HGVSp",
            "VEP_p",
        ]
    ]

    return vcf_df


def get_normalisation_duplicates(
    genie_data: pd.DataFrame, variant_key: str
) -> pd.DataFrame:
    """
    Find any rows where they have the same grch38_description but
    the original Genie_description values were different.

    Parameters
    ----------
    genie_data : pd.DataFrame
        DataFrame containing Genie variant data and liftover
    variant_key : str
        The name of the column to use as the variant key

    Returns
    -------
    pd.DataFrame
        DataFrame containing rows with conflicting normalisation
    """
    conflicting_norms = (
        genie_data.groupby(variant_key)["Genie_description"]
        .nunique()
        .reset_index()
        .query("Genie_description > 1")[variant_key]
    )

    conflicting_rows = (
        genie_data[genie_data[variant_key].isin(conflicting_norms)]
        .drop_duplicates()
        .sort_values(by=[variant_key, "Genie_description"])
    )

    return conflicting_rows


def unique_with_nan(list_of_values: list) -> list:
    """
    Get unique values from a list, preserving NaNs.

    Parameters
    ----------
    list_of_values : list
        Input list with potential NaN values.

    Returns
    -------
    list
        List of unique values, with NaN included if present.
    """
    arr = pd.Series(list_of_values, dtype=object)

    # uniques without NaN (order-preserving)
    uniques = pd.unique(arr[~arr.isna()]).tolist()

    # Append NaN if present
    if arr.isna().any():
        uniques.append(np.nan)

    return uniques


def convert_duplicates_to_one_variant_per_row(
    duplicates: pd.DataFrame,
    annotations_list: list,
    variant_key_for_grouping: str,
    variant_key_to_take_first: str,
) -> pd.DataFrame:
    """
    Convert the duplicates DataFrame to one row per grch38_description

    Parameters
    ----------
    duplicates : pd.DataFrame
        DataFrame containing duplicate rows
    annotations_list : list
        list of annotations to aggregate
    variant_key_for_grouping: str
        the column name to use as the unique variant descriptor
    variant_key_to_take_first : str
        The variant column name to take the first value from

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per grch38_description
    """
    agg_dict = {
        variant_key_to_take_first: "first",
        "Genie_description": lambda x: unique_with_nan(x),
    }
    for col in annotations_list:
        agg_dict[col] = lambda x: unique_with_nan(x)

    aggregated_df = (
        duplicates.groupby(variant_key_for_grouping)
        .agg(agg_dict)
        .reset_index()
    )

    return aggregated_df


def check_annotations_for_variants(
    df: pd.DataFrame, annotations_to_check: list
) -> dict:
    """
    Check if specified annotations are consistent for duplicates and return rows with differences.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with one row per grch38_description.
    annotations_to_check : list
        List of annotations to check for consistency.

    Returns
    -------
    dict
        Dictionary where keys are annotation names and values are DataFrames
        containing rows with differing annotations.
    """
    differing_rows = {}

    for annotation in annotations_to_check:
        df_temp = df.copy()
        # Create a boolean column indicating whether all values are the same
        same_col = f"same_{annotation}"
        df_temp[same_col] = df_temp[annotation].apply(
            lambda x: len(set(x)) == 1 if isinstance(x, list) else False
        )

        # Filter rows where annotation is not consistent
        diff_rows = df_temp.loc[~df_temp[same_col]]
        differing_rows[annotation] = diff_rows

        # Print summary
        if diff_rows.empty:
            print(f"No rows with different {annotation} found.")
        else:
            print(
                f"Found {len(diff_rows)} rows with different {annotation} for"
                " the same Genie variant."
            )

    return differing_rows


def first_or_nan(x):
    """
    Return the first element of a list or NaN.

    Parameters
    ----------
    x : list
        Input list to extract the first element from.

    Returns
    -------
    str or int
        The first element of the list or NaN.
    """
    return x[0] if isinstance(x, list) and x else np.nan


def merge_vep_annotations_with_duplicates(
    duplicates_one_row_per_variant: pd.DataFrame, vep_annotations: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge variant rows for duplicates with annotations from VEP.

    Parameters
    ----------
    duplicates_one_row_per_variant : pd.DataFrame
        DataFrame containing variant rows for duplicates
    vep_annotations : pd.DataFrame
        DataFrame containing VEP annotations for those variants

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing duplicates with their VEP annotations
    """
    merged = pd.merge(
        duplicates_one_row_per_variant,
        vep_annotations,
        on="grch37_norm",
        how="left",
    )

    # For any variant(s) not annotated at all by VEP, take the
    # annotations from the first instance of the variant
    mask = merged["VEP_Feature"].isna()
    if mask.any():
        print(
            "The following variant(s) were not annotated by VEP; the"
            " annotations for the first instance of the variant in the Genie"
            " data will be used:"
        )
        print(merged.loc[mask, "grch37_norm"].tolist())

    fallback_map = {
        "Consequence": "VEP_Consequence",
        "HGVSc": "VEP_HGVSc",
        "HGVSp": "VEP_p",
    }

    for source_column, destination_column in fallback_map.items():
        merged.loc[mask, destination_column] = merged.loc[
            mask, source_column
        ].apply(first_or_nan)

    return merged


def get_most_severe_consequence(
    consequence: str, effect_priority_dict: dict
) -> str:
    """
    Process VEP consequence annotations.

    Parameters
    ----------
    consequence : str
        VEP consequence annotation
    effect_priority_dict : dict
        Mapping of VEP consequence terms to their priority

    Returns
    -------
    str
        The single highest priority consequence term
    """
    if not consequence:
        return "intergenic_variant"

    terms = consequence.split("&")

    return min(terms, key=lambda e: effect_priority_dict.get(e, 20))


def split_out_grch37_chrom_pos_ref_alt(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split the 'grch37_norm' column into separate columns for CHROM,
    POS, REF, and ALT.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the 'grch37_norm' column.
    Returns
    -------
    pd.DataFrame
        DataFrame with separate columns for chromosome, position, reference, and alternate alleles.
    """
    # Split into separate parts
    df[["chrom", "pos", "ref", "alt"]] = df["grch37_norm"].str.split(
        "_", expand=True
    )

    return df


def classify_variant_type(ref: str, alt: str) -> tuple[str, bool]:
    """
    Classify a variant as SNP, DEL, INS, DNP, TNP, or ONP using the length
    of the ref and alt alleles. For indels determine whether inframe
    (multiple of 3).

    Parameters
    ----------
    ref : str
        Reference allele
    alt : str
        Alternate allele

    Returns
    -------
    tuple
        variant_type: str
        inframe: bool
    """
    ref_len = len(ref)
    alt_len = len(alt)

    if ref_len == 1 and alt_len == 1:
        return "SNP", False
    elif ref_len > alt_len:
        return "DEL", (abs(ref_len - alt_len) % 3 == 0)
    elif ref_len < alt_len:
        return "INS", (abs(ref_len - alt_len) % 3 == 0)
    elif ref_len == alt_len == 2:
        return "DNP", False
    elif ref_len == alt_len == 3:
        return "TNP", False
    elif ref_len == alt_len and ref_len > 3:
        return "ONP", False
    else:
        return "UNKNOWN", False


def get_variant_classification(
    effect: str,
    effect_map: dict,
    var_type: Optional[str] = None,
    inframe: bool = False,
) -> str:
    """
    Map a single VEP effect, variant type, and inframe info
    to a Variant_Classification string.

    Parameters
    ----------
    effect : str
        The VEP effect annotation.
    effect_map: dict
        Mapping of VEP effect terms to their corresponding Variant_Classification strings.
    var_type : str, optional
        The variant type (SNP, DEL, INS, etc.).
    inframe : bool, optional
        Whether the variant is in-frame.

    Returns
    -------
    str
        The corresponding Variant_Classification string.
    """
    if not effect:
        return "Targeted_Region"

    # Frameshift
    if (
        effect in ("frameshift_variant", "protein_altering_variant")
        and not inframe
    ):
        if var_type == "DEL":
            return "Frame_Shift_Del"
        if var_type == "INS":
            return "Frame_Shift_Ins"

    # In-frame
    if effect.endswith("inframe_insertion") or (
        effect == "protein_altering_variant" and inframe and var_type == "INS"
    ):
        return "In_Frame_Ins"

    if effect.endswith("inframe_deletion") or (
        effect == "protein_altering_variant" and inframe and var_type == "DEL"
    ):
        return "In_Frame_Del"

    # Default lookup
    return effect_map.get(effect, "Targeted_Region")


def merge_fixed_duplicates(
    genie_data: pd.DataFrame, duplicates_fixed_annotations: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge duplicates with correct VEP annotations back into Genie data to
    correct the rows.

    Parameters
    ----------
    genie_data : pd.DataFrame
        The original Genie data.
    duplicates_fixed_annotations : pd.DataFrame
        The DataFrame containing the fixed annotations for duplicates.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame with corrected annotations.
    """
    norm_dups_merged_with_vep_subset = duplicates_fixed_annotations[
        [
            "grch37_norm",
            "VEP_Consequence",
            "VEP_HGVSc",
            "VEP_p",
            "Fixed_Variant_Type",
            "Fixed_Variant_Classification",
        ]
    ]

    fixed_final = pd.merge(
        genie_data,
        norm_dups_merged_with_vep_subset,
        on="grch37_norm",
        how="left",
        indicator=True,
    )

    # Only overwrite the values for rows when merge created the NaN (row is
    # "left_only") and not where VEP annotated it as NaN
    for col_vep, col_orig in [
        ("VEP_Consequence", "Consequence"),
        ("VEP_HGVSc", "HGVSc"),
        ("VEP_p", "HGVSp"),
        ("Fixed_Variant_Type", "Variant_Type"),
        ("Fixed_Variant_Classification", "Variant_Classification"),
    ]:
        mask = (
            fixed_final["_merge"].eq("left_only") & fixed_final[col_vep].isna()
        )
        fixed_final.loc[mask, col_vep] = fixed_final.loc[mask, col_orig]

    fixed_final.drop(
        columns=[
            "_merge",
            "Consequence",
            "HGVSc",
            "HGVSp",
            "Variant_Type",
            "Variant_Classification",
        ],
        inplace=True,
    )
    fixed_final.rename(
        columns={
            "VEP_Consequence": "Consequence",
            "VEP_HGVSc": "HGVSc",
            "VEP_p": "HGVSp",
            "Fixed_Variant_Type": "Variant_Type",
            "Fixed_Variant_Classification": "Variant_Classification",
        },
        inplace=True,
    )

    return fixed_final


def main():
    args = parse_args()
    norm_dups_annotated = read_annotated_vcf_to_df(args.vep_vcf)
    norm_dups_matching_transcripts = norm_dups_annotated[
        norm_dups_annotated["Transcript_ID"]
        == norm_dups_annotated["VEP_Feature"]
    ]

    genie_data_plus_liftover = read_in_to_df(
        args.input,
        header=0,
        usecols=list(column_dtypes.keys()),
        dtype=column_dtypes,
    )

    norm_duplicate_rows = get_normalisation_duplicates(
        genie_data_plus_liftover, "grch37_norm"
    )

    annotations_to_check = [
        "Hugo_Symbol",
        "Consequence",
        "Variant_Classification",
        "Variant_Type",
        "Transcript_ID",
        "RefSeq",
        "HGVSc",
        "HGVSp",
    ]
    agg_duplicates = convert_duplicates_to_one_variant_per_row(
        norm_duplicate_rows,
        annotations_to_check,
        "grch37_norm",
        "grch38_description",
    )

    check_annotations_for_variants(
        agg_duplicates, ["Hugo_Symbol", "Transcript_ID"]
    )

    norm_dups_merged_with_vep = merge_vep_annotations_with_duplicates(
        agg_duplicates, norm_dups_matching_transcripts
    )

    # Extract the most severe consequence from the VEP-annotated consequences
    norm_dups_merged_with_vep["One_Consequence"] = norm_dups_merged_with_vep[
        "VEP_Consequence"
    ].apply(lambda x: get_most_severe_consequence(x, effect_priority))

    norm_dups_merged_with_vep = split_out_grch37_chrom_pos_ref_alt(
        norm_dups_merged_with_vep
    )

    norm_dups_merged_with_vep[["Fixed_Variant_Type", "Inframe"]] = (
        norm_dups_merged_with_vep.apply(
            lambda row: classify_variant_type(row["ref"], row["alt"]),
            axis=1,
            result_type="expand",
        )
    )

    norm_dups_merged_with_vep["Fixed_Variant_Classification"] = (
        norm_dups_merged_with_vep.apply(
            lambda row: get_variant_classification(
                row["One_Consequence"],
                effect_map=effect_map,
                var_type=row["Fixed_Variant_Type"],
                inframe=row["Inframe"],
            ),
            axis=1,
        )
    )

    fixed_final = merge_fixed_duplicates(
        genie_data=genie_data_plus_liftover,
        duplicates_fixed_annotations=norm_dups_merged_with_vep,
    )

    fixed_final.to_csv(
        args.output,
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
