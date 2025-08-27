import argparse
import numpy as np
import pandas as pd
import pysam

from utils.file_io import read_in_to_df
from utils.consequence_priorities import effect_priority, effect_map


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns
    -------
    args : Namespace
        Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser(
        description="Information required to generate counts from MAF file"
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
        help="Path to VCF of duplicates which have been annotated with VEP",
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
    Read in the VCF of the normalisation duplicates annotated by VEP to
    a dataframe.

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
    vcf_in = pysam.VariantFile(vcf_file, "r")

    # Store required data from GRCh38 VCF in a list of dictionaries then
    # convert to a DataFrame
    records = []
    for record in vcf_in:
        row = {
            "CHROM": str(record.chrom),
            "POS": int(record.pos),
            "REF": str(record.ref),
            "ALT": ",".join(str(a) for a in record.alts),
            "Genie_description": record.info.get("Genie_description", None),
            "Transcript_ID": record.info.get("Transcript_ID", None),
            "VEP_VARIANT_CLASS": record.info.get("CSQ_VARIANT_CLASS", None),
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

    for col in [
        "VEP_VARIANT_CLASS",
        "VEP_Consequence",
        "VEP_Feature",
        "VEP_HGVSc",
        "VEP_HGVSp",
    ]:
        vcf_df[col] = vcf_df[col].apply(
            lambda x: x[0] if isinstance(x, tuple) and len(x) == 1 else x
        )

    # Extract the version of the Ensembl transcript
    vcf_df["VEP_Feature_Version"] = (
        vcf_df["VEP_HGVSc"].astype(str).str.split(":").str[0]
    )

    # Replace '.' with NaNs to match the rest of the variants
    cols = [
        "VEP_VARIANT_CLASS",
        "VEP_Consequence",
        "VEP_Feature",
        "VEP_Feature_Version",
        "VEP_HGVSc",
        "VEP_HGVSp",
    ]
    vcf_df[cols] = vcf_df[cols].replace(".", np.nan)

    # Split out just the p. from the HGVSp
    vcf_df["VEP_p"] = vcf_df["VEP_HGVSp"].astype(str).str.split(":").str[1]
    # Replace URL encoding to equals sign
    vcf_df["VEP_p"] = vcf_df["VEP_p"].str.replace("%3D", "=", regex=False)

    # Reorder columns
    vcf_df = vcf_df[
        [
            "grch37_norm",
            "Genie_description",
            "Transcript_ID",
            "VEP_VARIANT_CLASS",
            "VEP_Consequence",
            "VEP_Feature",
            "VEP_Feature_Version",
            "VEP_HGVSc",
            "VEP_HGVSp",
            "VEP_p",
        ]
    ]

    return vcf_df


def get_normalisation_duplicates(genie_data: pd.DataFrame) -> pd.DataFrame:
    """
    Find any rows where they have the same grch38_description but
    the original Genie_description values were different.

    Parameters
    ----------
    genie_data : pd.DataFrame
        DataFrame containing Genie variant data and liftover

    Returns
    -------
    pd.DataFrame
        DataFrame containing rows with conflicting normalisation
    """
    conflicting_norms = (
        genie_data.groupby("grch38_description")["Genie_description"]
        .nunique()
        .reset_index()
        .query("Genie_description > 1")["grch38_description"]
    )

    conflicting_rows = (
        genie_data[genie_data["grch38_description"].isin(conflicting_norms)]
        .drop_duplicates()
        .sort_values(by=["grch38_description", "Genie_description"])
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
    seen = []
    for val in list_of_values:
        if pd.isna(val):
            continue
        if val not in seen:
            seen.append(val)
    if any(pd.isna(val) for val in list_of_values):
        seen.append(np.nan)

    return seen


def convert_duplicates_to_one_row_per_grch38(
    duplicates: pd.DataFrame, annotations_list: list
) -> pd.DataFrame:
    """
    Convert the duplicates DataFrame to one row per grch38_description

    Parameters
    ----------
    duplicates : pd.DataFrame
        DataFrame containing duplicate rows

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per grch38_description
    """
    agg_dict = {
        "grch37_norm": "first",
        "Genie_description": lambda x: pd.unique(
            x.dropna().tolist() + ([np.nan] if x.isna().any() else [])
        ),
    }
    for col in annotations_list:
        agg_dict[col] = lambda x: unique_with_nan(x)

    aggregated_df = (
        duplicates.groupby("grch38_description").agg(agg_dict).reset_index()
    )

    return aggregated_df


def add_columns_for_whether_annotations_same(
    aggregated_dups: pd.DataFrame, annotations_to_check: list
) -> pd.DataFrame:
    """
    Check if the annotations for the normalisation duplicates DataFrame
    are the same across all Genie descriptions

    Parameters
    ----------
    aggregated_dups : pd.DataFrame
        DataFrame with one row per grch38_description
    annotations_to_check : list
        List of annotations to check for consistency

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns indicating if annotations are the same
    """
    # Check whether there are more than 1 value in the annotation column
    # - if there is, then all rows don't have the same annotation and
    # is set to False
    for annotation in annotations_to_check:
        aggregated_dups[f"same_{annotation}"] = aggregated_dups[
            annotation
        ].apply(lambda x: len(set(x)) == 1 if isinstance(x, list) else False)

    return aggregated_dups


def check_whether_transcripts_and_gene_for_dups_are_same(
    aggregated_dups_check: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find any annotations in the Genie data for the same variant that have a
    different Ensembl transcript or different Hugo_Symbol (this would be bad).

    Parameters
    ----------
    aggregated_dups_check : pd.DataFrame
        DataFrame with one row per grch38_description

    Returns
    -------
    pd.DataFrame
        DataFrame containing rows where annotations are against different transcripts
    """
    rows_with_different_tx = aggregated_dups_check[
        ~aggregated_dups_check["same_Transcript_ID"]
    ]

    rows_with_different_symbol = aggregated_dups_check[
        ~aggregated_dups_check["same_Hugo_Symbol"]
    ]

    if (len(rows_with_different_tx) == 0) and (
        len(rows_with_different_symbol) == 0
    ):
        print(
            "No rows with different Transcript_ID or Hugo_Symbol found for the"
            " same normalised variant"
        )
        return True, True
    else:
        print(
            f"Found {len(rows_with_different_tx)} rows with different"
            " RefSeq annotations."
        )
        print(
            f"Found {len(rows_with_different_symbol)} rows found with"
            " different Hugo_Symbol"
        )

        return rows_with_different_tx, rows_with_different_symbol


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
    norm_dups_merged_with_vep = pd.merge(
        duplicates_one_row_per_variant,
        vep_annotations,
        on="grch37_norm",
        how="left",
    )

    # For variant(s) not annotated at all by VEP, take the
    # annotations from the first instance of the variant
    mask = norm_dups_merged_with_vep["VEP_Feature"].isna()
    # Fill the values using the first element of the lists
    norm_dups_merged_with_vep.loc[mask, "VEP_Consequence"] = (
        norm_dups_merged_with_vep.loc[mask, "Consequence"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan
        )
    )
    norm_dups_merged_with_vep.loc[mask, "VEP_HGVSc"] = (
        norm_dups_merged_with_vep.loc[mask, "HGVSc"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan
        )
    )
    norm_dups_merged_with_vep.loc[mask, "VEP_p"] = (
        norm_dups_merged_with_vep.loc[mask, "HGVSp"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan
        )
    )

    return norm_dups_merged_with_vep


def get_most_severe_consequence(
    consequence: str, effect_priority_dict: dict
) -> tuple:
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
    Split the 'grch37_norm' column into separate columns for chromosome, position, reference, and alternate alleles.

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


def classify_variant_type(ref: str, alt: str) -> str:
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
        (variant_type, inframe)
        variant_type: str
        inframe: bool or None
    """
    ref_len = len(ref)
    alt_len = len(alt)

    if ref_len == 1 and alt_len == 1:
        return "SNP", None
    elif ref_len > alt_len:
        return "DEL", (abs(ref_len - alt_len) % 3 == 0)
    elif ref_len < alt_len:
        return "INS", (abs(ref_len - alt_len) % 3 == 0)
    elif ref_len == alt_len == 2:
        return "DNP", None
    elif ref_len == alt_len == 3:
        return "TNP", None
    elif ref_len == alt_len and ref_len > 3:
        return "ONP", None
    else:
        return "UNKNOWN", None


def get_variant_classification(
    effect: str, effect_map: dict, var_type: str = None, inframe: bool = False
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

    for col_vep, col_orig in [
        ("VEP_Consequence", "Consequence"),
        ("VEP_HGVSc", "HGVSc"),
        ("VEP_p", "HGVSp"),
        ("Fixed_Variant_Type", "Variant_Type"),
        ("Fixed_Variant_Classification", "Variant_Classification"),
    ]:
        # Only overwrite when merge created the NaN (row is "left_only")
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
        usecols=[
            "Hugo_Symbol",
            "Strand",
            "Consequence",
            "Variant_Classification",
            "Variant_Type",
            "Chromosome",
            "Start_Position",
            "End_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
            "Tumor_Sample_Barcode",
            "HGVSc",
            "HGVSp",
            "Transcript_ID",
            "RefSeq",
            "Protein_position",
            "Codons",
            "Exon_Number",
            "PATIENT_ID",
            "SAMPLE_ID",
            "AGE_AT_SEQ_REPORT",
            "ONCOTREE_CODE",
            "SAMPLE_TYPE",
            "SEQ_ASSAY_ID",
            "CANCER_TYPE",
            "CANCER_TYPE_DETAILED",
            "SAMPLE_TYPE_DETAILED",
            "SAMPLE_CLASS",
            "variant_description",
            "chrom_grch38",
            "pos_grch38",
            "ref_grch38",
            "alt_grch38",
            "Genie_description",
            "grch37_norm",
            "grch38_description",
        ],
        dtype={
            "Hugo_Symbol": "str",
            "Strand": "str",
            "Chromosome": "str",
            "Start_Position": "Int64",
            "Entrez_Gene_Id": "Int64",
            "Reference_Allele": "str",
            "Tumor_Seq_Allele2": "str",
            "Consequence": "str",
            "Variant_Classification": "str",
            "Variant_Type": "str",
            "Transcript_ID": "str",
            "RefSeq": "str",
            "HGVSc": "str",
            "HGVSp": "str",
            "Protein_position": "str",
            "Exon_Number": "str",
            "chrom_grch38": "str",
            "pos_grch38": "Int64",
            "ref_grch38": "str",
            "alt_grch38": "str",
            "AGE_AT_SEQ_REPORT": "str",
        },
    )

    # Find duplicates where the grch38 values correspond to multiple
    # grch37 norm values
    norm_duplicate_rows = get_normalisation_duplicates(
        genie_data_plus_liftover
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
    agg_duplicates = convert_duplicates_to_one_row_per_grch38(
        norm_duplicate_rows,
        annotations_to_check,
    )

    agg_duplicates_plus_whether_same = (
        add_columns_for_whether_annotations_same(
            agg_duplicates, annotations_to_check
        )
    )

    rows_with_different_tx, rows_with_different_symbol = (
        check_whether_transcripts_and_gene_for_dups_are_same(
            agg_duplicates_plus_whether_same
        )
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
