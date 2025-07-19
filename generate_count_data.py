import argparse
import pandas as pd
import re


from utils import read_in_to_df, read_txt_file_to_list


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
        "--columns_to_aggregate",
        required=True,
        type=str,
        help="TXT file containing a list of variant columns to aggregate",
    )

    parser.add_argument(
        "--haemonc_cancer_types",
        required=False,
        type=str,
        help=(
            "Path to file which lists haemonc cancer types we're interested in"
        ),
    )

    parser.add_argument(
        "--output", required=True, type=str, help="Name of output file"
    )

    return parser.parse_args()


def create_df_with_one_row_per_variant(
    df: pd.DataFrame, columns_to_aggregate: list
) -> pd.DataFrame:
    """
    Create a DataFrame with one row per unique variant by aggregating
    specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input MAF dataframe with variant information
    columns_to_aggregate : list
        List of columns we want to keep and aggregate

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per unique variant and specified
        fields aggregated to unique values (joined by '&')
    """
    aggregated_df = (
        df.groupby("grch38_description")[columns_to_aggregate]
        .agg(lambda x: "&".join(sorted(set(map(str, x)))))
        .reset_index()
    )

    return aggregated_df


def calculate_patient_count_and_per_cancer_type(df):
    """
    Calculate the number of unique patients and number of unique patients
    per cancer type

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns PATIENT_ID, CANCER_TYPE

    Returns
    -------
    total_patient_n: int
        Total number of unique patients across all cancer types
    unique_patients_per_cancer: dict
        Dictionary with cancer types as keys and number of unique patients
        as values
    """
    total_patient_n = df["PATIENT_ID"].nunique()

    unique_patients_per_cancer = (
        df.groupby("CANCER_TYPE")["PATIENT_ID"].nunique().to_dict()
    )

    return total_patient_n, unique_patients_per_cancer


def count_nucleotide_change(
    df: pd.DataFrame,
    unique_patient_total: int,
    unique_patients_per_cancer: dict,
) -> pd.DataFrame:
    """
    Add nucleotide change counts:
    - Total across all cancer types
    - Total per cancer type

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns PATIENT_ID, grch38_description,
        CANCER_TYPE
    unique_patient_total : int
        Total number of unique patients across all cancer types
    unique_patients_per_cancer : dict
        Dictionary with cancer types as keys and number of unique patients
        as values

    Returns
    -------
    pd.DataFrame
        DataFrame with grch38_description and nucleotide change counts
    """
    # Total nucleotide change count across all cancers
    # Drop duplicates for same variant for same patient
    nucleotide_change_total = (
        df.drop_duplicates(
            subset=["PATIENT_ID", "grch38_description"], keep="first"
        )
        .groupby("grch38_description")
        .agg({"PATIENT_ID": "count"})
        .rename(
            columns={
                "PATIENT_ID": (
                    f"NucleotideChange.Total_Count_N_{unique_patient_total}"
                )
            }
        )
        .reset_index()
    )

    # Nucleotide change count per cancer type
    # Drop duplicates for same variant for same patient and cancer type
    # Count patients per nt change and present cancer types
    agg_df = (
        df.drop_duplicates(
            subset=["PATIENT_ID", "grch38_description", "CANCER_TYPE"],
            keep="first",
        )
        .groupby(["grch38_description", "CANCER_TYPE"])
        .agg(patient_count=("PATIENT_ID", "count"))
        .reset_index()
    )

    # Pivot so all cancer types have counts, filling NAs as zero
    pivot_df = agg_df.pivot_table(
        index="grch38_description",
        columns="CANCER_TYPE",
        values="patient_count",
        fill_value=0,
    ).reset_index()

    # Rename columns to include patient N per cancer type
    new_columns = pivot_df.columns
    first_col = new_columns[0]
    renamed_columns = [first_col] + [
        f"NucleotideChange.{cancer_type}_count_N_{unique_patients_per_cancer[cancer_type]}"
        for cancer_type in new_columns[1:]
    ]
    pivot_df.columns = renamed_columns

    # Merge the total counts with the per cancer type counts
    all_nucleotide_counts = pd.merge(
        nucleotide_change_total, pivot_df, on="grch38_description", how="outer"
    )

    return all_nucleotide_counts


def count_amino_acid_change(
    df: pd.DataFrame,
    unique_patient_total: int,
    unique_patients_per_cancer: dict,
):
    """
    Add amino acid change counts:
    - Total across all cancer types
    - Total per cancer type

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns PATIENT_ID, Hugo_Symbol, HGVSp,
        grch38_description, CANCER_TYPE
    unique_patient_total : int
        Total number of unique patients across all cancer types
    unique_patients_per_cancer : dict
        Dictionary with cancer types as keys and number of unique patients
        as values

    Returns
    -------
    pd.DataFrame
        DataFrame with amino acid change counts
    """
    # Amino acid change count across all cancers
    # Drop duplicates for same variant for same patient
    amino_acid_change_count = (
        df.drop_duplicates(
            subset=["PATIENT_ID", "grch38_description"], keep="first"
        )
        .groupby(["Hugo_Symbol", "HGVSp"])
        .size()
        .reset_index(
            name=f"AminoAcidChange.Total_Count_N_{unique_patient_total}"
        )
    )

    # Amino acid change per cancer type
    agg_df = (
        df.drop_duplicates(
            subset=["PATIENT_ID", "grch38_description", "CANCER_TYPE"],
            keep="first",
        )
        .groupby(["Hugo_Symbol", "HGVSp", "CANCER_TYPE"])
        .agg(patient_count=("PATIENT_ID", "count"))
        .reset_index()
    )

    # Pivot so all cancer types have counts, filling NAs as zero
    pivot_df = agg_df.pivot_table(
        index=["Hugo_Symbol", "HGVSp"],
        columns="CANCER_TYPE",
        values="patient_count",
        fill_value=0,
    ).reset_index()

    # Rename columns to include patient N per cancer type
    new_columns = pivot_df.columns
    first_cols = new_columns[0:2]
    renamed_columns = [*first_cols] + [
        f"AminoAcidChange.{cancer_type}_count_N_{unique_patients_per_cancer[cancer_type]}"
        for cancer_type in new_columns[2:]
    ]
    pivot_df.columns = renamed_columns

    all_amino_acid_counts = pd.merge(
        amino_acid_change_count,
        pivot_df,
        on=["Hugo_Symbol", "HGVSp"],
        how="outer",
    )

    return all_amino_acid_counts


def extract_position_from_CDS(hgvsc_value: str) -> int | None:
    """
    Extract the position affected from the HGVSp string

    Parameters
    ----------
    hgvsc_value : str
        HGVS c. string

    Returns
    -------
    int
        Position affected, or None if not found
    """
    if not isinstance(hgvsc_value, str):
        return None
    match = re.search(r"-?\d+", hgvsc_value.replace("c.", ""))

    return int(match.group()) if match else None


def extract_position_affected(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the position affected from the HGVSp string and add it as a new
    column 'position_affected'

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the HGVSc column

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column 'CDS_position' containing the
        position affected extracted from the HGVSc string
    """
    df = df.copy()
    # Split out the CDS change from the HGVSc string
    df["CDS.change"] = df["HGVSc"].str.split(":", 1, expand=True)[1]
    df["CDS_position"] = df["CDS.change"].apply(extract_position_from_CDS)

    return df


def count_downstream_patients(group):
    """
    Count how many unique patients have a variant at the same position or
    downstream in the same gene

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing variants in the same gene

    Returns
    -------
    pd.DataFrame
        DataFrame with CDS_position and downstream_patient_count
    """
    group = group.copy()
    group["CDS_position"] = pd.to_numeric(
        group["CDS_position"], errors="coerce"
    )
    unique_positions = sorted(group["CDS_position"].unique())
    result_rows = []
    for pos in unique_positions:
        same_or_downstream = group[group["CDS_position"] >= pos]
        count = same_or_downstream["PATIENT_ID"].nunique()
        result_rows.append(
            {"CDS_position": pos, "downstream_patient_count": count}
        )
    return pd.DataFrame(result_rows)


def get_truncating_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get truncating variants from the DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the Genie data with columns PATIENT_ID,
        Hugo_Symbol, HGVSp, Variant_Classification

    Returns
    -------
    pd.DataFrame
        DataFrame with truncating variants
    """
    truncating = df["Variant_Classification"].isin(
        [
            "Frame_Shift_Del",
            "Frame_Shift_Ins",
            "Nonsense_Mutation",
        ]
    ) & (df["HGVSp"].str.contains("Ter", na=False))

    return df[truncating].copy()


def add_frameshift_truncating_and_nonsense_count(
    truncating_plus_position: pd.DataFrame,
    patient_total: int,
) -> pd.DataFrame:
    """
    Add count for frameshift (truncating) and nonsense variants:
    - Count how many patients have frameshift (truncating) or nonsense variant
    at that position or downstream in the same gene for all cancers

    Parameters
    ----------
    truncating_plus_position : pd.DataFrame
        DataFrame containing truncating variants with position
    patient_total : int
        Total number of unique patients across all cancer types

    Returns
    -------
    pd.DataFrame
        DataFrame with frameshift (truncating) and nonsense counts
    """
    truncating_dedup = truncating_plus_position.drop_duplicates(
        subset=["PATIENT_ID", "grch38_description"],
        keep="first",
    )

    # Sort by CDS position within gene with highest position first
    truncating_sorted = truncating_dedup.sort_values(
        by=["Hugo_Symbol", "CDS_position"], ascending=[True, False]
    )

    # For each gene, add counts at each position or downstream
    result = (
        truncating_sorted.groupby("Hugo_Symbol", group_keys=False)
        .apply(count_downstream_patients)
        .reset_index()
    )

    # Rename column to add patient N
    result = result.rename(
        columns={
            "downstream_patient_count": (
                f"SameOrDownstreamTruncatingVariantsPerCDS.Total_Count_N_{patient_total}"
            )
        }
    )

    return result


def add_frameshift_truncating_and_nonsense_counts_per_cancer(
    truncating_plus_position: pd.DataFrame,
    per_cancer_patient_total: dict,
):
    """
    Add counts for frameshift (truncating) and nonsense variants:
    - Count how many patients have frameshift (truncating) or nonsense variant
    at that position or downstream in the same gene per cancer type

    Parameters
    ----------
    truncating_plus_position : pd.DataFrame
        DataFrame containing truncating variants with position
    per_cancer_patient_total : dict
        Dictionary with cancer types as keys and number of unique patients
        as values

    Returns
    -------
    pd.DataFrame
        DataFrame with frameshift (truncating) and nonsense counts per cancer
    """
    # Remove variants for the same patient for the same cancer type
    truncating_per_cancer_dedup = truncating_plus_position.drop_duplicates(
        subset=["PATIENT_ID", "grch38_description", "CANCER_TYPE"],
        keep="first",
    )

    truncating_sorted = truncating_per_cancer_dedup.sort_values(
        by=["Hugo_Symbol", "CDS_position", "CANCER_TYPE"]
    )

    # Apply per gene and cancer type
    result = (
        truncating_sorted.groupby(["Hugo_Symbol", "CANCER_TYPE"])
        .apply(count_downstream_patients)
        .reset_index()
    )

    all_cancers = list(per_cancer_patient_total.keys())
    full_index = (
        truncating_sorted.groupby("Hugo_Symbol")["CDS_position"]
        .unique()
        .explode()
        .reset_index()
        .rename(columns={0: "CDS_position"})
    )

    # Cross join with all cancer types
    full_index = (
        full_index.assign(key=1)
        .merge(pd.DataFrame({"CANCER_TYPE": all_cancers, "key": 1}), on="key")
        .drop(columns="key")
    )

    # Merge actual results with complete index and fill zeros
    result_filled = full_index.merge(
        result, on=["Hugo_Symbol", "CDS_position", "CANCER_TYPE"], how="left"
    )
    result_filled["downstream_patient_count"] = (
        result_filled["downstream_patient_count"].fillna(0).astype(int)
    )

    # Pivot to final format
    per_cancer_pivot = result_filled.pivot_table(
        index=["Hugo_Symbol", "CDS_position"],
        columns="CANCER_TYPE",
        values="downstream_patient_count",
        fill_value=0,
    ).reset_index()

    # Rename columns to include patient N per cancer type
    new_columns = []
    for col in per_cancer_pivot.columns:
        if col in per_cancer_patient_total:
            new_columns.append(
                f"SameOrDownstreamTruncatingVariantsPerCDS.{col}_count_N_{per_cancer_patient_total[col]}"
            )
        else:
            new_columns.append(col)

    per_cancer_pivot.columns = new_columns

    return per_cancer_pivot


def calculate_count_for_all_haemonc_cancers(
    genie_data_with_sample_info, haemonc_cancer_types, variant_columns
):
    """
    Create df with one row per unique variant and a count of each variant in
    haemonc cancers

    Parameters
    ----------
    genie_data_with_sample_info : pd.DataFrame
        Genie data to calculate counts from
    haemonc_cancer_types : list
        List of haemonc cancer types (strings) to subset to (must match the
        names of the cancer types in the Genie file)

    Returns
    -------
    haemonc_cancer_counts : pd.DataFrame
        Count data for each variant in haemonc cancers
    """
    # Create multiindex of all unique variants in the dataset
    all_variant_index = genie_data_with_sample_info.drop_duplicates(
        subset=variant_columns
    )[variant_columns]

    # Create index from the four variant columns
    all_variant_index = all_variant_index.set_index(
        variant_columns
    ).sort_index()

    # Subset to just haemonc cancer types
    haemonc_cancer_type_rows = genie_data_with_sample_info[
        genie_data_with_sample_info["CANCER_TYPE"].isin(haemonc_cancer_types)
    ]

    subset_fields = ["PATIENT_ID"] + variant_columns
    # Group by patient and variant and count
    haemonc_cancer_counts = (
        haemonc_cancer_type_rows.drop_duplicates(subset=subset_fields)
        .groupby(variant_columns)
        .agg(exact_haemonc_cancers_count=("PATIENT_ID", "count"))
    )

    # Use the index made earlier to set any variants which aren't present in
    # any haemonc cancers to a count of zero
    haemonc_cancer_counts = haemonc_cancer_counts.reindex(
        all_variant_index.index, fill_value=0
    ).reset_index()

    return haemonc_cancer_counts


# def convert_count(x):
#     """
#     Convert a count to int (if exists) or return '.' if not a valid variant
#     type for that count

#     Parameters
#     ----------
#     x : _type_
#         _description_

#     Returns
#     -------
#     _type_
#         _description_
#     """
#     if x in [".", None] or pd.isna(x):
#         return "."
#     return int(x)


def merge_counts_together(df1, df2, variant_columns):
    """
    Merge the counts together to get a single dataframe with multiple counts
    (all have one row per unique variant so will have same number of rows)

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe to merge
    df2 : pd.DataFrame
        Second dataframe to merge

    Returns
    -------
    merged_counts : pd.DataFrame
        Merged dataframe with all counts
    """
    merged_counts = pd.merge(
        df1,
        df2,
        on=variant_columns,
        how="left",
    )

    return merged_counts


def reorder_final_columns(df, patient_total, per_cancer_patient_total):
    """
    Reorder the final DataFrame columns to match the expected output format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the counts and variant information
    patient_total : int
        Total number of unique patients across all cancer types
    per_cancer_patient_total : dict
        Dictionary with cancer types as keys and number of unique patients
        as values

    Returns
    -------
    pd.DataFrame
        DataFrame with columns reordered to match the expected output format
    """
    # Set the total count to be at the beginning of all count columns
    first_cols = [
        "Hugo_Symbol",
        "Entrez_Gene_Id",
        "grch38_description",
        "grch37_norm",
        "Genie_description",
        "Transcript_ID",
        "RefSeq",
        "Consequence",
        "HGVSc",
        "HGVSp",
    ]
    count_cols = [
        f"NucleotideChange.Total_Count_N_{patient_total}",
        f"AminoAcidChange.Total_Count_N_{patient_total}",
        f"SameOrDownstreamTruncatingVariantsPerCDS.Total_Count_N_{patient_total}",
    ]
    # Add in cancer type columns in cancer order
    for cancer_type in per_cancer_patient_total.keys():
        count_cols.extend(
            [
                f"NucleotideChange.{cancer_type}_count_N_{per_cancer_patient_total[cancer_type]}",
                f"AminoAcidChange.{cancer_type}_count_N_{per_cancer_patient_total[cancer_type]}",
                f"SameOrDownstreamTruncatingVariantsPerCDS.{cancer_type}_count_N_{per_cancer_patient_total[cancer_type]}",
            ]
        )

    # Reorder the DataFrame columns
    other_cols = [
        col
        for col in df.columns
        if (col not in count_cols) and (col not in first_cols)
    ]
    final_col_order = first_cols + other_cols + count_cols
    reordered_df = df[final_col_order]

    reordered_df.drop(columns=["level_1", "CDS_position"], inplace=True)

    return reordered_df


def main():
    args = parse_args()
    genie_data = read_in_to_df(
        args.input,
        header=0,
        dtype={
            "Hugo_Symbol": "str",
            "Start_Position": "Int64",
            "Entrez_Gene_Id": "Int64",
        },
        converters={
            col: lambda x: str(x).strip() if pd.notnull(x) else x
            for col in [
                "Chromosome",
                "Reference_Allele",
                "Tumor_Seq_Allele2",
                "chrom_grch38",
                "pos_grch38",
                "ref_grch38",
                "alt_grch38",
            ]
        },
    )

    columns_to_aggregate = read_txt_file_to_list(args.columns_to_aggregate)

    patient_total, per_cancer_patient_total = (
        calculate_patient_count_and_per_cancer_type(genie_data)
    )

    one_row_per_variant_agg = create_df_with_one_row_per_variant(
        df=genie_data,
        columns_to_aggregate=columns_to_aggregate,
    )

    nucleotide_change_counts = count_nucleotide_change(
        df=genie_data,
        unique_patient_total=patient_total,
        unique_patients_per_cancer=per_cancer_patient_total,
    )

    merged_nucleotide_counts = pd.merge(
        one_row_per_variant_agg,
        nucleotide_change_counts,
        on="grch38_description",
        how="left",
    )

    amino_acid_change_counts = count_amino_acid_change(
        df=genie_data,
        unique_patient_total=patient_total,
        unique_patients_per_cancer=per_cancer_patient_total,
    )

    merged_amino_acid_counts = pd.merge(
        merged_nucleotide_counts,
        amino_acid_change_counts,
        on=["Hugo_Symbol", "HGVSp"],
        how="left",
    )

    truncating_variants = get_truncating_variants(genie_data)
    truncating_plus_position = extract_position_affected(truncating_variants)

    fs_counts = add_frameshift_truncating_and_nonsense_count(
        truncating_plus_position=truncating_plus_position,
        patient_total=patient_total,
    )

    fs_per_cancer_count = (
        add_frameshift_truncating_and_nonsense_counts_per_cancer(
            truncating_plus_position=truncating_plus_position,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )

    all_fs_counts = pd.merge(
        fs_counts,
        fs_per_cancer_count,
        on=["Hugo_Symbol", "CDS_position"],
        how="left",
    )

    truncating_plus_position_no_dups = (
        truncating_plus_position.drop_duplicates(
            subset="grch38_description", keep="first"
        )[["Hugo_Symbol", "grch38_description", "CDS_position"]]
    )

    truncating_with_counts = pd.merge(
        truncating_plus_position_no_dups,
        all_fs_counts,
        on=["Hugo_Symbol", "CDS_position"],
        how="left",
    )

    genie_plus_fs_counts = pd.merge(
        merged_amino_acid_counts,
        truncating_with_counts,
        on=["Hugo_Symbol", "grch38_description"],
        how="left",
    )

    final_df = reorder_final_columns(
        genie_plus_fs_counts,
        patient_total,
        per_cancer_patient_total,
    )

    final_df.to_csv(
        args.output,
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
