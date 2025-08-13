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


def calculate_patient_count_and_per_cancer_type(df):
    """
    Calculate the number of unique patients overall and per cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns PATIENT_ID, CANCER_TYPE

    Returns
    -------
    total_patient_n: int
        Total number of unique patients overall
    unique_patient_n_per_cancer: dict
        Dictionary with cancer types as keys and number of unique patients
        as values
    """
    total_patient_n = df["PATIENT_ID"].nunique()

    unique_patient_n_per_cancer = (
        df.groupby("CANCER_TYPE")["PATIENT_ID"].nunique().to_dict()
    )

    return total_patient_n, unique_patient_n_per_cancer


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
        List of columns we want to keep for each variant and aggregate

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


def deduplicate_variant_by_patient(df):
    """
    Remove multiple instances of the same variant (GRCh38 description) for the
    same patient.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns PATIENT_ID, grch38_description

    Returns
    -------
    pd.DataFrame
        DataFrame with unique variants per patient and variant description
    """
    df_deduplicated_by_patient = df.drop_duplicates(
        subset=["PATIENT_ID", "grch38_description"], keep="first"
    )

    return df_deduplicated_by_patient


def deduplicate_variant_by_patient_and_cancer_type(df):
    """
    Remove multiple instances of the same variant (GRCh38 description) for the same patient and for the same cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns PATIENT_ID, grch38_description, CANCER_TYPE

    Returns
    -------
    pd.DataFrame
        DataFrame with unique variants per patient, variant description and cancer type
    """
    df_deduplicated_by_pt_and_cancer = df.drop_duplicates(
        subset=["PATIENT_ID", "grch38_description", "CANCER_TYPE"],
        keep="first",
    )

    return df_deduplicated_by_pt_and_cancer


def count_nucleotide_change_all_cancers(
    df: pd.DataFrame, unique_patient_total: int
) -> pd.DataFrame:
    """
    Count how many patients have the exact variant across all cancers.

    Parameters
    ----------
    df : pd.DataFrame
        Input Genie MAF data deduplicated by variant and patient
    unique_patient_total : int
        Total number of unique patients in the dataset

    Returns
    -------
    pd.DataFrame
        DataFrame with nucleotide change counts across all cancer types
    """
    # Group by variant, count patients and add patient N to the column name
    nucleotide_change_counts = (
        df.groupby("grch38_description")
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

    return nucleotide_change_counts


def count_nucleotide_change_per_cancer_type(
    df: pd.DataFrame,
    unique_patients_per_cancer: dict,
) -> pd.DataFrame:
    """
    Count how many patients have the exact variant per cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        Input Genie MAF data deduplicated by variant, patient and cancer type
    unique_patients_per_cancer : dict
        Total number of unique patients in the dataset per cancer type

    Returns
    -------
    pd.DataFrame
        DataFrame with grch38_description and nucleotide change counts
    """
    # Group by variant and cancer type, count patients and add patient N
    # per cancer type to the column name
    agg_df = (
        df.groupby(["grch38_description", "CANCER_TYPE"])
        .agg(patient_count=("PATIENT_ID", "count"))
        .reset_index()
    )

    # Pivot so all cancer types present have counts, filling NAs with zero
    pivot_df = agg_df.pivot_table(
        index="grch38_description",
        columns="CANCER_TYPE",
        values="patient_count",
        fill_value=0,
    ).reset_index()

    # Rename columns to include patient N for the cancer type count columns
    pivot_df.columns = [
        (
            "grch38_description"
            if i == 0
            else f"NucleotideChange.{col}_Count_N_{unique_patients_per_cancer[col]}"
        )
        for i, col in enumerate(pivot_df.columns)
    ]

    return pivot_df


def merge_nucleotide_counts(
    df: pd.DataFrame,
    nucleotide_change_counts_all_cancers: pd.DataFrame,
    nucleotide_change_counts_per_cancer: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the nucleotide change counts across all cancers and per cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with variant information
    nucleotide_change_counts_all_cancers : pd.DataFrame
        DataFrame with nucleotide change counts across all cancer types
    nucleotide_change_counts_per_cancer : pd.DataFrame
        DataFrame with nucleotide change counts per cancer type

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with nucleotide change counts
    """
    merged_df = pd.merge(
        df,
        nucleotide_change_counts_all_cancers,
        on="grch38_description",
        how="left",
    )

    merged_df = pd.merge(
        merged_df,
        nucleotide_change_counts_per_cancer,
        on="grch38_description",
        how="left",
    )

    return merged_df


def count_amino_acid_change_all_cancers(
    df: pd.DataFrame, unique_patient_total: int
) -> pd.DataFrame:
    """
    Count how many patients have the same amino acid change in that gene
    across all cancers.

    Parameters
    ----------
    df : pd.DataFrame
        Input Genie MAF data
    unique_patient_total : int
        Total number of unique patients in the dataset

    Returns
    -------
    pd.DataFrame
        DataFrame with amino acid change counts across all cancer types
    """
    # Group by gene and amino acid changes and count patients, adding
    # patient N to the column name
    amino_acid_change_counts = (
        df.groupby(["Hugo_Symbol", "HGVSp"])
        .agg({"PATIENT_ID": "count"})
        .rename(
            columns={
                "PATIENT_ID": (
                    f"AminoAcidChange.Total_Count_N_{unique_patient_total}"
                )
            }
        )
        .reset_index()
    )

    return amino_acid_change_counts


def count_amino_acid_change_per_cancer_type(
    df: pd.DataFrame,
    unique_patients_per_cancer: dict,
):
    """
    Count how many patients have the same amino acid change in that gene
    per cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        Input Genie MAF data
    unique_patient_total : int
        Total number of unique patients in the dataset
    unique_patients_per_cancer : dict
        Total number of unique patients in the dataset per cancer type

    Returns
    -------
    pd.DataFrame
        DataFrame with amino acid change counts per cancer
    """
    # Amino acid change per cancer type
    amino_acid_count_per_cancer = (
        df.groupby(["Hugo_Symbol", "HGVSp", "CANCER_TYPE"])
        .agg(patient_count=("PATIENT_ID", "count"))
        .reset_index()
    )

    # Pivot so all cancer types have counts, filling NAs as zero
    pivot_df = amino_acid_count_per_cancer.pivot_table(
        index=["Hugo_Symbol", "HGVSp"],
        columns="CANCER_TYPE",
        values="patient_count",
        fill_value=0,
    ).reset_index()

    # Rename columns to include patient N per cancer type
    pivot_df.columns = [
        (
            col
            if i < 2
            else f"AminoAcidChange.{col}_Count_N_{unique_patients_per_cancer[col]}"
        )
        for i, col in enumerate(pivot_df.columns)
    ]

    return pivot_df


def merge_amino_acid_counts(
    df: pd.DataFrame,
    amino_acid_counts_all_cancers: pd.DataFrame,
    amino_acid_counts_per_cancer: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the amino acid counts across all cancers and per cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with nucleotide change counts
    amino_acid_counts_all_cancers : pd.DataFrame
        DataFrame with amino acid change counts across all cancer types
    amino_acid_counts_per_cancer : pd.DataFrame
        DataFrame with amino acid change counts per cancer type

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with amino acid change counts
    """
    merged_df = pd.merge(
        df,
        amino_acid_counts_all_cancers,
        on=["Hugo_Symbol", "HGVSp"],
        how="left",
    )

    merged_df = pd.merge(
        merged_df,
        amino_acid_counts_per_cancer,
        on=["Hugo_Symbol", "HGVSp"],
        how="left",
    )

    return merged_df


def extract_position_from_cds(hgvsc_value: str) -> int | None:
    """
    Extract the position affected from the HGVSc string

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
    Extract the position affected from the HGVSp string for frameshift
    (truncating) and nonsense variants and add new columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the HGVSc column

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column 'CDS.change' with just the c.
        notation and column 'CDS_position' containing the first aa affected
    """
    df = df.copy()
    # Split out the CDS change and position from HGVSc string
    df["CDS.change"] = df["HGVSc"].str.split(":", 1, expand=True)[1]
    df["CDS_position"] = df["CDS.change"].apply(extract_position_from_cds)

    return df


def get_truncating_variants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rows of truncating variants from the Genie data.

    Parameters
    ----------
    df : pd.DataFrame
        Input Genie MAF data

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


def count_patients_with_variant_at_same_position_or_downstream(gene_group):
    """
    Count how many unique patients which have a variant at the same
    position or downstream in the same gene.

    Parameters
    ----------
    group : pd.DataFrame
        DataFrame containing variants in the same gene

    Returns
    -------
    pd.DataFrame
        DataFrame with CDS_position and downstream_patient_count
    """
    gene_group = gene_group.copy()
    gene_group["CDS_position"] = pd.to_numeric(
        gene_group["CDS_position"], errors="coerce"
    )

    # Loop over each unique position, get rows with the same position or later
    # then count how many unique patients that represents
    unique_positions = sorted(gene_group["CDS_position"].unique())
    result_rows = []
    for pos in unique_positions:
        same_or_downstream = gene_group[gene_group["CDS_position"] >= pos]
        count = same_or_downstream["PATIENT_ID"].nunique()
        result_rows.append(
            {"CDS_position": pos, "downstream_patient_count": count}
        )
    return pd.DataFrame(result_rows)


def count_frameshift_truncating_and_nonsense_all_cancers(
    df: pd.DataFrame,
    patient_total: int,
) -> pd.DataFrame:
    """
    Count how many patients have a frameshift (truncating) or nonsense variant
    at the same position or downstream in the same gene.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing truncating variants with position
    patient_total : int
        Total number of unique patients in the dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with frameshift (truncating) and nonsense counts
    """
    # For each gene, add counts at each position or downstream
    result = (
        df.groupby("Hugo_Symbol")
        .apply(count_patients_with_variant_at_same_position_or_downstream)
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


def count_frameshift_truncating_and_nonsense_per_cancer_type(
    df: pd.DataFrame,
    per_cancer_patient_total: dict,
):
    """
    Count how many patients have a frameshift (truncating) or nonsense variant
    at the same position or downstream in the same gene per cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing truncating variants with position
    per_cancer_patient_total : dict
        Total number of unique patients in the dataset per cancer type

    Returns
    -------
    pd.DataFrame
        DataFrame with frameshift (truncating) and nonsense counts per cancer
    """
    # Apply per gene and cancer type
    result = (
        df.groupby(["Hugo_Symbol", "CANCER_TYPE"])
        .apply(count_patients_with_variant_at_same_position_or_downstream)
        .reset_index()
    )

    # Get all unique gene + CDS position pairs
    all_cancers = list(per_cancer_patient_total.keys())
    full_index = (
        df.groupby("Hugo_Symbol")["CDS_position"]
        .unique()
        .explode()
        .reset_index()
        .rename(columns={0: "CDS_position"})
    )

    # Ceate all combinations with cancer types (cross join)
    full_index = (
        full_index.assign(key=1)
        .merge(pd.DataFrame({"CANCER_TYPE": all_cancers, "key": 1}), on="key")
        .drop(columns="key")
    )

    # Merge actual counts with complete index and fill NAs with zeros
    result_filled = full_index.merge(
        result, on=["Hugo_Symbol", "CDS_position", "CANCER_TYPE"], how="left"
    )
    result_filled["downstream_patient_count"] = (
        result_filled["downstream_patient_count"].fillna(0).astype(int)
    )

    # Pivot table so each row is a gene and CDS position pair and each column
    # is a cancer type, filling any missing with zeros
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
                f"SameOrDownstreamTruncatingVariantsPerCDS.{col}_Count_N_{per_cancer_patient_total[col]}"
            )
        else:
            new_columns.append(col)
    per_cancer_pivot.columns = new_columns

    return per_cancer_pivot


def merge_truncating_variants_counts(
    truncating_variants: pd.DataFrame,
    truncating_counts_all_cancers: pd.DataFrame,
    truncating_counts_per_cancer: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge the truncating variant counts across all cancers and per cancer type.

    Parameters
    ----------
    truncating_variants : pd.DataFrame
        DataFrame with truncating variants and their positions
    truncating_counts_all_cancers : pd.DataFrame
        DataFrame with truncating variant counts across all cancer types
    truncating_counts_per_cancer : pd.DataFrame
        DataFrame with truncating variant counts per cancer type

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with truncating variant counts
    """
    merged_frameshift_counts = pd.merge(
        truncating_counts_all_cancers,
        truncating_counts_per_cancer,
        on=["Hugo_Symbol", "CDS_position"],
    )

    merged_with_variants = pd.merge(
        truncating_variants,
        merged_frameshift_counts,
        on=["Hugo_Symbol", "CDS_position"],
        how="left",
    )

    return merged_with_variants


def get_inframe_deletions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get inframe deletions from the DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the Genie data

    Returns
    -------
    pd.DataFrame
        DataFrame with inframe deletion variants
    """
    inframe_deletions = df[
        df["Variant_Classification"] == "In_Frame_Del"
    ].copy()

    # Remove any where HGVSc is NaN as we won't get positions from these
    inframe_deletions = inframe_deletions[inframe_deletions["HGVSc"].notna()]

    return inframe_deletions


def extract_hgvsc_deletion_positions(hgvsc):
    """
    Extract deletion start and end from an HGVSc string like:
    'ENST00000269305.4:c.480_485del' or 'ENST00000296930.5:c.511_524+1del'.
    Ignore intronic offsets, extracting only numeric coding coordinates.
    TODO - handle offsets

    Parameters
    ----------
    hgvsc : str
        HGVSc string to extract deletion positions from

    Returns
    -------
    tuple
        A tuple containing the start and end positions of the deletion,
        or (None, None) if the format is not recognised
    """
    if not isinstance(hgvsc, str):
        return None, None

    # Split on colon to get only the c. portion
    if ":" in hgvsc:
        c_part = hgvsc.split(":")[1]
    else:
        c_part = hgvsc

    # Match deletion pattern
    match = re.match(r"c\.(\d+)(?:_(\d+))?(?:[\+\-]\d+)?del", c_part)

    if match:
        start = match.group(1)
        end = match.group(2) if match.group(2) else start
        return start, end

    return None, None


def add_deletion_positions(inframe_deletions: pd.DataFrame) -> pd.DataFrame:
    """
    Add start and end positions to inframe deletions.

    Parameters
    ----------
    inframe_deletions : pd.DataFrame
        DataFrame containing inframe deletions with HGVSc column

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns for deletion start and end
    """
    inframe_deletions[["del_start", "del_end"]] = inframe_deletions[
        "HGVSc"
    ].apply(lambda x: pd.Series(extract_hgvsc_deletion_positions(x)))

    return inframe_deletions


def count_patients_with_nested_deletions(gene_group):
    """
    Count how many unique patients have deletions that are the same
    as or nested within each deletion in the same gene.

    Parameters
    ----------
    gene_group : pd.DataFrame
        DataFrame containing deletions for a single gene (and optionally a cancer type)

    Returns
    -------
    pd.DataFrame
        DataFrame with del_start, del_end, and nested_patient_count
    """
    gene_group = gene_group.copy()

    # Get unique deletion ranges
    unique_deletions = (
        gene_group[["del_start", "del_end"]]
        .drop_duplicates()
        .sort_values(["del_start", "del_end"])
    )

    result_rows = []
    for _, row in unique_deletions.iterrows():
        current_start = row["del_start"]
        current_end = row["del_end"]

        # Select deletions that are nested within the current one
        nested = gene_group[
            (gene_group["del_start"] >= current_start)
            & (gene_group["del_end"] <= current_end)
        ]

        count = nested["PATIENT_ID"].nunique()
        result_rows.append(
            {
                "del_start": current_start,
                "del_end": current_end,
                "nested_patient_count": count,
            }
        )

    return pd.DataFrame(result_rows)


def count_same_or_nested_inframe_deletions_all_cancers(
    inframe_deletions_df: pd.DataFrame, patient_total: int
) -> pd.DataFrame:
    """
    Count the number of unique patients with inframe deletions that are either
    the same as or nested within the current deletion for all cancers.

    Parameters
    ----------
    inframe_deletions_df : pd.DataFrame
        DataFrame containing inframe deletions with patient information.
    patient_total : int
        Total number of unique patients in the dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with counts of matching or nested inframe deletions.
    """
    inframe_counts = (
        inframe_deletions_df.groupby("Hugo_Symbol")
        .apply(count_patients_with_nested_deletions)
        .reset_index()
    ).rename(
        columns={
            "nested_patient_count": (
                f"NestedInframeDeletionsPerCDS.Total_Count_N_{patient_total}"
            )
        }
    )

    return inframe_counts


def count_same_or_nested_inframe_deletions_per_cancer_type(
    inframe_deletions_df: pd.DataFrame, per_cancer_patient_total: dict
) -> pd.DataFrame:
    """
    Count the number of unique patients with inframe deletions that are either
    the same as or nested within the current deletion, grouped by cancer type.

    Parameters
    ----------
    inframe_deletions_df : pd.DataFrame
        DataFrame containing inframe deletions with patient information.
    per_cancer_patient_total : dict
        Total number of unique patients in the dataset per cancer type.

    Returns
    -------
    pd.DataFrame
        DataFrame with counts of matching or nested inframe deletions per cancer type.
    """
    result = inframe_deletions_df.copy()
    nested_per_cancer_counts = (
        result.groupby(["Hugo_Symbol", "CANCER_TYPE"])
        .apply(count_patients_with_nested_deletions)
        .reset_index()
    )

    all_cancers = list(per_cancer_patient_total.keys())
    full_index = (
        result[["Hugo_Symbol", "del_start", "del_end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    full_index = (
        full_index.assign(key=1)
        .merge(pd.DataFrame({"CANCER_TYPE": all_cancers, "key": 1}), on="key")
        .drop(columns="key")
    )

    nested_counts_filled = full_index.merge(
        nested_per_cancer_counts,
        on=["Hugo_Symbol", "CANCER_TYPE", "del_start", "del_end"],
        how="left",
    )

    nested_counts_filled["nested_patient_count"] = (
        nested_counts_filled["nested_patient_count"].fillna(0).astype(int)
    )

    # Pivot table so each row is a gene and CDS position pair and each column
    # is a cancer type, filling any missing with zeros
    per_cancer_pivot = nested_counts_filled.pivot_table(
        index=["Hugo_Symbol", "del_start", "del_end"],
        columns="CANCER_TYPE",
        values="nested_patient_count",
        fill_value=0,
    ).reset_index()

    # Rename columns to include patient N per cancer type
    new_columns = []
    for col in per_cancer_pivot.columns:
        if col in per_cancer_patient_total:
            new_columns.append(
                f"NestedInframeDeletionsPerCDS.{col}_Count_N_{per_cancer_patient_total[col]}"
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
        f"NestedInframeDeletionsPerCDS.Total_Count_N_{patient_total}",
    ]
    # Add in cancer type columns in cancer order``
    for cancer_type in per_cancer_patient_total.keys():
        count_cols.extend(
            [
                f"NucleotideChange.{cancer_type}_Count_N_{per_cancer_patient_total[cancer_type]}",
                f"AminoAcidChange.{cancer_type}_Count_N_{per_cancer_patient_total[cancer_type]}",
                f"SameOrDownstreamTruncatingVariantsPerCDS.{cancer_type}_Count_N_{per_cancer_patient_total[cancer_type]}",
                f"NestedInframeDeletionsPerCDS.{cancer_type}_Count_N_{per_cancer_patient_total[cancer_type]}",
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

    reordered_df.drop(
        columns=[
            "level_1_x",
            "level_1_y",
            "CDS_position",
            "del_start",
            "del_end",
        ],
        inplace=True,
    )

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

    deduplicated_by_patient = deduplicate_variant_by_patient(
        df=genie_data,
    )

    deduplicated_by_patient_and_cancer = (
        deduplicate_variant_by_patient_and_cancer_type(
            df=deduplicated_by_patient
        )
    )

    nucleotide_change_counts_all_cancer = count_nucleotide_change_all_cancers(
        df=genie_data, unique_patient_total=patient_total
    )

    nucleotide_change_counts_per_cancer = (
        count_nucleotide_change_per_cancer_type(
            df=deduplicated_by_patient_and_cancer,
            unique_patients_per_cancer=per_cancer_patient_total,
        )
    )

    merged_nucleotide_counts = merge_nucleotide_counts(
        df=one_row_per_variant_agg,
        nucleotide_change_counts_all_cancers=nucleotide_change_counts_all_cancer,
        nucleotide_change_counts_per_cancer=nucleotide_change_counts_per_cancer,
    )

    amino_acid_change_counts_all_cancer = count_amino_acid_change_all_cancers(
        df=genie_data,
        unique_patient_total=patient_total,
    )

    amino_acid_change_counts_per_cancer = (
        count_amino_acid_change_per_cancer_type(
            df=deduplicated_by_patient_and_cancer,
            unique_patients_per_cancer=per_cancer_patient_total,
        )
    )

    merged_amino_acid_counts = merge_amino_acid_counts(
        df=merged_nucleotide_counts,
        amino_acid_counts_all_cancers=amino_acid_change_counts_all_cancer,
        amino_acid_counts_per_cancer=amino_acid_change_counts_per_cancer,
    )

    truncating_variants = get_truncating_variants(genie_data)
    truncating_plus_position = extract_position_affected(truncating_variants)

    frameshift_counts_all_cancers = (
        count_frameshift_truncating_and_nonsense_all_cancers(
            df=truncating_plus_position,
            patient_total=patient_total,
        )
    )

    frameshift_counts_per_cancer = (
        count_frameshift_truncating_and_nonsense_per_cancer_type(
            df=truncating_plus_position,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )

    truncating_plus_position_no_dups = (
        truncating_plus_position.drop_duplicates(
            subset="grch38_description", keep="first"
        )[["Hugo_Symbol", "grch38_description", "CDS_position"]]
    )

    merged_frameshift_counts_only = merge_truncating_variants_counts(
        truncating_variants=truncating_plus_position_no_dups,
        truncating_counts_all_cancers=frameshift_counts_all_cancers,
        truncating_counts_per_cancer=frameshift_counts_per_cancer,
    )

    merged_frameshift_counts = pd.merge(
        merged_amino_acid_counts,
        merged_frameshift_counts_only,
        on=["grch38_description", "Hugo_Symbol"],
        how="left",
    )

    inframe_deletions = get_inframe_deletions(genie_data)
    inframe_deletions_with_positions = add_deletion_positions(
        inframe_deletions
    )
    inframe_deletions_count_all_cancers = (
        count_same_or_nested_inframe_deletions_all_cancers(
            inframe_deletions_df=inframe_deletions_with_positions,
            patient_total=patient_total,
        )
    )

    inframe_deletions_count_per_cancer = (
        count_same_or_nested_inframe_deletions_per_cancer_type(
            inframe_deletions_df=inframe_deletions_with_positions,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )

    merged_inframe_deletions = pd.merge(
        inframe_deletions_count_all_cancers,
        inframe_deletions_count_per_cancer,
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    )

    inframe_with_positions_no_dups = (
        inframe_deletions_with_positions.drop_duplicates(
            subset="grch38_description", keep="first"
        )[["Hugo_Symbol", "grch38_description", "del_start", "del_end"]]
    )

    inframe_deletions_with_counts = pd.merge(
        inframe_with_positions_no_dups,
        merged_inframe_deletions,
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    )

    merged_inframe_deletion_counts = pd.merge(
        merged_frameshift_counts,
        inframe_deletions_with_counts,
        on=["grch38_description", "Hugo_Symbol"],
        how="left",
    )

    final_df = reorder_final_columns(
        merged_inframe_deletion_counts,
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
