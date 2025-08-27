import pandas as pd
import re


def count_same_nucleotide_change_all_cancers(
    df: pd.DataFrame, unique_patient_total: int, count_type: str
) -> pd.DataFrame:
    """
    Count how many patients have the same exact variant across all cancers.

    Parameters
    ----------
    df : pd.DataFrame
        Input Genie MAF data
    unique_patient_total : int
        Total number of unique patients for the count type
    count_type: str
        Type of count being performed (e.g. "all_cancers", "per_cancer")

    Returns
    -------
    pd.DataFrame
        DataFrame with nucleotide change counts across all cancer types
    """
    # Group by variant, count patients, adding patient N to column name
    nucleotide_change_counts = (
        df.groupby("grch38_description")
        .agg({"PATIENT_ID": "nunique"})
        .rename(
            columns={
                "PATIENT_ID": (
                    f"SameNucleotideChange.{count_type}_Count_N_{unique_patient_total}"
                )
            }
        )
        .reset_index()
    )

    return nucleotide_change_counts


def count_same_nucleotide_change_per_cancer_type(
    df: pd.DataFrame,
    unique_patients_per_cancer: dict,
) -> pd.DataFrame:
    """
    Count how many patients have the exact variant per cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        Input Genie MAF data
    unique_patients_per_cancer : dict
        Total number of unique patients in the dataset per cancer type

    Returns
    -------
    pd.DataFrame
        DataFrame with grch38_description and nucleotide change counts
    """
    # Group by variant and cancer type and count patients
    per_cancer_counts = (
        df.groupby(["grch38_description", "CANCER_TYPE"])
        .agg(patient_count=("PATIENT_ID", "nunique"))
        .reset_index()
    )

    # Pivot so all cancer types are present, filling NAs with zero
    all_cancer_counts = per_cancer_counts.pivot_table(
        index="grch38_description",
        columns="CANCER_TYPE",
        values="patient_count",
        fill_value=0,
    ).reset_index()

    # Rename columns to include patient N for the cancer type count columns
    all_cancer_counts.columns = [
        (
            "grch38_description"
            if i == 0
            else f"SameNucleotideChange.{col}_Count_N_{unique_patients_per_cancer[col]}"
        )
        for i, col in enumerate(all_cancer_counts.columns)
    ]

    return all_cancer_counts


def count_nucleotide_change_haemonc_cancers(
    haemonc_data, genie_data, haemonc_patient_total
):
    """
    Count how many patients have the exact variant in haemonc cancers.

    Parameters
    ----------
    haemonc_data : pd.DataFrame
        Genie data in haemonc cancer types to calculate counts from
    genie_data : pd.DataFrame
        Full Genie data to ensure all variants are present
    haemonc_patient_total : int
        Total number of unique patients in haemonc cancer types

    Returns
    -------
    pd.DataFrame
        Count data for each variant in haemonc cancers
    """
    # Count unique patients per variant
    counts = (
        haemonc_data.groupby("grch38_description")["PATIENT_ID"]
        .nunique()
        .rename(
            f"SameNucleotideChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}"
        )
        .reset_index()
    )

    # Ensure all variants are present, fill missing with zero
    all_variants = genie_data.drop_duplicates(subset="grch38_description")[
        "grch38_description"
    ]
    result = pd.merge(
        all_variants, counts, on="grch38_description", how="left"
    ).fillna(0)
    result[
        f"SameNucleotideChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}"
    ] = result[
        f"SameNucleotideChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}"
    ].astype(
        int
    )
    return result


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
        .agg({"PATIENT_ID": "nunique"})
        .rename(
            columns={
                "PATIENT_ID": (
                    f"SameAminoAcidChange.All_Cancers_Count_N_{unique_patient_total}"
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
    # Count how many patients have same amino acid change for each cancer type
    amino_acid_count_per_present_cancer = (
        df.groupby(["Hugo_Symbol", "HGVSp", "CANCER_TYPE"])
        .agg(patient_count=("PATIENT_ID", "nunique"))
        .reset_index()
    )

    # Pivot so all cancer types have counts, filling NAs as zero
    aa_per_cancer_counts = amino_acid_count_per_present_cancer.pivot_table(
        index=["Hugo_Symbol", "HGVSp"],
        columns="CANCER_TYPE",
        values="patient_count",
        fill_value=0,
    ).reset_index()

    # Rename columns to include patient N per cancer type
    aa_per_cancer_counts.columns = [
        (
            col
            if i < 2
            else f"SameAminoAcidChange.{col}_Count_N_{unique_patients_per_cancer[col]}"
        )
        for i, col in enumerate(aa_per_cancer_counts.columns)
    ]

    return aa_per_cancer_counts


def count_amino_acid_change_haemonc_cancers(
    haemonc_data: pd.DataFrame,
    genie_data: pd.DataFrame,
    haemonc_patient_total: int,
):
    """
    Count how many patients have the same amino acid change in that gene
    in haemonc cancers.

    Parameters
    ----------
    haemonc_data : pd.DataFrame
        Genie data in haemonc cancer types to calculate counts from
    genie_data : pd.DataFrame
        Full Genie data to ensure all variants are present
    haemonc_patient_total : int
        Total number of unique patients in haemonc cancer types

    Returns
    -------
    pd.DataFrame
        Count data for each variant in haemonc cancers
    """
    # Count unique patients per variant
    counts = (
        haemonc_data.groupby(["Hugo_Symbol", "HGVSp"])["PATIENT_ID"]
        .nunique()
        .rename(
            f"SameAminoAcidChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}"
        )
        .reset_index()
    )
    # Ensure all gene and HGVSp combinations are present, fill missing with zero
    all_aa_variants = genie_data.drop_duplicates(
        subset=["Hugo_Symbol", "HGVSp"]
    )[["Hugo_Symbol", "HGVSp"]]

    result = pd.merge(
        all_aa_variants, counts, on=["Hugo_Symbol", "HGVSp"], how="left"
    ).fillna(0)
    result[
        f"SameAminoAcidChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}"
    ] = result[
        f"SameAminoAcidChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}"
    ].astype(
        int
    )

    return result


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

    # Split HGVSc into transcript ID and c. string
    hgvsc_parts = hgvsc_value.split(":", 1)
    if len(hgvsc_parts) == 2:
        hgvsc_c = hgvsc_parts[1]
    else:
        hgvsc_c = hgvsc_value

    # Extract the position number
    match = re.search(r"-?\d+", hgvsc_c.replace("c.", ""))

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
    df["CDS_position"] = df["HGVSc"].apply(extract_position_from_cds)

    return df


def count_patients_with_variant_at_same_position_or_downstream(gene_group):
    """
    Count how many unique patients have a variant at the same position or
    downstream for rows in the same gene.

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
                f"SameOrDownstreamTruncatingVariantsPerCDS.All_Cancers_Count_N_{patient_total}"
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
    # Apply per gene and cancer type to get counts at each present position
    result = (
        df.groupby(["Hugo_Symbol", "CANCER_TYPE"])
        .apply(count_patients_with_variant_at_same_position_or_downstream)
        .reset_index()
    )

    # Get all unique gene + CDS position pairs
    full_index = (
        df.groupby("Hugo_Symbol")["CDS_position"]
        .unique()
        .explode()
        .reset_index()
        .rename(columns={0: "CDS_position"})
    )

    # Create all combinations with cancer types (cross join) and all gene
    # and CDS positions
    all_cancers = list(per_cancer_patient_total.keys())
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


def count_frameshift_truncating_and_nonsense_haemonc_cancers(
    df: pd.DataFrame,
    patient_total: int,
) -> pd.DataFrame:
    """
    Count how many patients have a frameshift (truncating) or nonsense variant
    at the same position or downstream in the same gene in haemonc cancers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing truncating variants with position
    patient_total : int
        Total number of unique patients in haemonc cancers.

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
                f"SameOrDownstreamTruncatingVariantsPerCDS.Haemonc_Cancers_Count_N_{patient_total}"
            )
        }
    )

    return result


def extract_hgvsc_deletion_positions(hgvsc):
    """
    Extract deletion coding start and end from an HGVSc string, ignoring
    intronic offsets.

    Examples
    --------
    'ENST00000269305.4:c.480_485del'    -> (480, 485)
    'ENST00000296930.5:c.511_524+1del'  -> (511, 524)
    'ENST00000269305.4:c.480del'        -> (480, 480)

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

    # Match deletion with range, capturing only base numbers (ignore intronic offsets)
    match = re.search(r"c\.(\d+)[+-]?\d*_?(\d+)?[+-]?\d*del", hgvsc)
    if match:
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
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
    Count how many unique patients have deletions that are the same positions
    or nested within the deletion in the same gene.

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

    # Get all unique deletion ranges in that gene
    unique_deletions = (
        gene_group[["del_start", "del_end"]]
        .drop_duplicates()
        .sort_values(["del_start", "del_end"])
    )

    # Loop over each unique deletion and count how many patients have deletions
    # that are the same positions or nested within it
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


def count_nested_inframe_deletions_all_cancers(
    inframe_deletions_df: pd.DataFrame, patient_total: int
) -> pd.DataFrame:
    """
    Count the number of unique patients with inframe deletions that are either
    the same as or nested within the current deletion for all cancers.

    Parameters
    ----------
    inframe_deletions_df : pd.DataFrame
        DataFrame containing inframe deletions
    patient_total : int
        Total number of unique patients in the dataset

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
                f"NestedInframeDeletionsPerCDS.All_Cancers_Count_N_{patient_total}"
            )
        }
    )

    return inframe_counts


def count_nested_inframe_deletions_per_cancer_type(
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

    # Get unique combinations of genes and deletion start and ends
    full_index = (
        result[["Hugo_Symbol", "del_start", "del_end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Get combinations of all cancer types with all genes and deletion start
    # and ends
    all_cancers = list(per_cancer_patient_total.keys())
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


def count_nested_inframe_deletions_haemonc_cancers(
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
                f"NestedInframeDeletionsPerCDS.Haemonc_Cancers_Count_N_{patient_total}"
            )
        }
    )

    return inframe_counts
