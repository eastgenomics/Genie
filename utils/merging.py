from functools import reduce

import polars as pl


def multi_merge(
    base_df: pl.DataFrame, merge_dfs: list, on: str, how="left"
) -> pl.DataFrame:
    """
    Merge multiple Polars DataFrames into a base DataFrame on specified keys.

    Parameters
    ----------
    base_df : pl.DataFrame
        The base Polars DataFrame to merge into.
    merge_dfs : list of pl.DataFrame
        List of DataFrames to merge with the base DataFrame.
    on : str or list of str
        Column(s) to merge on.
    how : str, optional
        Type of merge to perform, default is 'left'.

    Returns
    -------
    pl.DataFrame
        Merged DataFrame.
    """
    merged = base_df
    for df in merge_dfs:
        merged = merged.join(df, on=on, how=how)
    return merged


def merge_truncating_variants_counts(
    truncating_variants: pl.DataFrame,
    truncating_counts_all_cancers: pl.DataFrame,
    truncating_counts_per_cancer: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge the truncating variant counts across all cancers and per cancer type.

    Parameters
    ----------
    truncating_variants : pl.DataFrame
        DataFrame with truncating variants and their positions
    truncating_counts_all_cancers : pl.DataFrame
        DataFrame with truncating variant counts across all cancer types
    truncating_counts_per_cancer : pl.DataFrame
        DataFrame with truncating variant counts per cancer type

    Returns
    -------
    pl.DataFrame
        Merged DataFrame with truncating variant counts
    """
    # Get only unique truncating variants to add counts
    truncating_variants_no_dups = truncating_variants.unique(
        subset="grch38_description", keep="first"
    ).select(
        [
            "Hugo_Symbol",
            "grch38_description",
            "RefSeq",
            "Protein_position_start",
        ]
    )

    # Merge the truncating variants with the counts
    merged_counts = multi_merge(
        truncating_variants_no_dups,
        [truncating_counts_all_cancers, truncating_counts_per_cancer],
        on=["Hugo_Symbol", "RefSeq", "Protein_position_start"],
        how="left",
    )

    # Remove these columns to not cause merge issues later
    merged_counts = merged_counts.drop(
        ["Hugo_Symbol", "RefSeq", "Protein_position_start"]
    )

    return merged_counts


def merge_inframe_deletions_with_counts(
    inframe_deletions_with_positions: pl.DataFrame,
    inframe_deletions_count_all_cancers: pl.DataFrame,
    inframe_deletions_count_per_cancer: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge inframe deletions with their counts across all cancers and per
    cancer type.

    Parameters
    ----------
    inframe_deletions_with_positions : pl.DataFrame
        DataFrame with inframe deletions and their positions
    inframe_deletions_count_all_cancers : pl.DataFrame
        DataFrame with inframe deletions counts across all cancer types
    inframe_deletions_count_per_cancer : pl.DataFrame
        DataFrame with inframe deletions counts per cancer type

    Returns
    -------
    pl.DataFrame
        Merged DataFrame with inframe deletion counts
    """
    # Remove duplicates of the same variant and deletion start and ends
    inframe_with_positions_no_dups = inframe_deletions_with_positions.unique(
        subset="grch38_description", keep="first"
    ).select(
        [
            "Hugo_Symbol",
            "RefSeq",
            "grch38_description",
            "del_start",
            "del_end",
        ]
    )
    dfs_to_merge = [
        inframe_deletions_count_all_cancers,
        inframe_deletions_count_per_cancer,
    ]
    # Merge the inframe deletion counts together
    merged_counts = reduce(
        lambda left, right: left.join(
            right,
            on=["Hugo_Symbol", "RefSeq", "del_start", "del_end"],
            how="left",
        ),
        dfs_to_merge,
    )

    # Merge the unique inframe deletions with the counts
    inframe_deletions_with_counts = inframe_with_positions_no_dups.join(
        merged_counts,
        on=["Hugo_Symbol", "RefSeq", "del_start", "del_end"],
        how="left",
    ).drop(["Hugo_Symbol", "RefSeq", "del_start", "del_end"])

    return inframe_deletions_with_counts


def reorder_final_columns(
    df: pl.DataFrame,
    patient_total: int,
    per_cancer_patient_total: dict,
    haemonc_patient_total: int = None,
    solid_patient_total: int = None,
) -> pl.DataFrame:
    """
    Reorder the final DataFrame columns to match the expected output format,
    keeping the original structure and adding two extra columns per count type:
    <CountType>.Duplicate_Patient_Count and <CountType>.Duplicate_Patient_IDs.
    """

    # Drop unwanted columns
    unwanted_prefixes = ["level", "del_start", "del_end"]
    cols_to_keep = [
        col
        for col in df.columns
        if not any(col.startswith(p) for p in unwanted_prefixes)
    ]
    df = df.select(cols_to_keep)

    # First columns
    first_cols = [
        "Hugo_Symbol",
        "Entrez_Gene_Id",
        "grch38_description",
        "grch37_norm",
        "Genie_description",
        "RefSeq",
        "Consequence",
        "HGVSc",
        "HGVSp",
        "Variant_Classification",
        "Variant_Type",
    ]

    # Base count types
    count_types = [
        "SameNucleotideChange",
        "SameAminoAcidChange",
        "SameOrDownstreamTruncatingVariantsPerAA",
        "NestedInframeDeletionsPerAA",
    ]

    # Build original count columns
    count_cols = []

    # All cancers
    for ct in count_types:
        col_name = f"{ct}.All_Cancers_Count_N_{patient_total}"
        if col_name in df.columns:
            count_cols.append(col_name)

    # Haemonc cancers
    if haemonc_patient_total is not None:
        for ct in count_types:
            col_name = f"{ct}.Haemonc_Cancers_Count_N_{haemonc_patient_total}"
            if col_name in df.columns:
                count_cols.append(col_name)

    # Solid cancers
    if solid_patient_total is not None:
        for ct in count_types:
            col_name = f"{ct}.Solid_Cancers_Count_N_{solid_patient_total}"
            if col_name in df.columns:
                count_cols.append(col_name)

    # Per cancer type
    for cancer_type, n_patients in per_cancer_patient_total.items():
        for ct in count_types:
            col_name = f"{ct}.{cancer_type}_Count_N_{n_patients}"
            if col_name in df.columns:
                count_cols.append(col_name)

    # Build duplicate columns immediately after each original count type
    duplicate_cols = []
    for ct in count_types:
        duplicate_cols.append(f"{ct}.Duplicate_Patient_Count")
        duplicate_cols.append(f"{ct}.Duplicate_Patient_IDs")

    # Other columns
    other_cols = [
        col
        for col in df.columns
        if col not in first_cols
        and col not in count_cols
        and col not in duplicate_cols
    ]

    # Final column order
    final_col_order = first_cols + other_cols + count_cols + duplicate_cols
    df = df.select([col for col in final_col_order if col in df.columns])

    # Sort by Hugo_Symbol and grch38_description
    df = df.sort(["Hugo_Symbol", "grch38_description"])

    return df
