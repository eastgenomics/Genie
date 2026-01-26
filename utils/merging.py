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


def build_count_block(df, prefix, totals):
    cols = []
    for label, n in totals:
        col = f"{prefix}.{label}_Count_N_{n}"
        if col in df.columns:
            cols.append(col)
    return cols


def reorder_final_columns(
    df: pl.DataFrame,
    patient_total: int,
    per_cancer_patient_total: dict,
    haemonc_patient_total: int = None,
    solid_patient_total: int = None,
) -> pl.DataFrame:
    """
    Reorder the final DataFrame columns to match the expected output format.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the counts and variant information
    patient_total : int
        Total number of unique patients across all cancer types
    per_cancer_patient_total : dict
        Dictionary with cancer types as keys and number of unique patients
        as values
    haemonc_patient_total : int, optional
        Total number of unique patients with haemonc cancers
    solid_patient_total : int, optional
        Total number of unique patients with solid cancers

    Returns
    -------
    pl.DataFrame
        DataFrame with columns reordered to match the expected output format
    """
    # Drop unwanted columns
    unwanted_prefixes = ("level", "del_start", "del_end")
    df = df.select(
        [c for c in df.columns if not c.startswith(unwanted_prefixes)]
    )

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
    first_cols = [c for c in first_cols if c in df.columns]

    prefixes = [
        "SameNucleotideChange",
        "SameAminoAcidChange",
        "SameOrDownstreamTruncatingVariantsPerAA",
        "NestedInframeDeletionsPerAA",
    ]

    scopes = [("All_Cancers", patient_total)]

    if haemonc_patient_total is not None:
        scopes.append(("Haemonc_Cancers", haemonc_patient_total))

    if solid_patient_total is not None:
        scopes.append(("Solid_Cancers", solid_patient_total))

    for cancer, n in per_cancer_patient_total.items():
        scopes.append((cancer, n))

    ordered_count_cols = []

    for prefix in prefixes:
        # Count columns for this prefix
        prefix_count_cols = []
        for scope, n in scopes:
            col = f"{prefix}.{scope}_Count_N_{n}"
            if col in df.columns:
                prefix_count_cols.append(col)

        ordered_count_cols.extend(prefix_count_cols)

        # Duplicate columns (once per prefix, after last count)
        dup_count = f"{prefix}.Duplicate_Patient_Count"
        dup_ids = f"{prefix}.Duplicate_Patient_IDs"

        if dup_count in df.columns:
            ordered_count_cols.append(dup_count)
        if dup_ids in df.columns:
            ordered_count_cols.append(dup_ids)

    used_cols = set(first_cols) | set(ordered_count_cols)
    other_cols = [c for c in df.columns if c not in used_cols]

    final_col_order = first_cols + other_cols + ordered_count_cols
    df = df.select(final_col_order)

    if "Hugo_Symbol" in df.columns and "grch38_description" in df.columns:
        df = df.sort(["Hugo_Symbol", "grch38_description"])

    return df
