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
    ).select("Hugo_Symbol", "grch38_description", "CDS_position")

    # Merge the truncating variants with the counts
    merged_counts = multi_merge(
        truncating_variants_no_dups,
        [truncating_counts_all_cancers, truncating_counts_per_cancer],
        on=["Hugo_Symbol", "CDS_position"],
        how="left",
    )

    # Remove these columns to not cause merge issues later
    merged_counts = merged_counts.drop(["Hugo_Symbol", "CDS_position"])

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
    ).select(["Hugo_Symbol", "grch38_description", "del_start", "del_end"])

    # Merge the inframe deletion counts together
    merged_counts = multi_merge(
        inframe_deletions_count_all_cancers,
        [inframe_deletions_count_per_cancer],
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    )

    # Merge the unique inframe deletions with the counts
    inframe_deletions_with_counts = inframe_with_positions_no_dups.join(
        merged_counts,
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    ).drop(["Hugo_Symbol", "del_start", "del_end"])

    return inframe_deletions_with_counts


def merge_truncating_variant_counts_grouped_cancer(
    merged_aa_grouped_cancer_counts: pl.DataFrame,
    truncating_plus_position: pl.DataFrame,
    frameshift_counts_grouped_cancer: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge truncating variants with their counts for grouped cancers.

    Parameters
    ----------
    merged_aa_grouped_cancer_counts : pl.DataFrame
        DataFrame with amino acid counts for grouped cancers
    truncating_plus_position : pl.DataFrame
        DataFrame with truncating variants and their positions
    frameshift_counts_grouped_cancer : pl.DataFrame
        DataFrame with frameshift counts for grouped cancers
    Returns
    -------
    pl.DataFrame
        Merged DataFrame with truncating variant counts for grouped cancers
    """
    truncating_variants_no_dups = truncating_plus_position.unique(
        subset=["grch38_description"]
    ).select(["Hugo_Symbol", "grch38_description", "CDS_position"])

    # Merge with frameshift counts
    merged_frameshift_grouped_counts = truncating_variants_no_dups.join(
        frameshift_counts_grouped_cancer,
        on=["Hugo_Symbol", "CDS_position"],
        how="left",
    ).fill_null(0)

    # Merge with amino acid counts
    merged_frameshift_grouped = merged_aa_grouped_cancer_counts.join(
        merged_frameshift_grouped_counts,
        on=["Hugo_Symbol", "grch38_description"],
        how="left",
    )

    return merged_frameshift_grouped


def merge_inframe_deletions_grouped_cancer(
    inframe_deletions: pl.DataFrame,
    inframe_deletions_count_grouped_cancers: pl.DataFrame,
    merged_frameshift_grouped_cancers: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge inframe deletions with their counts for grouped cancers.

    Parameters
    ----------
    inframe_deletions : pl.DataFrame
        DataFrame with inframe deletions and their positions
    inframe_deletions_count_grouped_cancers : pl.DataFrame
        DataFrame with inframe deletions counts for grouped cancers
    merged_frameshift_grouped_cancers : pl.DataFrame
        DataFrame with frameshift counts for grouped cancers

    Returns
    -------
    pl.DataFrame
        Merged DataFrame with inframe deletion counts for grouped cancers
    """
    # Drop duplicates on grch38_description and keep needed columns
    inframe_deletions_no_dups = inframe_deletions.unique(
        subset=["grch38_description"]
    ).select(["Hugo_Symbol", "grch38_description", "del_start", "del_end"])

    # Merge with inframe deletion counts on Hugo_Symbol + del_start/del_end
    inframe_deletions_grouped_counts = inframe_deletions_no_dups.join(
        inframe_deletions_count_grouped_cancers,
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    ).fill_null(0)

    # Merge with frameshift/aa counts on Hugo_Symbol + grch38_description
    all_counts_merged = merged_frameshift_grouped_cancers.join(
        inframe_deletions_grouped_counts,
        on=["Hugo_Symbol", "grch38_description"],
        how="left",
    )

    return all_counts_merged


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
    unwanted_prefixes = ["CDS_position", "level", "del_start", "del_end"]
    cols_to_keep = [
        col
        for col in df.columns
        if not any(col.startswith(p) for p in unwanted_prefixes)
    ]
    df = df.select(cols_to_keep)

    # Set first columns
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
        "Variant_Classification",
        "Variant_Type",
    ]

    # Build count columns
    count_cols = [
        f"SameNucleotideChange.All_Cancers_Count_N_{patient_total}",
        f"SameAminoAcidChange.All_Cancers_Count_N_{patient_total}",
        f"SameOrDownstreamTruncatingVariantsPerCDS.All_Cancers_Count_N_{patient_total}",
        f"NestedInframeDeletionsPerCDS.All_Cancers_Count_N_{patient_total}",
    ]

    if haemonc_patient_total is not None:
        count_cols += [
            f"SameNucleotideChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}",
            f"SameAminoAcidChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}",
            f"SameOrDownstreamTruncatingVariantsPerCDS.Haemonc_Cancers_Count_N_{haemonc_patient_total}",
            f"NestedInframeDeletionsPerCDS.Haemonc_Cancers_Count_N_{haemonc_patient_total}",
        ]

    if solid_patient_total is not None:
        count_cols += [
            f"SameNucleotideChange.Solid_Cancers_Count_N_{solid_patient_total}",
            f"SameAminoAcidChange.Solid_Cancers_Count_N_{solid_patient_total}",
            f"SameOrDownstreamTruncatingVariantsPerCDS.Solid_Cancers_Count_N_{solid_patient_total}",
            f"NestedInframeDeletionsPerCDS.Solid_Cancers_Count_N_{solid_patient_total}",
        ]

    for cancer_type, n_patients in per_cancer_patient_total.items():
        count_cols.extend(
            [
                f"SameNucleotideChange.{cancer_type}_Count_N_{n_patients}",
                f"SameAminoAcidChange.{cancer_type}_Count_N_{n_patients}",
                f"SameOrDownstreamTruncatingVariantsPerCDS.{cancer_type}_Count_N_{n_patients}",
                f"NestedInframeDeletionsPerCDS.{cancer_type}_Count_N_{n_patients}",
            ]
        )

    # Keep only columns that exist in df
    count_cols = [col for col in count_cols if col in df.columns]

    # Other columns
    other_cols = [
        col
        for col in df.columns
        if col not in first_cols and col not in count_cols
    ]

    # Final column order
    final_col_order = first_cols + other_cols + count_cols
    df = df.select([col for col in final_col_order if col in df.columns])

    # Sort by Hugo_Symbol and grch38_description
    df = df.sort(["Hugo_Symbol", "grch38_description"])

    return df
