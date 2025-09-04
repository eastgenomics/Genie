import pandas as pd


def multi_merge(base_df, merge_dfs, on, how="left"):
    """
    Merge multiple DataFrames into a base DataFrame on specified keys.

    Parameters
    ----------
    base_df : pd.DataFrame
        The base DataFrame to merge into.
    merge_dfs : list of pd.DataFrame
        List of DataFrames to merge with the base DataFrame.
    on : str or list of str
        Column(s) to merge on.
    how : str, optional
        Type of merge to perform, default is 'left'.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    merged = base_df.copy()
    for df in merge_dfs:
        merged = pd.merge(merged, df, on=on, how=how)
    return merged


def merge_truncating_variants_counts(
    merged_amino_acid_counts: pd.DataFrame,
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
    truncating_variants_no_dups = truncating_variants.drop_duplicates(
        subset="grch38_description", keep="first"
    )[["Hugo_Symbol", "grch38_description", "CDS_position"]]

    merged_counts = multi_merge(
        truncating_variants_no_dups,
        [truncating_counts_all_cancers, truncating_counts_per_cancer],
        on=["Hugo_Symbol", "CDS_position"],
        how="left",
    )

    merged = pd.merge(
        merged_amino_acid_counts,
        merged_counts,
        on=["Hugo_Symbol", "grch38_description"],
        how="left",
    )

    return merged


def merge_inframe_deletions_with_counts(
    merged_frameshift_counts: pd.DataFrame,
    inframe_deletions_with_positions: pd.DataFrame,
    inframe_deletions_count_all_cancers: pd.DataFrame,
    inframe_deletions_count_per_cancer: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge inframe deletions with their counts across all cancers and per
    cancer type.

    Parameters
    ----------
    merged_frameshift_counts : pd.DataFrame
        DataFrame with each variant and all counts so far
    inframe_deletions_with_positions : pd.DataFrame
        DataFrame with inframe deletions and their positions
    inframe_deletions_count_all_cancers : pd.DataFrame
        DataFrame with inframe deletions counts across all cancer types
    inframe_deletions_count_per_cancer : pd.DataFrame
        DataFrame with inframe deletions counts per cancer type

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with inframe deletion counts
    """
    merged_counts = multi_merge(
        inframe_deletions_count_all_cancers,
        [inframe_deletions_count_per_cancer],
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    )

    # Remove duplicates of the same variant and deletion start and ends
    inframe_with_positions_no_dups = (
        inframe_deletions_with_positions.drop_duplicates(
            subset="grch38_description", keep="first"
        )[["Hugo_Symbol", "grch38_description", "del_start", "del_end"]]
    )

    # Merge positions and counts
    inframe_deletions_with_counts = pd.merge(
        inframe_with_positions_no_dups,
        merged_counts,
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    )

    merged = pd.merge(
        merged_frameshift_counts,
        inframe_deletions_with_counts,
        on=["grch38_description", "Hugo_Symbol"],
        how="left",
    )

    return merged


def reorder_final_columns(
    df, patient_total, per_cancer_patient_total, haemonc_patient_total=None
):
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
    haemonc_patient_total : int, optional
        Total number of unique patients with haemonc cancers
    Returns
    -------
    pd.DataFrame
        DataFrame with columns reordered to match the expected output format
    """
    unwanted_prefixes = ["CDS_position", "level", "del_start", "del_end"]
    df = df[
        [
            col
            for col in df.columns
            if not any(col.startswith(p) for p in unwanted_prefixes)
        ]
    ]
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
        "Variant_Classification",
        "Variant_Type",
    ]
    count_cols = [
        f"SameNucleotideChange.Total_Count_N_{patient_total}",
        f"SameAminoAcidChange.Total_Count_N_{patient_total}",
        f"SameOrDownstreamTruncatingVariantsPerCDS.Total_Count_N_{patient_total}",
        f"NestedInframeDeletionsPerCDS.Total_Count_N_{patient_total}",
    ]
    if haemonc_patient_total is not None:
        count_cols += [
            f"SameNucleotideChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}",
            f"SameAminoAcidChange.Haemonc_Cancers_Count_N_{haemonc_patient_total}",
            f"SameOrDownstreamTruncatingVariantsPerCDS.Haemonc_Cancers_Count_N_{haemonc_patient_total}",
            f"NestedInframeDeletionsPerCDS.Haemonc_Cancers_Count_N_{haemonc_patient_total}",
        ]
    # Add in cancer type columns in cancer order
    for cancer_type in per_cancer_patient_total.keys():
        count_cols.extend(
            [
                f"SameNucleotideChange.{cancer_type}_Count_N_{per_cancer_patient_total[cancer_type]}",
                f"SameAminoAcidChange.{cancer_type}_Count_N_{per_cancer_patient_total[cancer_type]}",
                f"SameOrDownstreamTruncatingVariantsPerCDS.{cancer_type}_Count_N_{per_cancer_patient_total[cancer_type]}",
                f"NestedInframeDeletionsPerCDS.{cancer_type}_Count_N_{per_cancer_patient_total[cancer_type]}",
            ]
        )

    count_cols = [col for col in count_cols if col in df.columns]
    # Reorder the DataFrame columns
    other_cols = [
        col
        for col in df.columns
        if not (
            col.startswith("SameNucleotideChange.")
            or col.startswith("SameAminoAcidChange.")
            or col.startswith("SameOrDownstreamTruncatingVariantsPerCDS.")
            or col.startswith("NestedInframeDeletionsPerCDS.")
        )
        and (col not in first_cols)
    ]
    final_col_order = first_cols + other_cols + count_cols
    reordered_df = df[[col for col in final_col_order if col in df.columns]]

    reordered_df = reordered_df.sort_values(
        by=["Hugo_Symbol", "grch38_description"]
    ).reset_index(drop=True)

    return reordered_df
