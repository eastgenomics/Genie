import pandas as pd
import polars as pl


def calculate_unique_patient_counts(
    df: pd.DataFrame, haemonc_cancers: list = None
):
    """
    Calculate the number of unique patients overall, per cancer type and
    (optionally) for haemonc cancer types.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns PATIENT_ID, CANCER_TYPE
    haemonc_cancers : list, optional
        List of haemonc cancer types to filter by. If provided, will calculate
        unique patient counts for these cancer types only.

    Returns
    -------
    total_patient_n: int
        Total number of unique patients overall
    unique_patient_n_per_cancer: dict
        Dictionary with cancer types as keys and number of unique patients
        as values
    haemonc_patient_n : int or None
        Total number of unique patients in haemonc cancer types, if applicable
    """
    total_patient_n = df["PATIENT_ID"].nunique()

    unique_patient_n_per_cancer = (
        df.groupby("CANCER_TYPE")["PATIENT_ID"].nunique().to_dict()
    )

    haemonc_patient_n = None
    if haemonc_cancers is not None:
        haemonc_patient_n = df[df["CANCER_TYPE"].isin(haemonc_cancers)][
            "PATIENT_ID"
        ].nunique()

    return total_patient_n, unique_patient_n_per_cancer, haemonc_patient_n


def calculate_unique_patient_counts_polars(
    df: pl.DataFrame, haemonc_cancers: list = None
):
    """
    Calculate the number of unique patients overall, per cancer type and
    optionally for haemonc cancer types.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with columns PATIENT_ID, CANCER_TYPE
    haemonc_cancers : list, optional
        List of haemonc cancer types to filter by.

    Returns
    -------
    total_patient_n: int
        Total number of unique patients overall
    unique_patient_n_per_cancer: dict
        Dictionary with cancer types as keys and number of unique patients
        as values
    haemonc_patient_n : int or None
        Total number of unique patients in haemonc cancer types, if applicable
    """

    # Total unique patients
    total_patient_n = df.select(pl.col("PATIENT_ID").n_unique()).item()

    # Unique patients per cancer type
    unique_patient_n_per_cancer = (
        df.group_by("CANCER_TYPE")
        .agg(pl.col("PATIENT_ID").n_unique().alias("unique_patients"))
        .to_dict(as_series=False)
    )
    # Convert to {cancer_type: count}
    unique_patient_n_per_cancer = dict(
        zip(
            unique_patient_n_per_cancer["CANCER_TYPE"],
            unique_patient_n_per_cancer["unique_patients"],
        )
    )

    # Haemonc patient counts (optional)
    haemonc_patient_n = None
    if haemonc_cancers is not None:
        haemonc_patient_n = (
            df.filter(pl.col("CANCER_TYPE").is_in(haemonc_cancers))
            .select(pl.col("PATIENT_ID").n_unique())
            .item()
        )

    return total_patient_n, unique_patient_n_per_cancer, haemonc_patient_n


def create_df_with_one_row_per_variant_pd(
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


def create_df_with_one_row_per_variant(
    df: pl.DataFrame, columns_to_aggregate: list
) -> pl.DataFrame:
    """
    Create a DataFrame with one row per unique variant by aggregating
    specified columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with variant information
    columns_to_aggregate : list
        List of columns to aggregate per variant

    Returns
    -------
    pl.DataFrame
        DataFrame with one row per unique variant, with aggregated fields
        joined by '&'
    """
    aggregated_df = df.group_by("grch38_description").agg(
        [
            pl.col(c).unique().sort().str.join("&").alias(c)
            for c in columns_to_aggregate
        ]
    )

    return aggregated_df


def get_truncating_variants_pd(df: pd.DataFrame) -> pd.DataFrame:
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


def get_truncating_variants(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract rows of truncating variants from the Genie data.

    Parameters
    ----------
    df : pl.DataFrame
        Input Genie MAF data

    Returns
    -------
    pl.DataFrame
        DataFrame with truncating variants
    """
    truncating = df.filter(
        (
            pl.col("Variant_Classification").is_in(
                [
                    "Frame_Shift_Del",
                    "Frame_Shift_Ins",
                    "Nonsense_Mutation",
                ]
            )
        )
        & (pl.col("HGVSp").str.contains("Ter", literal=True, strict=False))
    )
    return truncating


def get_inframe_deletions_pd(df: pd.DataFrame) -> pd.DataFrame:
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


def get_inframe_deletions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get inframe deletions from the Polars DataFrame

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the Genie data

    Returns
    -------
    pl.DataFrame
        DataFrame with inframe deletion variants
    """
    inframe_deletions = df.filter(
        pl.col("Variant_Classification") == "In_Frame_Del"
    ).filter(pl.col("HGVSc").is_not_null())
    return inframe_deletions


def get_haemonc_cancer_rows(df: pd.DataFrame, haemonc_cancers: list):
    """
    Get rows for haemonc cancer types from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns CANCER_TYPE
    haemonc_cancers : list
        List of haemonc cancer types to filter by

    Returns
    -------
    pd.DataFrame
        DataFrame with rows for haemonc cancer types
    """
    return df.filter(pl.col("CANCER_TYPE").is_in(haemonc_cancers))
