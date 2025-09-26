import polars as pl


def calculate_unique_patient_counts(
    df: pl.DataFrame, haemonc_cancers: list = None, solid_cancers: list = None
):
    """
    Calculate the number of unique patients overall, per cancer type and
    optionally for grouped (haemonc or solid) cancer types.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with columns PATIENT_ID, CANCER_TYPE
    haemonc_cancers : list, optional
        List of haemonc cancer types to filter by
    solid_cancers : list, optional
        List of solid cancer types to filter by

    Returns
    -------
    total_patient_n: int
        Total number of unique patients overall
    unique_patient_n_per_cancer: dict
        Dictionary with cancer types as keys and number of unique patients
        as values
    haemonc_patient_n : int or None
        Total number of unique patients in haemonc cancer types, if applicable
    solid_patient_n : int or None
        Total number of unique patients in solid cancer types, if applicable
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

    haemonc_patient_n = None
    if haemonc_cancers is not None:
        haemonc_patient_n = (
            df.filter(pl.col("CANCER_TYPE").is_in(haemonc_cancers))
            .select(pl.col("PATIENT_ID").n_unique())
            .item()
        )

    solid_patient_n = None
    if solid_cancers is not None:
        solid_patient_n = (
            df.filter(pl.col("CANCER_TYPE").is_in(solid_cancers))
            .select(pl.col("PATIENT_ID").n_unique())
            .item()
        )

    return (
        total_patient_n,
        unique_patient_n_per_cancer,
        haemonc_patient_n,
        solid_patient_n,
    )


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
    ).select(
        "Hugo_Symbol",
        "grch38_description",
        "Transcript_ID",
        "HGVSc",
        "PATIENT_ID",
        "CANCER_TYPE",
    )
    return truncating


def get_inframe_deletions(df: pl.DataFrame, column_used: str) -> pl.DataFrame:
    """
    Get inframe deletions from the Polars DataFrame

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the Genie data
    column_used : str
        Column to check for non-null values, either 'HGVSc' or 'HGVSp'

    Returns
    -------
    pl.DataFrame
        DataFrame with inframe deletion variants
    """
    inframe_deletions = (
        df.filter(pl.col("Variant_Classification") == "In_Frame_Del").filter(
            pl.col(column_used).is_not_null()
        )
    ).select(
        "grch38_description",
        "Hugo_Symbol",
        "Transcript_ID",
        column_used,
        "PATIENT_ID",
        "CANCER_TYPE",
    )
    return inframe_deletions


def get_rows_for_cancer_types(df: pl.DataFrame, cancer_types: list):
    """
    Get rows for cancer types listed from the DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe with columns CANCER_TYPE
    cancer_types : list
        List of cancer types to filter by

    Returns
    -------
    pl.DataFrame
        DataFrame with rows for specified cancer types
    """
    return df.filter(pl.col("CANCER_TYPE").is_in(cancer_types))
