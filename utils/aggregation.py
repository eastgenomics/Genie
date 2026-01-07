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
            pl.when(pl.col(c).drop_nulls().len() == 0)
            .then(None)
            .otherwise(pl.col(c).drop_nulls().unique().sort().str.join("&"))
            .alias(c)
            for c in columns_to_aggregate
        ]
    )

    return aggregated_df


def get_truncating_variants(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract rows of truncating variants from the GENIE data. This takes
    any which have "Ter" (but not "ext") in the HGVSp notation and have
    a Consequence which includes "stop_gained" or "frameshift_variant".

    Parameters
    ----------
    df : pl.DataFrame
        Input GENIE MAF data

    Returns
    -------
    pl.DataFrame
        DataFrame with truncating variants
    """
    truncating = df.filter(
        pl.col("HGVSp").str.contains("Ter", literal=False)
        & ~pl.col("HGVSp").str.contains("ext", literal=False)
        & pl.col("Consequence").str.contains(
            "stop_gained|frameshift_variant", literal=False
        )
    ).select(
        "Hugo_Symbol",
        "grch38_description",
        "RefSeq",
        "Protein_position",
        "PATIENT_ID",
        "CANCER_TYPE",
    )

    return truncating


def add_protein_position_start(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract the first number from the protein position, as some are formatted
    e.g. 221-222 and we are using the first position to determine other
    variants which are downstream.

    Parameters
    ----------
    df : pl.DataFrame
        dataframe with Protein_position to extract start position from

    Returns
    -------
    pl.DataFrame
        dataframe with new column Protein_position_start
    """
    truncating_variants = df.with_columns(
        pl.col("Protein_position")
        .cast(pl.Utf8)
        .str.extract(r"^(\d+)", 1)
        .cast(pl.Int64)
        .alias("Protein_position_start")
    )

    return truncating_variants


def get_inframe_deletions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract rows of inframe deletion variants from the GENIE data. This takes
    any which have a Consequence which includes "inframe_deletion".

    Parameters
    ----------
    df : pl.DataFrame
        GENIE dataset

    Returns
    -------
    pl.DataFrame
        dataframe of inframe deletions
    """
    columns_to_select = [
        "grch38_description",
        "Hugo_Symbol",
        "RefSeq",
        "Protein_position",
        "PATIENT_ID",
        "CANCER_TYPE",
    ]

    inframe_deletions = df.filter(
        pl.col("Consequence").str.contains("inframe_deletion")
        & pl.col("Protein_position").is_not_null()
    ).select(columns_to_select)

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
