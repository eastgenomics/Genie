import pandas as pd


def calculate_unique_patient_counts(df: pd.DataFrame):
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
    return df[df["CANCER_TYPE"].isin(haemonc_cancers)].copy()


def calculate_unique_patients_haemonc_cancers(
    haemonc_rows: pd.DataFrame,
):
    """
    Count unique patients in haemonc cancer types.

    Parameters
    ----------
    haemonc_rows : pd.DataFrame
        Input dataframe with rows relevant to haemonc_cancers

    Returns
    -------
    int
        Total number of unique patients in haemonc cancer types
    """

    return haemonc_rows["PATIENT_ID"].nunique()


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


def deduplicate_variant_by_patient(df: pd.DataFrame):
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


def deduplicate_variant_by_patient_and_cancer_type(df: pd.DataFrame):
    """
    Remove multiple instances of the same variant (GRCh38 description) for the
    same patient and cancer type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns PATIENT_ID, grch38_description,
        CANCER_TYPE

    Returns
    -------
    pd.DataFrame
        DataFrame with unique variants per patient, variant description and
        cancer type
    """
    df_deduplicated_by_pt_and_cancer = df.drop_duplicates(
        subset=["PATIENT_ID", "grch38_description", "CANCER_TYPE"],
        keep="first",
    )

    return df_deduplicated_by_pt_and_cancer


def deduplicate_variant_by_patient_haemonc_cancers(haemonc_rows: pd.DataFrame):
    """
    Remove multiple instances of the same variant (GRCh38 description) for the
    same patient in haemonc cancer types.

    Parameters
    ----------
    haemonc_rows : pd.DataFrame
        Input dataframe with rows for haemonc cancer types

    Returns
    -------
    pd.DataFrame
        DataFrame with unique variants per patient and variant description
        in haemonc cancer types
    """
    df_deduplicated_by_patient_haemonc = haemonc_rows.drop_duplicates(
        subset=["PATIENT_ID", "grch38_description"], keep="first"
    )

    return df_deduplicated_by_patient_haemonc
