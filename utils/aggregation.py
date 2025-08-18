import pandas as pd


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
