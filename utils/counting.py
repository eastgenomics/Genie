import pandas as pd
import polars as pl
import re


def count_same_nucleotide_change(
    df: pl.DataFrame,
    unique_patient_total: int,
    count_type: str,
    all_variants_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Count how many patients have the same exact variant across all cancers (Polars version).

    Parameters
    ----------
    df : pl.DataFrame
        Input Genie MAF data
    unique_patient_total : int
        Total number of unique patients for the count type
    count_type: str
        Type of count being performed (e.g. "all_cancers", "per_cancer")
    all_variants_df : pl.DataFrame, optional
        Reference dataset to ensure all variants included (used for grouped
        counts like haemonc or solid cancer)

    Returns
    -------
    pl.DataFrame
        DataFrame with nucleotide change counts across all cancer types
    """
    count_col = (
        f"SameNucleotideChange.{count_type}_Count_N_{unique_patient_total}"
    )

    # Group by variant and count unique patients per variant
    nucleotide_change_counts = df.group_by("grch38_description").agg(
        pl.col("PATIENT_ID").n_unique().cast(pl.Int64).alias(count_col)
    )

    # If this is a grouped (e.g. haemonc cancer) count, then we want all
    # variants to have a count, so add 0 if not present
    if all_variants_df is not None:
        all_variants = all_variants_df.select("grch38_description").unique()

        result = (
            all_variants.join(
                nucleotide_change_counts, on="grch38_description", how="left"
            )
            .with_columns(pl.col(count_col).fill_null(0))
            .with_columns(pl.col(count_col).cast(pl.Int64))
        )
        return result

    return nucleotide_change_counts


def count_same_nucleotide_change_per_cancer_type(
    df: pl.DataFrame, unique_patients_per_cancer: dict
) -> pl.DataFrame:
    """
    Count how many patients have the exact variant per cancer type using Polars.

    Parameters
    ----------
    df : pl.DataFrame
        Input Genie MAF data
    unique_patients_per_cancer : dict
        Total number of unique patients in the dataset per cancer type

    Returns
    -------
    pl.DataFrame
        DataFrame with grch38_description and nucleotide change counts
    """
    # Group by variant and cancer type and count unique patients
    per_cancer_counts = df.group_by(["grch38_description", "CANCER_TYPE"]).agg(
        pl.col("PATIENT_ID").n_unique().alias("patient_count")
    )

    # Pivot so all cancer types are columns
    all_cancer_counts = per_cancer_counts.pivot(
        values="patient_count",
        index="grch38_description",
        columns="CANCER_TYPE",
        aggregate_function="first",
    )

    # Fill missing values with 0
    all_cancer_counts = all_cancer_counts.fill_null(0)

    # Rename columns to include patient N per cancer type
    new_columns = [
        (
            col
            if col == "grch38_description"
            else f"SameNucleotideChange.{col}_Count_N_{unique_patients_per_cancer[col]}"
        )
        for col in all_cancer_counts.columns
    ]
    all_cancer_counts.columns = new_columns

    return all_cancer_counts


def count_amino_acid_change(
    df: pl.DataFrame,
    unique_patient_total: int,
    count_type: str,
    all_variants_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Count how many patients have the same amino acid change in that gene
    across all cancers.

    Parameters
    ----------
    df : pl.DataFrame
        Input Genie MAF data
    unique_patient_total : int
        Total number of unique patients in the dataset
    count_type : str
        Type of count being performed
    all_variants_df : pl.DataFrame | None
        Reference dataset to ensure all variants are included

    Returns
    -------
    pl.DataFrame
        DataFrame with amino acid change counts across all cancer types
    """
    count_col = (
        f"SameAminoAcidChange.{count_type}_Count_N_{unique_patient_total}"
    )

    # Group by gene + amino acid change and count unique patients
    amino_acid_change_counts = (
        df.filter(pl.col("HGVSp").is_not_null())
        .group_by(["Hugo_Symbol", "HGVSp"])
        .agg(pl.col("PATIENT_ID").n_unique().cast(pl.Int64).alias(count_col))
    )

    if all_variants_df is not None:
        result = (
            all_variants_df.select(["Hugo_Symbol", "HGVSp"])
            .unique()
            .join(
                amino_acid_change_counts,
                on=["Hugo_Symbol", "HGVSp"],
                how="left",
            )
            .with_columns(pl.col(count_col).fill_null(0))
        )
        return result

    return amino_acid_change_counts


def count_amino_acid_change_per_cancer_type(
    df: pl.DataFrame, unique_patients_per_cancer: dict
):
    """
    Count how many patients have the same amino acid change in that gene
    per cancer type.

    Parameters
    ----------
    df : pl.DataFrame
        Input Genie MAF data
    unique_patient_total : int
        Total number of unique patients in the dataset
    unique_patients_per_cancer : dict
        Total number of unique patients in the dataset per cancer type

    Returns
    -------
    pl.DataFrame
        DataFrame with amino acid change counts per cancer
    """

    # Count how many patients have same amino acid change for each cancer type
    amino_acid_count_per_present_cancer = (
        df.filter(pl.col("HGVSp").is_not_null())
        .group_by(["Hugo_Symbol", "HGVSp", "CANCER_TYPE"])
        .agg(
            pl.col("PATIENT_ID")
            .n_unique()
            .cast(pl.Int64)
            .alias("patient_count")
        )
    )

    # Pivot so all cancer types have counts
    aa_per_cancer_counts = amino_acid_count_per_present_cancer.pivot(
        values="patient_count",
        index=["Hugo_Symbol", "HGVSp"],
        columns="CANCER_TYPE",
        aggregate_function="first",
    ).fill_null(0)

    # Rename columns to include patient N per cancer type
    new_columns = []
    for col in aa_per_cancer_counts.columns:
        if col in ["Hugo_Symbol", "HGVSp"]:
            new_columns.append(col)
        else:
            new_columns.append(
                f"SameAminoAcidChange.{col}_Count_N_{unique_patients_per_cancer[col]}"
            )

    aa_per_cancer_counts.columns = new_columns

    return aa_per_cancer_counts


def extract_position_from_cds(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract the position affected from the HGVSc string for frameshift
    (truncating) and nonsense variants and add new columns.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the HGVSc column

    Returns
    -------
    pl.DataFrame
        DataFrame with an additional column 'CDS_position'
        containing the first aa affected
    """
    return df.with_columns(
        pl.col("HGVSc")
        .str.extract(r"(?:.*:)?c\.?-?(\d+)")
        .cast(pl.Int64)
        .alias("CDS_position")
    )


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


def count_frameshift_truncating_and_nonsense_pd(
    df: pd.DataFrame,
    cancer_count_type: str,
    patient_total: int,
) -> pd.DataFrame:
    """
    Count how many patients have a frameshift (truncating) or nonsense variant
    at the same position or downstream in the same gene.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing truncating variants with position
    cancer_count_type : str
        the cancer count type, to include in name of the column
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
        .reset_index(level=1, drop=True)
        .reset_index()
    )

    # Rename column to add patient N
    result = result.rename(
        columns={
            "downstream_patient_count": (
                f"SameOrDownstreamTruncatingVariantsPerCDS.{cancer_count_type}_Count_N_{patient_total}"
            )
        }
    )

    return result


def count_downstream(group: pl.DataFrame) -> pl.DataFrame:
    """
    Count how many unique patients have a variant at the same position or
    downstream for rows in the same gene.

    Parameters
    ----------
    group : pl.DataFrame
        DataFrame containing variants in the same gene

    Returns
    -------
    pl.DataFrame
        DataFrame with CDS_position and downstream_patient_count
    """
    gene = group[0, "Hugo_Symbol"]

    # Get unique positions, sorted
    positions = sorted(set(group["CDS_position"].to_list()))
    patient_ids = group["PATIENT_ID"].to_list()

    rows = []
    for pos in positions:
        downstream = {
            pid
            for j, pid in enumerate(patient_ids)
            if group["CDS_position"][j] >= pos
        }
        rows.append(
            {
                "Hugo_Symbol": gene,
                "CDS_position": pos,
                "downstream_patient_count": len(downstream),
            }
        )

    return pl.DataFrame(rows)


def count_frameshift_truncating_and_nonsense(
    df: pl.DataFrame,
    cancer_count_type: str,
    patient_total: int,
) -> pl.DataFrame:
    """
    Count unique patients with frameshift or nonsense variants at the same
    CDS_position or downstream for each gene, using Polars.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing truncating variants with 'Hugo_Symbol' and 'CDS_position'.
    cancer_count_type : str
        Cancer count type (used in column naming)
    patient_total : int
        Total number of unique patients in the dataset.

    Returns
    -------
    pl.DataFrame
        DataFrame with 'CDS_position' and downstream counts per gene.
    """
    # Compute downstream counts using cumsum of unique patients
    df_counts = df.group_by("Hugo_Symbol").map_groups(count_downstream)

    col_name = f"SameOrDownstreamTruncatingVariantsPerCDS.{cancer_count_type}_Count_N_{patient_total}"
    df_counts = df_counts.rename({"downstream_patient_count": col_name})

    return df_counts


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

    count_cols = per_cancer_pivot.columns.difference(
        ["Hugo_Symbol", "CDS_position"]
    )
    per_cancer_pivot[count_cols] = per_cancer_pivot[count_cols].astype(int)
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


# def count_frameshift_truncating_and_nonsense_per_cancer_type_polars(
#     df: pl.DataFrame,
#     per_cancer_patient_total: dict,
# ):
#     """
#     Count how many patients have a frameshift (truncating) or nonsense variant
#     at the same position or downstream in the same gene per cancer type.
#     """

#     # Apply per gene and cancer type (this requires rewriting your custom function in Polars!)
#     result = df.group_by(["Hugo_Symbol", "CANCER_TYPE"]).map_batches(
#         count_downstream_pl
#     )

#     # # Ensure Polars DataFrame
#     # result = pl.DataFrame(result)

#     # Get all unique gene + CDS position pairs
#     full_index = (
#         df.group_by("Hugo_Symbol")
#         .agg(pl.col("CDS_position").unique())
#         .explode("CDS_position")
#     )

#     # Create all combinations with cancer types (cross join)
#     all_cancers = pl.DataFrame(
#         {"CANCER_TYPE": list(per_cancer_patient_total.keys())}
#     )
#     full_index = full_index.join(all_cancers, how="cross")

#     # Merge actual counts with complete index and fill missing
#     result_filled = full_index.join(
#         result, on=["Hugo_Symbol", "CDS_position", "CANCER_TYPE"], how="left"
#     ).with_columns(
#         pl.col("downstream_patient_count").fill_null(0).cast(pl.Int64)
#     )

#     # Pivot so each row = gene/CDS, each col = cancer type
#     per_cancer_pivot = result_filled.pivot(
#         values="downstream_patient_count",
#         index=["Hugo_Symbol", "CDS_position"],
#         columns="CANCER_TYPE",
#     ).fill_null(0)

#     # Rename columns to include patient N
#     rename_map = {}
#     for col in per_cancer_pivot.columns:
#         if col in per_cancer_patient_total:
#             rename_map[col] = (
#                 "SameOrDownstreamTruncatingVariantsPerCDS."
#                 f"{col}_Count_N_{per_cancer_patient_total[col]}"
#             )
#     per_cancer_pivot = per_cancer_pivot.rename(rename_map)

#     return per_cancer_pivot


def add_deletion_positions(inframe_deletions: pl.DataFrame) -> pl.DataFrame:
    """
    Add start and end positions as new columns to the inframe deletions
    DataFrame.

    Examples
    --------
    'ENST00000269305.4:c.480_485del'    -> (480, 485)
    'ENST00000296930.5:c.511_524+1del'  -> (511, 524)
    'ENST00000269305.4:c.480del'        -> (480, 480)

    Parameters
    ----------
    inframe_deletions : pl.DataFrame
        DataFrame containing inframe deletions with HGVSc column

    Returns
    -------
    pl.DataFrame
        DataFrame with additional columns for deletion start and end
    """
    return inframe_deletions.with_columns(
        [
            # Extract start position
            pl.col("HGVSc")
            .str.extract(r"c\.(\d+)[+-]?\d*_?(\d+)?[+-]?\d*del", group_index=1)
            .cast(pl.Int64)
            .alias("del_start"),
            # Extract end position, defaulting to start if missing
            pl.col("HGVSc")
            .str.extract(r"c\.(\d+)[+-]?\d*_?(\d+)?[+-]?\d*del", group_index=2)
            .cast(pl.Int64)
            .alias("del_end"),
        ]
    ).with_columns(
        # Fill missing del_end with del_start
        pl.when(pl.col("del_end").is_null())
        .then(pl.col("del_start"))
        .otherwise(pl.col("del_end"))
        .alias("del_end")
    )


def count_patients_with_nested_deletions_pd(gene_group):
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


def count_patients_with_nested_deletions(
    gene_group: pl.DataFrame,
) -> pl.DataFrame:
    """
    Count unique patients with deletions that are nested or equal in the same gene.
    """
    # Keep only unique deletion ranges
    unique_ranges = (
        gene_group.select(["del_start", "del_end"])
        .unique()
        .sort(["del_start", "del_end"])
    )

    results = []

    # For each unique range, find all patients with deletions nested within it
    for row in unique_ranges.iter_rows(named=True):
        ref_start = row["del_start"]
        ref_end = row["del_end"]

        # Find all deletions nested within this range
        nested_patients = (
            gene_group.filter(
                (pl.col("del_start") >= ref_start)
                & (pl.col("del_end") <= ref_end)
            )
            .select("PATIENT_ID")
            .unique()
            .height
        )

        results.append(
            {
                "del_start": ref_start,
                "del_end": ref_end,
                "nested_patient_count": nested_patients,
            }
        )

    return pl.DataFrame(results)


# def count_nested_inframe_deletions_pd(
#     inframe_deletions_df: pd.DataFrame,
#     cancer_count_type: str,
#     patient_total: int,
# ) -> pd.DataFrame:
#     """
#     Count the number of unique patients with inframe deletions that are either
#     the same as or nested within the current deletion for all cancers.

#     Parameters
#     ----------
#     inframe_deletions_df : pd.DataFrame
#         DataFrame containing inframe deletions
#     cancer_count_type : str
#         The cancer count type, to include in name of the column
#     patient_total : int
#         Total number of unique patients in the dataset

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with counts of matching or nested inframe deletions.
#     """
#     inframe_counts = (
#         inframe_deletions_df.groupby("Hugo_Symbol")
#         .apply(count_patients_with_nested_deletions_pd)
#         .reset_index(level=1, drop=True)
#         .reset_index()
#     ).rename(
#         columns={
#             "nested_patient_count": (
#                 f"NestedInframeDeletionsPerCDS.{cancer_count_type}_Count_N_{patient_total}"
#             )
#         }
#     )

#     return inframe_counts


def count_nested_inframe_deletions(
    inframe_deletions_df: pl.DataFrame,
    cancer_count_type: str,
    patient_total: int,
) -> pl.DataFrame:
    """
    Count the number of unique patients with inframe deletions that are either
    the same as or nested within the current deletion for all cancers.

    Parameters
    ----------
    inframe_deletions_df : pl.DataFrame
        DataFrame containing inframe deletions
    cancer_count_type : str
        The cancer count type, to include in name of the column
    patient_total : int
        Total number of unique patients in the dataset

    Returns
    -------
    pl.DataFrame
        DataFrame with counts of matching or nested inframe deletions.
    """
    genes = inframe_deletions_df["Hugo_Symbol"].unique().to_list()

    results = []
    for gene in genes:
        gene_data = inframe_deletions_df.filter(pl.col("Hugo_Symbol") == gene)
        nested_counts_df = count_patients_with_nested_deletions(gene_data)

        nested_counts_df = nested_counts_df.with_columns(
            pl.lit(gene).alias("Hugo_Symbol")
        )

        results.append(nested_counts_df)

    # Concatenate all genes
    inframe_counts = pl.concat(results)

    # Rename the count column
    new_col_name = f"NestedInframeDeletionsPerCDS.{cancer_count_type}_Count_N_{patient_total}"
    inframe_counts = inframe_counts.rename(
        {"nested_patient_count": new_col_name}
    )

    return inframe_counts


def count_nested_inframe_deletions_per_cancer_type(
    inframe_deletions_df: pl.DataFrame, per_cancer_patient_total: dict
) -> pl.DataFrame:
    """
    Count the number of unique patients with inframe deletions that are either
    the same as or nested within the current deletion, grouped by cancer type.

    Parameters
    ----------
    inframe_deletions_df : pl.DataFrame
        DataFrame containing inframe deletions with patient information.
    per_cancer_patient_total : dict
        Total number of unique patients in the dataset per cancer type.

    Returns
    -------
    pl.DataFrame
        DataFrame with counts of matching or nested inframe deletions per cancer type.
    """
    gene_cancer_combinations = (
        inframe_deletions_df.select(["Hugo_Symbol", "CANCER_TYPE"])
        .unique()
        .sort(["Hugo_Symbol", "CANCER_TYPE"])
    )

    # Apply the function to each gene-cancer combination
    all_results = []

    for row in gene_cancer_combinations.iter_rows(named=True):
        hugo_symbol = row["Hugo_Symbol"]
        cancer_type = row["CANCER_TYPE"]

        # Filter data for this specific gene-cancer combination
        gene_cancer_subset = inframe_deletions_df.filter(
            (pl.col("Hugo_Symbol") == hugo_symbol)
            & (pl.col("CANCER_TYPE") == cancer_type)
        )

        # Apply the nested deletions counting function
        nested_counts = count_patients_with_nested_deletions(
            gene_cancer_subset
        )

        # Add back the grouping columns
        nested_counts = nested_counts.with_columns(
            [
                pl.lit(hugo_symbol).alias("Hugo_Symbol"),
                pl.lit(cancer_type).alias("CANCER_TYPE"),
            ]
        )

        all_results.append(nested_counts)

    # Combine all results
    if all_results:
        nested_per_cancer_counts = pl.concat(all_results, how="vertical")
    else:
        # Handle empty case
        nested_per_cancer_counts = pl.DataFrame(
            {
                "Hugo_Symbol": [],
                "CANCER_TYPE": [],
                "del_start": [],
                "del_end": [],
                "nested_patient_count": [],
            },
            schema={
                "Hugo_Symbol": pl.Utf8,
                "CANCER_TYPE": pl.Utf8,
                "del_start": pl.Int64,
                "del_end": pl.Int64,
                "nested_patient_count": pl.Int64,
            },
        )

    # Step 2: Get unique combinations of genes and deletion positions
    full_index = inframe_deletions_df.select(
        ["Hugo_Symbol", "del_start", "del_end"]
    ).unique()

    # Step 3: Create cross product with all cancer types
    all_cancers = list(per_cancer_patient_total.keys())
    cancer_df = pl.DataFrame({"CANCER_TYPE": all_cancers})

    # Cross join using a temporary key column
    full_index = (
        full_index.with_columns(pl.lit(1).alias("key"))
        .join(
            cancer_df.with_columns(pl.lit(1).alias("key")),
            on="key",
            how="inner",
        )
        .drop("key")
    )

    # Step 4: Left join to fill missing combinations with zeros
    nested_counts_filled = full_index.join(
        nested_per_cancer_counts,
        on=["Hugo_Symbol", "CANCER_TYPE", "del_start", "del_end"],
        how="left",
    ).with_columns(
        [pl.col("nested_patient_count").fill_null(0).cast(pl.Int64)]
    )

    # Step 5: Pivot table - each row is gene/CDS position, each column is cancer type
    per_cancer_pivot = nested_counts_filled.pivot(
        index=["Hugo_Symbol", "del_start", "del_end"],
        columns="CANCER_TYPE",
        values="nested_patient_count",
        aggregate_function="first",  # Should be unique combinations
    ).fill_null(
        0
    )  # Fill any remaining nulls with 0

    # Step 6: Rename columns to include patient totals
    column_mapping = {}
    for col in per_cancer_pivot.columns:
        if col in per_cancer_patient_total:
            new_name = f"NestedInframeDeletionsPerCDS.{col}_Count_N_{per_cancer_patient_total[col]}"
            column_mapping[col] = new_name

    if column_mapping:
        per_cancer_pivot = per_cancer_pivot.rename(column_mapping)

    return per_cancer_pivot


def count_nested_inframe_deletions_per_cancer_type_pd(
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

    # Convert all count columns to int
    count_cols = per_cancer_pivot.columns.difference(
        ["Hugo_Symbol", "del_start", "del_end"]
    )
    per_cancer_pivot[count_cols] = per_cancer_pivot[count_cols].astype(int)

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
