import polars as pl


def count_same_nucleotide_change(
    df: pl.DataFrame,
    unique_patient_total: int,
    count_type: str,
    all_variants_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Count how many patients have the same exact variant across all cancers.

    Parameters
    ----------
    df : pl.DataFrame
        Input Genie MAF data
    unique_patient_total : int
        Total number of unique patients for the count type
    count_type: str
        Type of count being performed (e.g. "All_Cancers", "Haemonc_Cancers")
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

    # If this is a grouped (e.g. haemonc cancer) count, all variants should
    # have a count -> add 0 if var not present in the grouped count
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
    Count how many patients have the exact variant per cancer type.

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

    # Fill missing cancer count values with 0
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

    # If this is a grouped (e.g. haemonc cancer) count, then all variants
    #  with HGVSp should have a count, so add 0 if not present in grouped count
    if all_variants_df is not None:
        result = (
            all_variants_df.filter(pl.col("HGVSp").is_not_null())
            .select("grch38_description", "Hugo_Symbol", "HGVSp")
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
        on="CANCER_TYPE",
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


def extract_position_from_hgvsc(df: pl.DataFrame) -> pl.DataFrame:
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
        containing the position affected
    """
    return df.with_columns(
        pl.col("HGVSc")
        .str.extract(r"(?:.*:)?c\.?-?(\d+)")
        .cast(pl.Int64)
        .alias("CDS_position")
    )


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
    truncating_variants: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Count unique patients with frameshift or nonsense variants at the same
    CDS_position or downstream for each gene.

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

    # If this is a grouped (e.g. haemonc cancer) count, then all truncating
    #  variants should have a count, so add 0 if not present in grouped count
    if truncating_variants is not None:
        result = (
            truncating_variants.select(
                "grch38_description", "Hugo_Symbol", "CDS_position"
            )
            .unique()
            .join(
                df_counts,
                on=["Hugo_Symbol", "CDS_position"],
                how="left",
            )
            .with_columns(pl.col(col_name).fill_null(0))
        )
        return result

    return df_counts


def count_frameshift_truncating_and_nonsense_per_cancer_type(
    df: pl.DataFrame,
    per_cancer_patient_total: dict,
):
    """
    Count how many patients have a frameshift (truncating) or nonsense variant
    at the same position or downstream in the same gene per cancer type.
    """
    gene_cancer_combinations = (
        df.select(["Hugo_Symbol", "CANCER_TYPE"])
        .unique()
        .sort(["Hugo_Symbol", "CANCER_TYPE"])
    )

    all_results = []

    # Iterate over each gene-cancer combination
    for row in gene_cancer_combinations.iter_rows(named=True):
        gene = row["Hugo_Symbol"]
        cancer = row["CANCER_TYPE"]

        subset = df.filter(
            (pl.col("Hugo_Symbol") == gene) & (pl.col("CANCER_TYPE") == cancer)
        )

        # Count downstream patients for each CDS position
        positions = sorted(subset["CDS_position"].to_list())
        patient_ids = subset["PATIENT_ID"].to_list()
        rows = []
        for pos in positions:
            downstream_patients = {
                pid
                for j, pid in enumerate(patient_ids)
                if subset["CDS_position"][j] >= pos
            }
            rows.append(
                {
                    "CDS_position": pos,
                    "downstream_patient_count": len(downstream_patients),
                }
            )
        if rows:
            result_df = pl.DataFrame(rows).with_columns(
                [
                    pl.lit(gene).alias("Hugo_Symbol"),
                    pl.lit(cancer).alias("CANCER_TYPE"),
                ]
            )
            all_results.append(result_df)

    # Combine all results
    if all_results:
        combined = pl.concat(all_results, how="vertical")
    else:
        combined = pl.DataFrame(
            {
                "Hugo_Symbol": [],
                "CANCER_TYPE": [],
                "CDS_position": [],
                "downstream_patient_count": [],
            }
        )

    # Build full cross join for missing combinations
    full_index = df.select(["Hugo_Symbol", "CDS_position"]).unique()
    all_cancers = pl.DataFrame(
        {"CANCER_TYPE": list(per_cancer_patient_total.keys())}
    )
    full_index = full_index.with_columns(pl.lit(1).alias("key"))
    all_cancers = all_cancers.with_columns(pl.lit(1).alias("key"))
    full_index = full_index.join(all_cancers, on="key", how="inner").drop(
        "key"
    )

    # Left join counts onto full index and fill missing with 0
    filled = full_index.join(
        combined, on=["Hugo_Symbol", "CANCER_TYPE", "CDS_position"], how="left"
    )
    filled = filled.with_columns(
        pl.col("downstream_patient_count").fill_null(0).cast(pl.Int32)
    )

    # Pivot so each cancer type becomes a column
    pivoted = filled.pivot(
        values="downstream_patient_count",
        index=["Hugo_Symbol", "CDS_position"],
        on="CANCER_TYPE",
        aggregate_function="first",
    ).fill_null(0)

    for col in pivoted.columns:
        if col not in ["Hugo_Symbol", "CDS_position"]:
            pivoted = pivoted.with_columns(pl.col(col).cast(pl.Int64))

    # Rename columns to include patient totals
    new_column_names = {}
    for col in pivoted.columns:
        if col in per_cancer_patient_total:
            new_column_names[col] = (
                f"SameOrDownstreamTruncatingVariantsPerCDS.{col}_Count_N_{per_cancer_patient_total[col]}"
            )
    if new_column_names:
        pivoted = pivoted.rename(new_column_names)

    return pivoted


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
            pl.col("HGVSc")
            .str.extract(r"c\.(\d+)[+-]?\d*_?(\d+)?[+-]?\d*del", group_index=1)
            .cast(pl.Int64)
            .alias("del_start"),
            pl.col("HGVSc")
            .str.extract(r"c\.(\d+)[+-]?\d*_?(\d+)?[+-]?\d*del", group_index=2)
            .cast(pl.Int64)
            .alias("del_end"),
        ]
    ).with_columns(
        # Fill missing del_end with del_start (if it's deletion at one pos)
        pl.when(pl.col("del_end").is_null())
        .then(pl.col("del_start"))
        .otherwise(pl.col("del_end"))
        .alias("del_end")
    )


def count_patients_with_nested_deletions(
    gene_group: pl.DataFrame,
) -> pl.DataFrame:
    """
    Count how many unique patients have deletions that are the same positions
    or nested within the deletion in the same gene.

    Parameters
    ----------
    gene_group : pl.DataFrame
        DataFrame containing deletions for a single gene (and optionally a cancer type)

    Returns
    -------
    pl.DataFrame
        DataFrame with del_start, del_end, and nested_patient_count
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


def count_nested_inframe_deletions(
    inframe_deletions_df: pl.DataFrame,
    cancer_count_type: str,
    patient_total: int,
    inframe_deletions: pl.DataFrame | None = None,
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
    # If this is a grouped (e.g. haemonc cancer) count, then all in frame dels
    #  should have a count, so add 0 if not present in grouped count
    if inframe_deletions is not None:
        result = (
            inframe_deletions.select(
                "Hugo_Symbol", "grch38_description", "del_start", "del_end"
            )
            .unique()
            .join(
                inframe_counts,
                on=["Hugo_Symbol", "del_start", "del_end"],
                how="left",
            )
            .with_columns(pl.col(new_col_name).fill_null(0))
        )
        return result

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

    # Get unique combinations of genes and deletion positions
    full_index = inframe_deletions_df.select(
        ["Hugo_Symbol", "del_start", "del_end"]
    ).unique()

    # Create cross product with all cancer types
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

    # Left join to fill missing combinations with zeros
    nested_counts_filled = full_index.join(
        nested_per_cancer_counts,
        on=["Hugo_Symbol", "CANCER_TYPE", "del_start", "del_end"],
        how="left",
    ).with_columns(
        [pl.col("nested_patient_count").fill_null(0).cast(pl.Int64)]
    )

    # Pivot table - each row is gene/CDS position, each column is cancer type
    per_cancer_pivot = nested_counts_filled.pivot(
        index=["Hugo_Symbol", "del_start", "del_end"],
        on="CANCER_TYPE",
        values="nested_patient_count",
        aggregate_function="first",  # Should be unique combinations
    ).fill_null(
        0
    )  # Fill any remaining nulls with 0

    # Rename columns to include patient totals
    column_mapping = {}
    for col in per_cancer_pivot.columns:
        if col in per_cancer_patient_total:
            new_name = f"NestedInframeDeletionsPerCDS.{col}_Count_N_{per_cancer_patient_total[col]}"
            column_mapping[col] = new_name

    if column_mapping:
        per_cancer_pivot = per_cancer_pivot.rename(column_mapping)

    return per_cancer_pivot
