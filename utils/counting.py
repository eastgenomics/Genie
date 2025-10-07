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
        pl.col("PATIENT_ID").n_unique().cast(pl.Int64).alias("patient_count")
    )

    # Pivot so all cancer types are columns
    all_cancer_counts = per_cancer_counts.pivot(
        values="patient_count",
        index="grch38_description",
        on="CANCER_TYPE",
        aggregate_function="first",
    )

    # Fill missing cancer count values with 0
    all_cancer_counts = all_cancer_counts.fill_null(0)

    # Rename columns to include patient N per cancer type
    present = [
        c for c in all_cancer_counts.columns if c not in ("grch38_description")
    ]
    missing = set(present) - set(unique_patients_per_cancer)
    if missing:
        raise ValueError(
            f"Missing patient totals for cancer types: {sorted(missing)}"
        )
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
        .group_by(["Hugo_Symbol", "Transcript_ID", "HGVSp"])
        .agg(pl.col("PATIENT_ID").n_unique().cast(pl.Int64).alias(count_col))
    )

    # If this is a grouped (e.g. haemonc cancer) count, then all variants
    #  with HGVSp should have a count, so add 0 if not present in grouped count
    if all_variants_df is not None:
        result = (
            all_variants_df.filter(pl.col("HGVSp").is_not_null())
            .select(
                ["grch38_description", "Hugo_Symbol", "Transcript_ID", "HGVSp"]
            )
            .unique()
            .join(
                amino_acid_change_counts,
                on=["Hugo_Symbol", "Transcript_ID", "HGVSp"],
                how="left",
            )
            .with_columns(pl.col(count_col).fill_null(0))
        ).drop(["Hugo_Symbol", "Transcript_ID", "HGVSp"])
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
        .group_by(["Hugo_Symbol", "HGVSp", "Transcript_ID", "CANCER_TYPE"])
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
        index=["Hugo_Symbol", "HGVSp", "Transcript_ID"],
        on="CANCER_TYPE",
        aggregate_function="first",
    ).fill_null(0)

    # Rename columns to include patient N per cancer type
    present = [
        c
        for c in aa_per_cancer_counts.columns
        if c not in ["Hugo_Symbol", "HGVSp", "Transcript_ID"]
    ]
    missing = set(present) - set(unique_patients_per_cancer)
    if missing:
        raise ValueError(
            f"Missing patient totals for cancer types: {sorted(missing)}"
        )
    new_columns = []
    for col in aa_per_cancer_counts.columns:
        if col in ["Hugo_Symbol", "HGVSp", "Transcript_ID"]:
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
        DataFrame containing truncating variants with 'Hugo_Symbol',
        'Transcript_ID', and 'CDS_position'.
    cancer_count_type : str
        Cancer count type (used in column naming)
    patient_total : int
        Total number of unique patients in the dataset.

    Returns
    -------
    pl.DataFrame
        DataFrame with 'CDS_position' and downstream counts per gene.
    """
    all_results = []

    # Iterate over unique (gene, transcript) pairs
    for gene, transcript in (
        df.select(["Hugo_Symbol", "Transcript_ID"]).unique().iter_rows()
    ):
        subset = df.filter(
            (pl.col("Hugo_Symbol") == gene)
            & (pl.col("Transcript_ID") == transcript)
        )
        positions = sorted(subset["CDS_position"].unique().to_list())

        rows = []
        for pos in positions:
            downstream_patients = {
                pid
                for j, pid in enumerate(subset["PATIENT_ID"])
                if subset["CDS_position"][j] >= pos
            }
            rows.append(
                {
                    "Hugo_Symbol": gene,
                    "Transcript_ID": transcript,
                    "CDS_position": pos,
                    "downstream_patient_count": len(downstream_patients),
                }
            )

        all_results.append(pl.DataFrame(rows))

    # Combine all results
    df_counts = (
        pl.concat(all_results, how="vertical")
        if all_results
        else pl.DataFrame(
            {
                "Hugo_Symbol": [],
                "Transcript_ID": [],
                "CDS_position": [],
                "downstream_patient_count": [],
            }
        )
    )

    # If this is a grouped (e.g. haemonc cancer) count, then all truncating
    #  variants should have a count, so add 0 if not present in grouped count
    col_name = f"SameOrDownstreamTruncatingVariantsPerCDS.{cancer_count_type}_Count_N_{patient_total}"
    df_counts = df_counts.rename({"downstream_patient_count": col_name})

    # If given, join back to truncating_variants to ensure all rows are present
    if truncating_variants is not None:
        result = (
            truncating_variants.select(
                [
                    "grch38_description",
                    "Hugo_Symbol",
                    "Transcript_ID",
                    "CDS_position",
                ]
            )
            .unique()
            .join(
                df_counts,
                on=["Hugo_Symbol", "Transcript_ID", "CDS_position"],
                how="left",
            )
            .with_columns(pl.col(col_name).fill_null(0))
        ).drop("Hugo_Symbol", "Transcript_ID", "CDS_position")
        return result

    return df_counts


def count_frameshift_truncating_and_nonsense_per_cancer_type(
    df: pl.DataFrame,
    per_cancer_patient_total: dict,
):
    """
    Count how many patients have a frameshift (truncating) or nonsense variant
    at the same position or downstream in the same gene per cancer type.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing truncating variants with 'Hugo_Symbol',
        'Transcript_ID', 'CDS_position', 'CANCER_TYPE', and 'PATIENT_ID'.
    per_cancer_patient_total : dict
        Total number of unique patients in the dataset per cancer type.

    Returns
    -------
    pl.DataFrame
        DataFrame with CDS_position and downstream counts per gene per cancer type.
    """
    all_results = []

    # Iterate over unique (gene, transcript) pairs
    for gene, transcript in (
        df.select(["Hugo_Symbol", "Transcript_ID"]).unique().iter_rows()
    ):
        gene_tx_df = df.filter(
            (pl.col("Hugo_Symbol") == gene)
            & (pl.col("Transcript_ID") == transcript)
        )
        gene_positions = sorted(gene_tx_df["CDS_position"].unique().to_list())

        # Iterate over unique cancer types
        for cancer in per_cancer_patient_total.keys():
            subset = gene_tx_df.filter(pl.col("CANCER_TYPE") == cancer)

            # Map CDS_position to downstream patient counts
            rows = []
            for pos in gene_positions:
                downstream_patients = {
                    pid
                    for j, pid in enumerate(subset["PATIENT_ID"])
                    if subset["CDS_position"][j] >= pos
                }
                rows.append(
                    {
                        "CDS_position": pos,
                        "downstream_patient_count": len(downstream_patients),
                    }
                )

            result_df = pl.DataFrame(rows).with_columns(
                [
                    pl.lit(gene).alias("Hugo_Symbol"),
                    pl.lit(transcript).alias("Transcript_ID"),
                    pl.lit(cancer).alias("CANCER_TYPE"),
                ]
            )
            all_results.append(result_df)

    # Combine all results
    combined = (
        pl.concat(all_results, how="vertical")
        if all_results
        else pl.DataFrame(
            {
                "Hugo_Symbol": [],
                "Transcript_ID": [],
                "CANCER_TYPE": [],
                "CDS_position": [],
                "downstream_patient_count": [],
            }
        )
    )

    # Pivot so each cancer type becomes a column
    pivoted = combined.pivot(
        values="downstream_patient_count",
        index=["Hugo_Symbol", "Transcript_ID", "CDS_position"],
        on="CANCER_TYPE",
        aggregate_function="first",
    ).fill_null(0)

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


def add_deletion_positions(
    inframe_deletions: pl.DataFrame,
    source: str,
) -> pl.DataFrame:
    """
    Add start and end positions as new columns to the inframe deletions
    DataFrame based on HGVSp column.

    Examples
    --------
    HGVSc:
    'ENST00000269305.4:c.480_485del'    -> (480, 485)
    'ENST00000296930.5:c.511_524+1del'  -> (511, 524)
    'ENST00000269305.4:c.480del'        -> (480, 480)
    HGVSp:
    'p.Met160_Ala161del'        -> (160, 161)
    'p.Glu453del'               -> (453, 453)
    'p.Met237_Cys242delinsIle'  -> (237, 242)
    'p.Trp557_Val559delinsCys'  -> (557, 559)

    Parameters
    ----------
    inframe_deletions : pl.DataFrame
        DataFrame containing inframe deletions and the positions of the deletion
    source: str
        Source of the data, either 'HGVSc' or 'HGVSp'

    Returns
    -------
    pl.DataFrame
        DataFrame with additional columns for deletion start and end
    """
    if source == "HGVSc":
        pattern = r"c\.(\d+)[+-]?\d*_?(\d+)?[+-]?\d*del"
    elif source == "HGVSp":
        pattern = (
            r"p\.[A-Za-z]{3}(\d+)(?:_[A-Za-z]{3}(\d+))?del(?:ins[A-Za-z]+)?"
        )
    else:
        raise ValueError("source must be 'HGVSc' or 'HGVSp'")

    df = inframe_deletions.with_columns(
        [
            pl.col(source)
            .str.extract(pattern, 1)
            .cast(pl.Int64)
            .alias("del_start"),
            pl.col(source)
            .str.extract(pattern, 2)
            .cast(pl.Int64)
            .alias("del_end"),
        ]
    )

    # Fill missing del_end with del_start for single-position deletions
    df = df.with_columns(
        pl.when(pl.col("del_end").is_null())
        .then(pl.col("del_start"))
        .otherwise(pl.col("del_end"))
        .alias("del_end")
    )

    return df


def count_nested_inframe_deletions(
    inframe_deletions_df: pl.DataFrame,
    cancer_count_type: str,
    patient_total: int,
    position_method: str,
    inframe_deletions: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Count the number of unique patients with inframe deletions that are either
    the same as or nested within the current deletion for all cancers.

    Parameters
    ----------
    inframe_deletions_df : pl.DataFrame
        DataFrame containing a set of inframe deletions
    cancer_count_type : str
        The cancer count type, to include in name of the column
    patient_total : int
        Total number of unique patients in the dataset
    inframe_deletions: pl.DataFrame | None
        Reference dataset to ensure all inframe deletions are included
    position_method : str
        Method used to determine deletion positions, "CDS" or "AA"

    Returns
    -------
    pl.DataFrame
        DataFrame with counts of matching or nested inframe deletions.
    """
    all_results = []

    # Iterate over (gene, transcript) pairs
    for gene, transcript in (
        inframe_deletions_df.select(["Hugo_Symbol", "Transcript_ID"])
        .unique()
        .iter_rows()
    ):
        subset = inframe_deletions_df.filter(
            (pl.col("Hugo_Symbol") == gene)
            & (pl.col("Transcript_ID") == transcript)
        )

        unique_ranges = (
            subset.select(
                ["Hugo_Symbol", "Transcript_ID", "del_start", "del_end"]
            )
            .unique()
            .sort(["del_start", "del_end"])
        )

        # Get nested patient counts for each unique deletion range
        rows = []
        for del_start, del_end in unique_ranges.select(
            ["del_start", "del_end"]
        ).iter_rows():
            nested_patients = (
                subset.filter(
                    (pl.col("del_start") >= del_start)
                    & (pl.col("del_end") <= del_end)
                )
                .select("PATIENT_ID")
                .unique()
                .to_series()
                .to_list()
            )
            rows.append(
                {
                    "Hugo_Symbol": gene,
                    "Transcript_ID": transcript,
                    "del_start": del_start,
                    "del_end": del_end,
                    "nested_patient_count": len(nested_patients),
                }
            )

        all_results.append(pl.DataFrame(rows))

    # Combine all results
    inframe_counts = (
        pl.concat(all_results, how="vertical")
        if all_results
        else pl.DataFrame(
            {
                "Hugo_Symbol": [],
                "Transcript_ID": [],
                "del_start": [],
                "del_end": [],
                "nested_patient_count": [],
            }
        )
    )

    # Rename nested count column with cohort info
    col_name = f"NestedInframeDeletionsPer{position_method}.{cancer_count_type}_Count_N_{patient_total}"
    inframe_counts = inframe_counts.rename({"nested_patient_count": col_name})

    # If given, join back to reference deletions to ensure all rows are present
    if inframe_deletions is not None:
        result = (
            inframe_deletions.select(
                [
                    "Hugo_Symbol",
                    "grch38_description",
                    "Transcript_ID",
                    "del_start",
                    "del_end",
                ]
            )
            .unique()
            .join(
                inframe_counts,
                on=["Hugo_Symbol", "Transcript_ID", "del_start", "del_end"],
                how="left",
            )
            .with_columns(pl.col(col_name).fill_null(0))
        ).drop("Hugo_Symbol", "Transcript_ID", "del_start", "del_end")

        return result

    return inframe_counts


def count_nested_inframe_deletions_per_cancer_type(
    inframe_deletions_df: pl.DataFrame,
    per_cancer_patient_total: dict,
    position_method: str,
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
    position_method : str
        Method used to determine deletion positions, "CDS" or "AA".

    Returns
    -------
    pl.DataFrame
        DataFrame with counts of matching or nested inframe deletions per cancer type.
    """
    all_results = []

    # Iterate over unique (gene, transcript) pairs
    for gene, transcript in (
        inframe_deletions_df.select(["Hugo_Symbol", "Transcript_ID"])
        .unique()
        .iter_rows()
    ):
        subset_gene_tx = inframe_deletions_df.filter(
            (pl.col("Hugo_Symbol") == gene)
            & (pl.col("Transcript_ID") == transcript)
        )

        # Get unique deletion ranges for this transcript
        unique_ranges = (
            subset_gene_tx.select(["del_start", "del_end"])
            .unique()
            .sort(["del_start", "del_end"])
        )

        # Iterate over cancer types
        for cancer in per_cancer_patient_total.keys():
            subset_cancer = subset_gene_tx.filter(
                pl.col("CANCER_TYPE") == cancer
            )

            rows = []
            for del_start, del_end in unique_ranges.iter_rows():
                nested_patients = (
                    subset_cancer.filter(
                        (pl.col("del_start") >= del_start)
                        & (pl.col("del_end") <= del_end)
                    )
                    .select("PATIENT_ID")
                    .unique()
                    .to_series()
                    .to_list()
                )
                rows.append(
                    {
                        "Hugo_Symbol": gene,
                        "Transcript_ID": transcript,
                        "CANCER_TYPE": cancer,
                        "del_start": del_start,
                        "del_end": del_end,
                        "nested_patient_count": len(nested_patients),
                    }
                )

            if rows:
                all_results.append(pl.DataFrame(rows))

    # Combine all results
    nested_counts_df = (
        pl.concat(all_results, how="vertical")
        if all_results
        else pl.DataFrame(
            {
                "Hugo_Symbol": [],
                "Transcript_ID": [],
                "del_start": [],
                "del_end": [],
                "CANCER_TYPE": [],
                "nested_patient_count": [],
            }
        )
    )

    # Pivot cancer types into columns
    pivot_df = nested_counts_df.pivot(
        index=["Hugo_Symbol", "Transcript_ID", "del_start", "del_end"],
        on="CANCER_TYPE",
        values="nested_patient_count",
        aggregate_function="first",
    ).fill_null(0)

    # Rename columns to include patient totals
    column_mapping = {
        col: (
            f"NestedInframeDeletionsPer{position_method}.{col}_Count_N_{per_cancer_patient_total[col]}"
        )
        for col in per_cancer_patient_total.keys()
        if col in pivot_df.columns
    }
    if column_mapping:
        pivot_df = pivot_df.rename(column_mapping)

    return pivot_df
