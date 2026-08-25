import polars as pl


def multi_cancer_summary(
    df: pl.DataFrame,
    variant_cols: list[str],
    prefix: str,
    patient_col: str = "PATIENT_ID",
    cancer_col: str = "CANCER_TYPE",
) -> pl.DataFrame:
    """
    For a given variant, identify patients that appear in multiple cancer
    types and summarise.

    Parameters
    ----------
    df : pl.DataFrame
        dataframe of variants
    variant_cols : list[str]
        list of columns to group by
    prefix : str
        the count type which is the prefix of the column name
    patient_col : str
        column which has patient IDs, by default "PATIENT_ID"
    cancer_col : str
        column which has the cancer type, by default "CANCER_TYPE"

    Returns
    -------
    pl.DataFrame
        dataframe listing patient count and patient IDs for multiple cancer
        types
    """
    per_patient = (
        df.group_by(variant_cols + [patient_col])
        .agg(pl.col(cancer_col).unique().sort().alias("cancers"))
        .filter(pl.col("cancers").list.len() > 1)
        .with_columns(
            (
                pl.col(patient_col) + ":" + pl.col("cancers").list.join("|")
            ).alias("patient_cancer_map")
        )
    )

    return per_patient.group_by(variant_cols).agg(
        pl.len().alias(f"{prefix}.Duplicate_Patient_Count"),
        pl.col("patient_cancer_map")
        .sort()
        .str.join("&")
        .alias(f"{prefix}.Duplicate_Patient_IDs"),
    )


def rename(cols, suffix_fn, ignore_cols):
    """
    Rename columns using a function.

    Parameters
    ----------
    cols : list
        columns to rename
    suffix_fn : function
        function to apply to the column
    ignore_cols : list
        list of columns to ignore

    Returns
    -------
    list
        list of renamed columns
    """
    ignore = set(ignore_cols)
    return [col if col in ignore else suffix_fn(col) for col in cols]


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

        result = all_variants.join(
            nucleotide_change_counts, on="grch38_description", how="left"
        ).with_columns(
            [
                pl.col(count_col).fill_null(0).cast(pl.Int64),
            ]
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
    counts_wide = per_cancer_counts.pivot(
        values="patient_count",
        index="grch38_description",
        on="CANCER_TYPE",
        aggregate_function="first",
    ).fill_null(0)

    counts_wide.columns = rename(
        counts_wide.columns,
        lambda c: (
            f"SameNucleotideChange.{c}_Count_N_{unique_patients_per_cancer[c]}"
        ),
        ["grch38_description"],
    )

    multi_cancer = multi_cancer_summary(
        df, variant_cols=["grch38_description"], prefix="SameNucleotideChange"
    )

    return counts_wide.join(
        multi_cancer, on="grch38_description", how="left"
    ).with_columns(
        pl.col("SameNucleotideChange.Duplicate_Patient_Count")
        .fill_null(0)
        .cast(pl.Int64),
        pl.col("SameNucleotideChange.Duplicate_Patient_IDs").fill_null(""),
    )


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
        .group_by(["Hugo_Symbol", "RefSeq", "HGVSp"])
        .agg(
            pl.col("PATIENT_ID").n_unique().cast(pl.Int64).alias(count_col),
        )
    )

    # If this is a grouped (e.g. haemonc cancer) count, then all variants
    # with HGVSp should have a count, so add 0 if not present in grouped count
    if all_variants_df is not None:
        result = (
            all_variants_df.filter(pl.col("HGVSp").is_not_null())
            .select(["grch38_description", "Hugo_Symbol", "RefSeq", "HGVSp"])
            .unique()
            .join(
                amino_acid_change_counts,
                on=["Hugo_Symbol", "RefSeq", "HGVSp"],
                how="left",
            )
            .with_columns(
                [
                    pl.col(count_col).fill_null(0).cast(pl.Int64),
                ]
            )
        ).drop(["Hugo_Symbol", "RefSeq", "HGVSp"])
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
        .group_by(["Hugo_Symbol", "HGVSp", "RefSeq", "CANCER_TYPE"])
        .agg(
            pl.col("PATIENT_ID")
            .n_unique()
            .cast(pl.Int64)
            .alias("patient_count")
        )
    )

    # Pivot so all cancer types have counts
    counts_wide = amino_acid_count_per_present_cancer.pivot(
        values="patient_count",
        index=["Hugo_Symbol", "HGVSp", "RefSeq"],
        on="CANCER_TYPE",
        aggregate_function="first",
    ).fill_null(0)

    counts_wide.columns = rename(
        counts_wide.columns,
        lambda c: (
            f"SameAminoAcidChange.{c}_Count_N_{unique_patients_per_cancer[c]}"
        ),
        ["Hugo_Symbol", "HGVSp", "RefSeq"],
    )

    multi_cancer = multi_cancer_summary(
        df,
        prefix="SameAminoAcidChange",
        variant_cols=["Hugo_Symbol", "HGVSp", "RefSeq"],
    )

    return counts_wide.join(
        multi_cancer, on=["Hugo_Symbol", "HGVSp", "RefSeq"], how="left"
    ).with_columns(
        pl.col("SameAminoAcidChange.Duplicate_Patient_Count")
        .fill_null(0)
        .cast(pl.Int64),
        pl.col("SameAminoAcidChange.Duplicate_Patient_IDs").fill_null(""),
    )


def count_frameshift_truncating_and_nonsense(
    df: pl.DataFrame,
    cancer_count_type: str,
    patient_total: int,
    truncating_variants: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Count unique patients with frameshift or nonsense variants at the same
    position or downstream for each gene.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing truncating variants with 'Hugo_Symbol',
        'RefSeq', and 'Protein_position'.
    cancer_count_type : str
        Cancer count type (used in column naming)
    patient_total : int
        Total number of unique patients in the dataset.
    truncating_variants : pl.DataFrame
        dataframe of all truncating variants in the dataset

    Returns
    -------
    pl.DataFrame
        DataFrame with 'Protein_position' and downstream counts per gene.
    """
    all_results = []

    # Anchor positions (the variants a count is computed for) must come from
    # the full, unfiltered set of truncating variants when given, so that a
    # position with no patients in this subgroup (e.g. solid/haemonc cancers)
    # is still a valid anchor. Only the patients counted as being at-or-
    # downstream of that position are restricted to this subgroup's df.
    anchor_df = truncating_variants if truncating_variants is not None else df

    # Iterate over unique (gene, transcript) pairs
    for gene, transcript in (
        anchor_df.select(["Hugo_Symbol", "RefSeq"]).unique().iter_rows()
    ):
        anchor_subset = anchor_df.filter(
            (pl.col("Hugo_Symbol") == gene) & (pl.col("RefSeq") == transcript)
        )
        subset = df.filter(
            (pl.col("Hugo_Symbol") == gene) & (pl.col("RefSeq") == transcript)
        )
        positions = sorted(
            anchor_subset["Protein_position_start"].unique().to_list()
        )

        rows = []
        for pos in positions:
            downstream_patients = {
                pid
                for j, pid in enumerate(subset["PATIENT_ID"])
                if subset["Protein_position_start"][j] >= pos
            }
            rows.append(
                {
                    "Hugo_Symbol": gene,
                    "RefSeq": transcript,
                    "Protein_position_start": pos,
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
                "Hugo_Symbol": pl.Series([], dtype=pl.Utf8),
                "RefSeq": pl.Series([], dtype=pl.Utf8),
                "Protein_position_start": pl.Series([], dtype=pl.Int64),
                "downstream_patient_count": pl.Series([], dtype=pl.Int64),
            }
        )
    )

    # If this is a grouped (e.g. haemonc cancer) count, then all truncating
    #  variants should have a count, so add 0 if not present in grouped count
    col_name = f"SameOrDownstreamTruncatingVariantsPerAA.{cancer_count_type}_Count_N_{patient_total}"
    df_counts = df_counts.rename({"downstream_patient_count": col_name})

    # If given, join back to truncating_variants to ensure all rows are present
    if truncating_variants is not None:
        result = (
            truncating_variants.select(
                [
                    "grch38_description",
                    "Hugo_Symbol",
                    "RefSeq",
                    "Protein_position_start",
                ]
            )
            .unique()
            .join(
                df_counts,
                on=["Hugo_Symbol", "RefSeq", "Protein_position_start"],
                how="left",
            )
            .with_columns(pl.col(col_name).fill_null(0))
        ).drop("Hugo_Symbol", "RefSeq", "Protein_position_start")
        return result

    return df_counts


def count_frameshift_truncating_and_nonsense_per_cancer_type(
    df: pl.DataFrame, per_cancer_patient_total: dict
) -> pl.DataFrame:
    """
    Count patients with frameshift (truncating) or nonsense variants at the
    same position or downstream in the same transcript, and flag patients
    contributing to multiple cancer types.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing truncating variants with 'Hugo_Symbol',
        'RefSeq', 'Protein_position', 'CANCER_TYPE', and 'PATIENT_ID'.
    per_cancer_patient_total : dict
        Total number of unique patients in the dataset per cancer type.

    Returns
    -------
    pl.DataFrame
        DataFrame with Protein_position and downstream counts per gene per cancer type.
    """
    all_gene_results = []

    # Iterate over unique (gene, transcript) pairs
    for gene in df["Hugo_Symbol"].unique().to_list():
        gene_df = df.filter(pl.col("Hugo_Symbol") == gene)

        for transcript in gene_df["RefSeq"].unique().to_list():
            tx_df = gene_df.filter(pl.col("RefSeq") == transcript)

            positions = sorted(
                tx_df["Protein_position_start"].unique().to_list()
            )

            rows = []
            for pos in positions:
                # Patients at or downstream of this position
                downstream_df = tx_df.filter(
                    pl.col("Protein_position_start") >= pos
                )

                # Count unique patients per cancer type
                counts_per_cancer = {
                    f"SameOrDownstreamTruncatingVariantsPerAA.{cancer}_Count_N_{per_cancer_patient_total[cancer]}": (
                        downstream_df.filter(pl.col("CANCER_TYPE") == cancer)[
                            "PATIENT_ID"
                        ].n_unique()
                    )
                    for cancer in per_cancer_patient_total
                }

                # Identify multi-cancer patients and map to cancer types
                patient_cancer_map = {}
                for row in downstream_df.iter_rows(named=True):
                    patient = row["PATIENT_ID"]
                    cancer = row["CANCER_TYPE"]
                    patient_cancer_map.setdefault(patient, set()).add(cancer)

                multi_cancer_map = {
                    p: "|".join(sorted(cancers))
                    for p, cancers in patient_cancer_map.items()
                    if len(cancers) > 1
                }

                multi_cancer_ids_str = "&".join(
                    f"{p}:{cs}" for p, cs in sorted(multi_cancer_map.items())
                )

                # Build row
                row_data = {
                    "Hugo_Symbol": gene,
                    "RefSeq": transcript,
                    "Protein_position_start": pos,
                    **counts_per_cancer,
                    "SameOrDownstreamTruncatingVariantsPerAA.Duplicate_Patient_Count": len(
                        multi_cancer_map
                    ),
                    "SameOrDownstreamTruncatingVariantsPerAA.Duplicate_Patient_IDs": (
                        multi_cancer_ids_str
                    ),
                }

                rows.append(row_data)

            all_gene_results.append(pl.DataFrame(rows))

    # Combine all results
    if all_gene_results:
        return pl.concat(all_gene_results, how="vertical")
    else:
        empty_cols = {
            "Hugo_Symbol": pl.Series([], dtype=pl.Utf8),
            "RefSeq": pl.Series([], dtype=pl.Utf8),
            "Protein_position_start": pl.Series([], dtype=pl.Int64),
            **{
                f"SameOrDownstreamTruncatingVariantsPerAA.{cancer}_Count_N_{per_cancer_patient_total[cancer]}": pl.Series(
                    [], dtype=pl.Int64
                )
                for cancer in per_cancer_patient_total
            },
            "SameOrDownstreamTruncatingVariantsPerAA.Duplicate_Patient_Count": pl.Series(
                [], dtype=pl.Int64
            ),
            "SameOrDownstreamTruncatingVariantsPerAA.Duplicate_Patient_IDs": (
                pl.Series([], dtype=pl.Utf8)
            ),
        }
        return pl.DataFrame(empty_cols)


def add_deletion_positions(
    inframe_deletions: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add start and end positions as new columns to the inframe deletions
    DataFrame based on the Protein_position.

    Parameters
    ----------
    inframe_deletions : pl.DataFrame
        DataFrame containing inframe deletions and the positions of the deletion

    Returns
    -------
    pl.DataFrame
        DataFrame with additional columns for deletion start and end
    """
    df = (
        inframe_deletions.with_columns(
            pl.col("Protein_position")
            .str.splitn("-", 2)
            .struct.rename_fields(["del_start", "del_end"])
        )
        .unnest("Protein_position")
        .with_columns(
            [
                pl.col("del_start").cast(pl.Int64),
                pl.col("del_end")
                .fill_null(pl.col("del_start"))
                .cast(pl.Int64),
            ]
        )
    )

    return df


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
        DataFrame containing a set of inframe deletions
    cancer_count_type : str
        The cancer count type, to include in name of the column
    patient_total : int
        Total number of unique patients in the dataset
    inframe_deletions: pl.DataFrame | None
        Reference dataset to ensure all inframe deletions are included

    Returns
    -------
    pl.DataFrame
        DataFrame with counts of matching or nested inframe deletions.
    """
    all_results = []

    # Anchor ranges (the deletions a count is computed for) must come from
    # the full, unfiltered set of inframe deletions when given, so that a
    # range with no patients in this subgroup (e.g. solid/haemonc cancers)
    # is still a valid anchor. Only the patients counted as nested within
    # that range are restricted to this subgroup's inframe_deletions_df.
    anchor_df = (
        inframe_deletions
        if inframe_deletions is not None
        else inframe_deletions_df
    )

    # Iterate over (gene, transcript) pairs
    for gene, transcript in (
        anchor_df.select(["Hugo_Symbol", "RefSeq"]).unique().iter_rows()
    ):
        anchor_subset = anchor_df.filter(
            (pl.col("Hugo_Symbol") == gene) & (pl.col("RefSeq") == transcript)
        )
        subset = inframe_deletions_df.filter(
            (pl.col("Hugo_Symbol") == gene) & (pl.col("RefSeq") == transcript)
        )

        unique_ranges = (
            anchor_subset.select(
                ["Hugo_Symbol", "RefSeq", "del_start", "del_end"]
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
                    "RefSeq": transcript,
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
                "Hugo_Symbol": pl.Series([], dtype=pl.Utf8),
                "RefSeq": pl.Series([], dtype=pl.Utf8),
                "del_start": pl.Series([], dtype=pl.Int64),
                "del_end": pl.Series([], dtype=pl.Int64),
                "nested_patient_count": pl.Series([], dtype=pl.Int64),
            }
        )
    )

    # Rename nested count column with cohort info
    col_name = f"NestedInframeDeletionsPerAA.{cancer_count_type}_Count_N_{patient_total}"
    inframe_counts = inframe_counts.rename({"nested_patient_count": col_name})

    # If given, join back to reference deletions to ensure all rows are present
    if inframe_deletions is not None:
        result = (
            inframe_deletions.select(
                [
                    "Hugo_Symbol",
                    "grch38_description",
                    "RefSeq",
                    "del_start",
                    "del_end",
                ]
            )
            .unique()
            .join(
                inframe_counts,
                on=["Hugo_Symbol", "RefSeq", "del_start", "del_end"],
                how="left",
            )
            .with_columns(pl.col(col_name).fill_null(0))
        ).drop("Hugo_Symbol", "RefSeq", "del_start", "del_end")

        return result

    return inframe_counts


def count_nested_inframe_deletions_per_cancer_type(
    inframe_deletions_df: pl.DataFrame,
    per_cancer_patient_total: dict,
) -> pl.DataFrame:
    """
    Count the number of unique patients with inframe deletions that are either
    the same as or nested within the current deletion, grouped by cancer type.
    Flag any individual patients with multiple cancer types.

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
    all_results = []

    # Iterate over unique (gene, transcript) pairs
    for gene, transcript in (
        inframe_deletions_df.select(["Hugo_Symbol", "RefSeq"])
        .unique()
        .iter_rows()
    ):
        subset_gene_tx = inframe_deletions_df.filter(
            (pl.col("Hugo_Symbol") == gene) & (pl.col("RefSeq") == transcript)
        )

        # Get unique deletion ranges for this transcript
        for del_start, del_end in (
            subset_gene_tx.select(["del_start", "del_end"])
            .unique()
            .sort(["del_start", "del_end"])
            .iter_rows()
        ):
            nested_df = subset_gene_tx.filter(
                (pl.col("del_start") >= del_start)
                & (pl.col("del_end") <= del_end)
            )

            # Count patients per cancer type
            counts_per_cancer = {
                cancer: (
                    nested_df.filter(pl.col("CANCER_TYPE") == cancer)[
                        "PATIENT_ID"
                    ].n_unique()
                )
                for cancer in per_cancer_patient_total
            }

            # Multi-cancer patient summary
            per_patient = (
                nested_df.group_by(["PATIENT_ID"])
                .agg(pl.col("CANCER_TYPE").unique().sort().alias("cancers"))
                .filter(pl.col("cancers").list.len() > 1)
                .with_columns(
                    (
                        pl.col("PATIENT_ID")
                        + ":"
                        + pl.col("cancers").list.join("|")
                    ).alias("patient_cancer_map")
                )
            )

            multi_cancer_count = per_patient.height
            multi_cancer_ids = "&".join(
                per_patient["patient_cancer_map"].to_list()
            )

            # Build row
            row_data = {
                "Hugo_Symbol": gene,
                "RefSeq": transcript,
                "del_start": del_start,
                "del_end": del_end,
                **counts_per_cancer,
                "Duplicate_Patient_Count": multi_cancer_count,
                "Duplicate_Patient_IDs": multi_cancer_ids,
            }

            all_results.append(row_data)

    # Combine all rows
    combined_df = (
        pl.DataFrame(all_results)
        if all_results
        else pl.DataFrame(
            {
                "Hugo_Symbol": pl.Series([], dtype=pl.Utf8),
                "RefSeq": pl.Series([], dtype=pl.Utf8),
                "del_start": pl.Series([], dtype=pl.Int64),
                "del_end": pl.Series([], dtype=pl.Int64),
                **{
                    cancer: pl.Series([], dtype=pl.Int64)
                    for cancer in per_cancer_patient_total
                },
                "Duplicate_Patient_Count": pl.Series([], dtype=pl.Int64),
                "Duplicate_Patient_IDs": pl.Series([], dtype=pl.Utf8),
            }
        )
    )

    # Rename columns to match naming convention
    rename_mapping = {
        **{
            cancer: (
                f"NestedInframeDeletionsPerAA.{cancer}_Count_N_{per_cancer_patient_total[cancer]}"
            )
            for cancer in per_cancer_patient_total
            if cancer in combined_df.columns
        },
        "Duplicate_Patient_Count": (
            "NestedInframeDeletionsPerAA.Duplicate_Patient_Count"
        ),
        "Duplicate_Patient_IDs": (
            "NestedInframeDeletionsPerAA.Duplicate_Patient_IDs"
        ),
    }

    result = combined_df.rename(rename_mapping)

    return result
