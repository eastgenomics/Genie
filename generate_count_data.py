import argparse
import pandas as pd
import polars as pl

from datetime import datetime

from utils.file_io import (
    read_in_to_df,
    read_in_to_polars_df,
    read_txt_file_to_list,
)
from utils.aggregation import (
    calculate_unique_patient_counts,
    create_df_with_one_row_per_variant,
    get_truncating_variants,
    get_inframe_deletions,
    get_haemonc_cancer_rows,
)
from utils.dtypes import column_dtypes
from utils.counting import (
    count_same_nucleotide_change,
    count_same_nucleotide_change_per_cancer_type,
    count_amino_acid_change,
    count_amino_acid_change_per_cancer_type,
    count_frameshift_truncating_and_nonsense,
    count_frameshift_truncating_and_nonsense_per_cancer_type,
    add_deletion_positions,
    extract_position_from_cds,
    count_nested_inframe_deletions,
    count_nested_inframe_deletions_per_cancer_type,
)
from utils.merging import (
    multi_merge,
    merge_truncating_variants_counts,
    merge_inframe_deletions_with_counts,
    merge_truncating_variant_counts_haemonc,
    merge_inframe_deletions_haemonc_counts,
    multi_merge_polars,
    reorder_final_columns,
    reorder_final_columns_polars,
    merge_truncating_variant_counts_haemonc_polars,
    merge_inframe_deletions_haemonc_counts_polars,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns
    -------
    args : Namespace
        Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser(
        description="Information required to generate counts from MAF file"
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help=(
            "Path to MAF file which includes patient and sample information "
            "and GRCh38 liftover"
        ),
    )

    parser.add_argument(
        "--columns_to_aggregate",
        required=True,
        type=str,
        help="TXT file containing a list of variant columns to aggregate",
    )

    parser.add_argument(
        "--haemonc_cancer_types",
        required=False,
        type=str,
        help=(
            "Path to file which lists haemonc cancer types we're interested in"
        ),
    )

    parser.add_argument(
        "--output", required=True, type=str, help="Name of output file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # genie_data = read_in_to_df(args.input, header=0, dtype=column_dtypes)
    genie_data = read_in_to_polars_df(args.input, sep="\t")
    columns_to_aggregate = read_txt_file_to_list(args.columns_to_aggregate)
    if args.haemonc_cancer_types:
        haemonc_cancers = read_txt_file_to_list(args.haemonc_cancer_types)
    else:
        haemonc_cancers = None

    (patient_total, per_cancer_patient_total, haemonc_cancer_patient_total) = (
        calculate_unique_patient_counts(genie_data, haemonc_cancers)
    )

    print(f"{datetime.now()}, Aggregating data")
    one_row_per_variant_agg = create_df_with_one_row_per_variant(
        df=genie_data,
        columns_to_aggregate=columns_to_aggregate,
    )

    print(f"{datetime.now()}, Generating nucleotide counts")
    nucleotide_change_counts_all_cancer = count_same_nucleotide_change(
        df=genie_data.select("grch38_description", "PATIENT_ID"),
        unique_patient_total=patient_total,
        count_type="All_Cancers",
    )

    nucleotide_change_counts_per_cancer = (
        count_same_nucleotide_change_per_cancer_type(
            df=genie_data.select(
                "grch38_description", "PATIENT_ID", "CANCER_TYPE"
            ),
            unique_patients_per_cancer=per_cancer_patient_total,
        )
    )

    merged_nt_counts = nucleotide_change_counts_all_cancer.join(
        nucleotide_change_counts_per_cancer,
        on="grch38_description",
        how="left",
    )

    print(f"{datetime.now()}, Generating amino acid counts")
    amino_acid_change_counts_all_cancer = count_amino_acid_change(
        df=genie_data.select("Hugo_Symbol", "HGVSp", "PATIENT_ID"),
        unique_patient_total=patient_total,
        count_type="All_Cancers",
    )

    amino_acid_change_counts_per_cancer = (
        count_amino_acid_change_per_cancer_type(
            df=genie_data.select(
                "Hugo_Symbol", "HGVSp", "PATIENT_ID", "CANCER_TYPE"
            ),
            unique_patients_per_cancer=per_cancer_patient_total,
        )
    )

    merged_aa_counts = amino_acid_change_counts_all_cancer.join(
        amino_acid_change_counts_per_cancer,
        on=["Hugo_Symbol", "HGVSp"],
        how="left",
    )

    print(f"{datetime.now()}, Generating truncating variant counts")
    truncating_variants = get_truncating_variants(genie_data).select(
        "Hugo_Symbol",
        "grch38_description",
        "HGVSc",
        "PATIENT_ID",
        "CANCER_TYPE",
    )
    truncating_plus_position = extract_position_from_cds(truncating_variants)

    frameshift_counts_all_cancers = count_frameshift_truncating_and_nonsense(
        df=truncating_plus_position,
        patient_total=patient_total,
        cancer_count_type="All_Cancers",
    )

    truncating_plus_position_pd = truncating_plus_position.to_pandas()
    frameshift_counts_per_cancer = (
        count_frameshift_truncating_and_nonsense_per_cancer_type(
            df=truncating_plus_position_pd,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )
    frameshift_counts_per_cancer = pl.from_pandas(frameshift_counts_per_cancer)

    truncating_variants_no_dups = truncating_plus_position.unique(
        subset=["grch38_description"], keep="first"
    ).select(["Hugo_Symbol", "grch38_description", "CDS_position"])

    trunc_counts = multi_merge_polars(
        truncating_variants_no_dups,
        [frameshift_counts_all_cancers, frameshift_counts_per_cancer],
        on=["Hugo_Symbol", "CDS_position"],
        how="left",
    )
    trunc_counts = trunc_counts.drop(["Hugo_Symbol", "CDS_position"])

    print(datetime.now(), "Generating inframe deletion counts")
    inframe_deletions = get_inframe_deletions(genie_data).select(
        "grch38_description",
        "Hugo_Symbol",
        "HGVSc",
        "PATIENT_ID",
        "CANCER_TYPE",
    )
    inframe_deletions_with_positions = add_deletion_positions(
        inframe_deletions
    )
    inframe_deletions_count_all_cancers = count_nested_inframe_deletions(
        inframe_deletions_df=inframe_deletions_with_positions,
        cancer_count_type="All_Cancers",
        patient_total=patient_total,
    )

    inframe_deletions_count_per_cancer = (
        count_nested_inframe_deletions_per_cancer_type(
            inframe_deletions_df=inframe_deletions_with_positions,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )

    inframe_with_positions_no_dups = inframe_deletions_with_positions.unique(
        subset="grch38_description", keep="first"
    ).select(["Hugo_Symbol", "grch38_description", "del_start", "del_end"])

    print(datetime.now(), "Merging inframe positions and counts")
    merged_counts = multi_merge_polars(
        inframe_deletions_count_all_cancers,
        [inframe_deletions_count_per_cancer],
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    )

    inframe_deletions_with_counts = inframe_with_positions_no_dups.join(
        merged_counts,
        on=["Hugo_Symbol", "del_start", "del_end"],
        how="left",
    ).drop("Hugo_Symbol", "del_start", "del_end")

    one_row_per_variant_agg = one_row_per_variant_agg.join(
        merged_nt_counts,
        on="grch38_description",
        how="left",
    )

    print(datetime.now(), "Merging amino acid counts")
    one_row_per_variant_agg = one_row_per_variant_agg.join(
        merged_aa_counts,
        on=["Hugo_Symbol", "HGVSp"],
        how="left",
    )

    print(datetime.now(), "Merging truncating variant counts")
    one_row_per_variant_agg = one_row_per_variant_agg.join(
        trunc_counts,
        on="grch38_description",
        how="left",
    )

    print(datetime.now(), "Merging inframe deletion counts")
    one_row_per_variant_agg = one_row_per_variant_agg.join(
        inframe_deletions_with_counts,
        on="grch38_description",
        how="left",
    )

    if args.haemonc_cancer_types:
        haemonc_rows = get_haemonc_cancer_rows(
            df=genie_data,
            haemonc_cancers=haemonc_cancers,
        )

        print(datetime.now(), "Generating nucleotide haemonc cancer counts")
        nucleotide_counts_haemonc_cancers = count_same_nucleotide_change(
            df=haemonc_rows,
            unique_patient_total=haemonc_cancer_patient_total,
            count_type="Haemonc_Cancers",
            all_variants_df=genie_data,
        )

        merged_nt_haemonc_counts = multi_merge_polars(
            base_df=one_row_per_variant_agg,
            merge_dfs=[nucleotide_counts_haemonc_cancers],
            on="grch38_description",
            how="left",
        )

        all_variants_with_hgvsp = genie_data.filter(
            pl.col("HGVSp").is_not_null()
        )

        print(datetime.now(), "Generating amino acid haemonc counts")
        amino_acid_counts_haemonc_cancers = count_amino_acid_change(
            df=haemonc_rows,
            unique_patient_total=haemonc_cancer_patient_total,
            count_type="Haemonc_Cancers",
            all_variants_df=all_variants_with_hgvsp,
        )

        print(datetime.now(), "Merging amino acid haemonc counts")
        merged_aa_haemonc_counts = merged_nt_haemonc_counts.join(
            amino_acid_counts_haemonc_cancers,
            on=["Hugo_Symbol", "HGVSp"],
            how="left",
        )

        print(datetime.now(), "Haemonc truncating counts")
        truncating_variants_haemonc = get_truncating_variants(df=haemonc_rows)
        truncating_variants_haemonc_position = extract_position_from_cds(
            df=truncating_variants_haemonc
        )

        frameshift_counts_haemonc = count_frameshift_truncating_and_nonsense(
            df=truncating_variants_haemonc_position,
            patient_total=haemonc_cancer_patient_total,
            cancer_count_type="Haemonc_Cancers",
        )

        print(datetime.now(), "Merging haemonc truncating counts")
        merged_frameshift_ho = merge_truncating_variant_counts_haemonc_polars(
            merged_aa_haemonc_counts=merged_aa_haemonc_counts,
            truncating_plus_position=truncating_plus_position,
            frameshift_counts_haemonc=frameshift_counts_haemonc,
        )

        inframe_deletions_haemonc = get_inframe_deletions(df=haemonc_rows)
        inframe_deletions_haemonc_positions = add_deletion_positions(
            inframe_deletions_haemonc
        )
        inframe_deletions_count_haemonc_cancers = (
            count_nested_inframe_deletions(
                inframe_deletions_df=inframe_deletions_haemonc_positions,
                cancer_count_type="Haemonc_Cancers",
                patient_total=haemonc_cancer_patient_total,
            )
        )

        print(datetime.now(), "Merging in frame deletion haemonc counts")
        one_row_per_variant_agg = merge_inframe_deletions_haemonc_counts_polars(
            inframe_deletions=inframe_deletions_with_positions,
            inframe_deletions_count_haemonc_cancers=inframe_deletions_count_haemonc_cancers,
            merged_frameshift_ho=merged_frameshift_ho,
        )

    print(datetime.now(), "Reordering final columns")
    final_df = reorder_final_columns_polars(
        one_row_per_variant_agg,
        patient_total,
        per_cancer_patient_total,
        haemonc_cancer_patient_total,
    )

    final_df.write_csv(
        args.output,
        separator="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
