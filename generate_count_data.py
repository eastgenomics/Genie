import argparse
import pandas as pd


from utils.file_io import read_in_to_df, read_txt_file_to_list
from utils.aggregation import (
    calculate_unique_patient_counts,
    create_df_with_one_row_per_variant,
    get_truncating_variants,
    get_inframe_deletions,
)
from utils.counting import (
    count_same_nucleotide_change_all_cancers,
    count_same_nucleotide_change_per_cancer_type,
    count_amino_acid_change_all_cancers,
    count_amino_acid_change_per_cancer_type,
    count_frameshift_truncating_and_nonsense_all_cancers,
    count_frameshift_truncating_and_nonsense_per_cancer_type,
    add_deletion_positions,
    extract_position_affected,
    count_nested_inframe_deletions_all_cancers,
    count_nested_inframe_deletions_per_cancer_type,
)
from utils.merging import (
    multi_merge,
    merge_truncating_variants_counts,
    merge_inframe_deletions_with_counts,
    reorder_final_columns,
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
    genie_data = read_in_to_df(
        args.input,
        header=0,
        dtype={
            "Hugo_Symbol": "str",
            "Start_Position": "Int64",
            "Entrez_Gene_Id": "Int64",
        },
        converters={
            col: lambda x: str(x).strip() if pd.notnull(x) else x
            for col in [
                "Chromosome",
                "Reference_Allele",
                "Tumor_Seq_Allele2",
                "chrom_grch38",
                "pos_grch38",
                "ref_grch38",
                "alt_grch38",
            ]
        },
    )

    columns_to_aggregate = read_txt_file_to_list(args.columns_to_aggregate)
    if args.haemonc_cancer_types:
        haemonc_cancers = read_txt_file_to_list(args.haemonc_cancer_types)
    else:
        haemonc_cancers = None

    (patient_total, per_cancer_patient_total, haemonc_cancer_patient_total) = (
        calculate_unique_patient_counts(genie_data, haemonc_cancers)
    )

    one_row_per_variant_agg = create_df_with_one_row_per_variant(
        df=genie_data,
        columns_to_aggregate=columns_to_aggregate,
    )

    nucleotide_change_counts_all_cancer = (
        count_same_nucleotide_change_all_cancers(
            df=genie_data, unique_patient_total=patient_total
        )
    )

    nucleotide_change_counts_per_cancer = (
        count_same_nucleotide_change_per_cancer_type(
            df=genie_data,
            unique_patients_per_cancer=per_cancer_patient_total,
        )
    )

    merged_nucleotide_counts = multi_merge(
        base_df=one_row_per_variant_agg,
        merge_dfs=[
            nucleotide_change_counts_all_cancer,
            nucleotide_change_counts_per_cancer,
        ],
        on="grch38_description",
    )

    amino_acid_change_counts_all_cancer = count_amino_acid_change_all_cancers(
        df=genie_data,
        unique_patient_total=patient_total,
    )

    amino_acid_change_counts_per_cancer = (
        count_amino_acid_change_per_cancer_type(
            df=genie_data,
            unique_patients_per_cancer=per_cancer_patient_total,
        )
    )

    merged_amino_acid_counts = multi_merge(
        base_df=merged_nucleotide_counts,
        merge_dfs=[
            amino_acid_change_counts_all_cancer,
            amino_acid_change_counts_per_cancer,
        ],
        on=["Hugo_Symbol", "HGVSp"],
    )

    truncating_variants = get_truncating_variants(genie_data)
    truncating_plus_position = extract_position_affected(truncating_variants)

    frameshift_counts_all_cancers = (
        count_frameshift_truncating_and_nonsense_all_cancers(
            df=truncating_plus_position,
            patient_total=patient_total,
        )
    )

    frameshift_counts_per_cancer = (
        count_frameshift_truncating_and_nonsense_per_cancer_type(
            df=truncating_plus_position,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )

    merged_frameshift_counts = merge_truncating_variants_counts(
        merged_amino_acid_counts=merged_amino_acid_counts,
        truncating_variants=truncating_plus_position,
        truncating_counts_all_cancers=frameshift_counts_all_cancers,
        truncating_counts_per_cancer=frameshift_counts_per_cancer,
    )

    inframe_deletions = get_inframe_deletions(genie_data)
    inframe_deletions_with_positions = add_deletion_positions(
        inframe_deletions
    )
    inframe_deletions_count_all_cancers = (
        count_nested_inframe_deletions_all_cancers(
            inframe_deletions_df=inframe_deletions_with_positions,
            patient_total=patient_total,
        )
    )

    inframe_deletions_count_per_cancer = (
        count_nested_inframe_deletions_per_cancer_type(
            inframe_deletions_df=inframe_deletions_with_positions,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )

    all_counts_merged = merge_inframe_deletions_with_counts(
        merged_frameshift_counts=merged_frameshift_counts,
        inframe_deletions_with_positions=inframe_deletions_with_positions,
        inframe_deletions_count_all_cancers=inframe_deletions_count_all_cancers,
        inframe_deletions_count_per_cancer=inframe_deletions_count_per_cancer,
    )

    final_df = reorder_final_columns(
        all_counts_merged,
        patient_total,
        per_cancer_patient_total,
        haemonc_cancer_patient_total,
    )

    final_df.to_csv(
        args.output,
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
