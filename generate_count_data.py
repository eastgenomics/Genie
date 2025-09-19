import argparse
import gc
from datetime import datetime

from utils.file_io import (
    read_in_to_polars_df,
    read_txt_file_to_list,
)
from utils.aggregation import (
    calculate_unique_patient_counts,
    create_df_with_one_row_per_variant,
    get_rows_for_cancer_types,
    get_truncating_variants,
    get_inframe_deletions,
)
from utils.counting import (
    count_same_nucleotide_change,
    count_same_nucleotide_change_per_cancer_type,
    count_amino_acid_change,
    count_amino_acid_change_per_cancer_type,
    count_frameshift_truncating_and_nonsense,
    count_frameshift_truncating_and_nonsense_per_cancer_type,
    add_deletion_positions,
    extract_position_from_hgvsc,
    count_nested_inframe_deletions,
    count_nested_inframe_deletions_per_cancer_type,
)
from utils.merging import (
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
        help="Path to file which lists haemonc cancer types for grouping",
    )

    parser.add_argument(
        "--solid_cancer_types",
        required=False,
        type=str,
        help="Path to file which lists solid cancer types for grouping",
    )

    parser.add_argument(
        "--column_for_inframe_deletions",
        required=True,
        choices=["HGVSc", "HGVSp"],
        help=(
            "Column to use for extracting deletion positions for inframe"
            " deletions"
        ),
    )

    parser.add_argument(
        "--output", required=True, type=str, help="Name of output file"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    genie_data = read_in_to_polars_df(args.input, sep="\t")
    columns_to_aggregate = read_txt_file_to_list(args.columns_to_aggregate)

    deletion_source = args.column_for_inframe_deletions
    haemonc_cancers = solid_cancers = None
    if args.haemonc_cancer_types:
        haemonc_cancers = read_txt_file_to_list(args.haemonc_cancer_types)
    if args.solid_cancer_types:
        solid_cancers = read_txt_file_to_list(args.solid_cancer_types)

    # Validate no overlap between cancer type groups
    if haemonc_cancers and solid_cancers:
        overlap = set(haemonc_cancers) & set(solid_cancers)
        if overlap:
            raise ValueError(
                "The haemonc and solid cancer type lists have the following"
                f" overlap: {', '.join(overlap)}. Please ensure there is no"
                " overlap between the two lists."
            )
    (
        patient_total,
        per_cancer_patient_total,
        haemonc_cancer_patient_total,
        solid_cancer_patient_total,
    ) = calculate_unique_patient_counts(
        genie_data, haemonc_cancers, solid_cancers
    )

    print(
        f"{datetime.now().replace(microsecond=0)} Aggregating data to one row"
        " per variant"
    )
    one_row_per_variant_agg = create_df_with_one_row_per_variant(
        df=genie_data,
        columns_to_aggregate=columns_to_aggregate,
    )

    print(
        f"{datetime.now().replace(microsecond=0)} Generating nucleotide counts"
    )
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

    print(
        f"{datetime.now().replace(microsecond=0)} Generating amino acid counts"
    )
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

    print(
        f"{datetime.now().replace(microsecond=0)} Generating truncating"
        " variant counts"
    )
    truncating_variants = get_truncating_variants(genie_data).select(
        "Hugo_Symbol",
        "grch38_description",
        "Transcript_ID",
        "HGVSc",
        "PATIENT_ID",
        "CANCER_TYPE",
    )
    truncating_variants = extract_position_from_hgvsc(truncating_variants)

    truncating_counts_all_cancers = count_frameshift_truncating_and_nonsense(
        df=truncating_variants,
        patient_total=patient_total,
        cancer_count_type="All_Cancers",
    )

    truncating_counts_per_cancer = (
        count_frameshift_truncating_and_nonsense_per_cancer_type(
            df=truncating_variants,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )

    trunc_counts = merge_truncating_variants_counts(
        truncating_variants=truncating_variants,
        truncating_counts_all_cancers=truncating_counts_all_cancers,
        truncating_counts_per_cancer=truncating_counts_per_cancer,
    )

    print(
        f"{datetime.now().replace(microsecond=0)} Generating inframe"
        " deletion counts"
    )
    inframe_deletions = get_inframe_deletions(
        df=genie_data, column_used=deletion_source
    ).select(
        "grch38_description",
        "Hugo_Symbol",
        "Transcript_ID",
        deletion_source,
        "PATIENT_ID",
        "CANCER_TYPE",
    )
    inframe_deletions = add_deletion_positions(
        inframe_deletions, source=deletion_source
    )
    inframe_deletions_count_all_cancers = count_nested_inframe_deletions(
        inframe_deletions_df=inframe_deletions,
        cancer_count_type="All_Cancers",
        patient_total=patient_total,
    )

    inframe_deletions_count_per_cancer = (
        count_nested_inframe_deletions_per_cancer_type(
            inframe_deletions_df=inframe_deletions,
            per_cancer_patient_total=per_cancer_patient_total,
        )
    )

    inframe_deletions_with_counts = merge_inframe_deletions_with_counts(
        inframe_deletions_with_positions=inframe_deletions,
        inframe_deletions_count_all_cancers=inframe_deletions_count_all_cancers,
        inframe_deletions_count_per_cancer=inframe_deletions_count_per_cancer,
    )

    # Merge all the counts together
    one_row_per_variant_agg = one_row_per_variant_agg.join(
        merged_nt_counts,
        on="grch38_description",
        how="left",
    )

    one_row_per_variant_agg = one_row_per_variant_agg.join(
        merged_aa_counts,
        on=["Hugo_Symbol", "HGVSp"],
        how="left",
    )

    one_row_per_variant_agg = one_row_per_variant_agg.join(
        trunc_counts,
        on="grch38_description",
        how="left",
    )

    one_row_per_variant_agg = one_row_per_variant_agg.join(
        inframe_deletions_with_counts,
        on="grch38_description",
        how="left",
    )
    del (
        nucleotide_change_counts_all_cancer,
        nucleotide_change_counts_per_cancer,
        merged_nt_counts,
        amino_acid_change_counts_all_cancer,
        amino_acid_change_counts_per_cancer,
        merged_aa_counts,
        truncating_counts_all_cancers,
        truncating_counts_per_cancer,
        trunc_counts,
        inframe_deletions_count_all_cancers,
        inframe_deletions_count_per_cancer,
        inframe_deletions_with_counts,
    )
    gc.collect()

    if args.haemonc_cancer_types:
        print(
            f"{datetime.now().replace(microsecond=0)} Generating grouped"
            " haemonc cancer counts"
        )
        haemonc_rows = get_rows_for_cancer_types(
            df=genie_data,
            cancer_types=haemonc_cancers,
        )

        nucleotide_counts_haemonc_cancers = count_same_nucleotide_change(
            df=haemonc_rows,
            unique_patient_total=haemonc_cancer_patient_total,
            count_type="Haemonc_Cancers",
            all_variants_df=genie_data,
        )

        amino_acid_counts_haemonc_cancers = count_amino_acid_change(
            df=haemonc_rows,
            unique_patient_total=haemonc_cancer_patient_total,
            count_type="Haemonc_Cancers",
            all_variants_df=genie_data,
        )

        truncating_variants_haemonc = get_truncating_variants(df=haemonc_rows)
        truncating_variants_haemonc = extract_position_from_hgvsc(
            df=truncating_variants_haemonc
        )
        frameshift_counts_haemonc = count_frameshift_truncating_and_nonsense(
            df=truncating_variants_haemonc,
            patient_total=haemonc_cancer_patient_total,
            cancer_count_type="Haemonc_Cancers",
            truncating_variants=truncating_variants,
        )

        inframe_deletions_haemonc = get_inframe_deletions(
            df=haemonc_rows, column_used=deletion_source
        ).select(
            "Hugo_Symbol",
            "grch38_description",
            "Transcript_ID",
            deletion_source,
            "PATIENT_ID",
        )
        inframe_deletions_haemonc = add_deletion_positions(
            inframe_deletions_haemonc, source=deletion_source
        )
        inframe_deletions_count_haemonc_cancers = (
            count_nested_inframe_deletions(
                inframe_deletions_df=inframe_deletions_haemonc,
                cancer_count_type="Haemonc_Cancers",
                patient_total=haemonc_cancer_patient_total,
                inframe_deletions=inframe_deletions,
            )
        )

        all_ho_counts = nucleotide_counts_haemonc_cancers.join(
            amino_acid_counts_haemonc_cancers,
            on="grch38_description",
            how="left",
        )

        all_ho_counts = all_ho_counts.join(
            frameshift_counts_haemonc,
            on=["Hugo_Symbol", "grch38_description"],
            how="left",
        )

        all_ho_counts = all_ho_counts.join(
            inframe_deletions_count_haemonc_cancers,
            on=["Hugo_Symbol", "grch38_description"],
            how="left",
        )

        all_ho_counts = all_ho_counts.drop(
            "Hugo_Symbol",
            "HGVSp",
            "Transcript_ID",
            "del_start",
            "del_end",
            "CDS_position",
        )

        one_row_per_variant_agg = one_row_per_variant_agg.join(
            all_ho_counts, on="grch38_description", how="left"
        )
        del (
            haemonc_rows,
            nucleotide_counts_haemonc_cancers,
            amino_acid_counts_haemonc_cancers,
            frameshift_counts_haemonc,
            inframe_deletions_haemonc,
            inframe_deletions_count_haemonc_cancers,
            all_ho_counts,
        )
        gc.collect()

    if args.solid_cancer_types:
        print(
            f"{datetime.now().replace(microsecond=0)} Generating grouped"
            " solid cancer counts"
        )
        solid_rows = get_rows_for_cancer_types(
            df=genie_data,
            cancer_types=solid_cancers,
        )

        nucleotide_counts_solid_cancers = count_same_nucleotide_change(
            df=solid_rows,
            unique_patient_total=solid_cancer_patient_total,
            count_type="Solid_Cancers",
            all_variants_df=genie_data,
        )

        amino_acid_counts_solid_cancers = count_amino_acid_change(
            df=solid_rows,
            unique_patient_total=solid_cancer_patient_total,
            count_type="Solid_Cancers",
            all_variants_df=genie_data,
        )

        truncating_variants_solid = get_truncating_variants(df=solid_rows)
        truncating_variants_solid = extract_position_from_hgvsc(
            df=truncating_variants_solid
        )
        frameshift_counts_solid = count_frameshift_truncating_and_nonsense(
            df=truncating_variants_solid,
            patient_total=solid_cancer_patient_total,
            cancer_count_type="Solid_Cancers",
            truncating_variants=truncating_variants,
        )

        inframe_deletions_solid = get_inframe_deletions(
            df=solid_rows, column_used=deletion_source
        ).select(
            "Hugo_Symbol",
            "grch38_description",
            deletion_source,
            "PATIENT_ID",
        )
        inframe_deletions_solid = add_deletion_positions(
            inframe_deletions_solid, source=deletion_source
        )
        inframe_deletions_count_solid_cancers = count_nested_inframe_deletions(
            inframe_deletions_df=inframe_deletions_solid,
            cancer_count_type="Solid_Cancers",
            patient_total=solid_cancer_patient_total,
            inframe_deletions=inframe_deletions,
        )

        all_solid_counts = nucleotide_counts_solid_cancers.join(
            amino_acid_counts_solid_cancers,
            on="grch38_description",
            how="left",
        )

        all_solid_counts = all_solid_counts.join(
            frameshift_counts_solid,
            on=["Hugo_Symbol", "grch38_description"],
            how="left",
        )

        all_solid_counts = all_solid_counts.join(
            inframe_deletions_count_solid_cancers,
            on=["Hugo_Symbol", "grch38_description"],
            how="left",
        )

        all_solid_counts = all_solid_counts.drop(
            "Hugo_Symbol",
            "HGVSp",
            "del_start",
            "del_end",
            "CDS_position",
        )

        one_row_per_variant_agg = one_row_per_variant_agg.join(
            all_solid_counts, on="grch38_description", how="left"
        )

    print(
        f"{datetime.now().replace(microsecond=0)} Reformatting and writing"
        " output"
    )
    final_df = reorder_final_columns(
        one_row_per_variant_agg,
        patient_total,
        per_cancer_patient_total,
        haemonc_cancer_patient_total,
        solid_cancer_patient_total,
    )

    final_df.write_csv(
        args.output,
        separator="\t",
    )


if __name__ == "__main__":
    main()
