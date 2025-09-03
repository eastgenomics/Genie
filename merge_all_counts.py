import pandas as pd

from utils.file_io import read_in_to_df, read_txt_file_to_list
from utils.dtypes import column_dtypes

from utils.aggregation import create_df_with_one_row_per_variant


def safe_merge_by_gene(
    big_df, small_df, on=["Hugo_Symbol", "grch38_description"], how="left"
):
    """
    Memory-friendly merge: splits by Hugo_Symbol before merging.
    """
    out_chunks = []

    small_genes = set(small_df["Hugo_Symbol"].unique())
    for gene in small_genes:
        big_chunk = big_df.loc[big_df["Hugo_Symbol"] == gene]
        if big_chunk.empty:
            continue
        small_chunk = small_df.loc[small_df["Hugo_Symbol"] == gene]
        merged = pd.merge(big_chunk, small_chunk, on=on, how=how)
        out_chunks.append(merged)

    # add rows for genes not in small_df (no merge needed, just keep them as-is)
    other_genes = big_df.loc[~big_df["Hugo_Symbol"].isin(small_genes)]
    if not other_genes.empty:
        out_chunks.append(other_genes)

    return pd.concat(out_chunks, axis=0, ignore_index=True)


def main():
    genie_data = read_in_to_df(
        "data_mutations_extended_clinical_info_GRCh38_fixed.txt",
        header=0,
        dtype=column_dtypes,
    )

    columns_to_aggregate = read_txt_file_to_list("columns_to_aggregate.txt")

    one_row_per_variant_agg = create_df_with_one_row_per_variant(
        df=genie_data,
        columns_to_aggregate=columns_to_aggregate,
    )
    nt_counts = read_in_to_df("nucleotide_counts.tsv")
    aa_counts = read_in_to_df("amino_acid_counts.tsv")
    frameshift_counts = read_in_to_df("frameshift_counts.tsv")
    inframe_counts = read_in_to_df("inframe_counts.tsv")
