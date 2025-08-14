import argparse
import numpy as np
import os
import pandas as pd
import pysam

from utils import read_in_to_df


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns
    -------
    args : Namespace
        Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser(
        description="Information required to add GRCh38 liftover information"
    )
    parser.add_argument(
        "--genie_clinical",
        required=True,
        type=str,
        help=(
            "Path to TSV file of Genie data (with clinical info) to add "
            "GRCh38 liftover info to"
        ),
    )

    parser.add_argument(
        "--vcf",
        required=True,
        type=str,
        help=(
            "Path to VCF file which has been lifted over to GRCh38 which "
            "includes original Genie variant description"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Name of output TSV file with GRCh38 liftover info added",
    )

    return parser.parse_args()


def add_unique_variant_field(genie_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a field to the Genie data which is a unique description of the variant.

    Parameters
    ----------
    genie_data : pd.DataFrame
        DataFrame containing Genie data with columns for Chromosome,
        Start_Position, Reference_Allele, and Tumor_Seq_Allele2.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional column 'variant_description' that
        uniquely describes each variant.
    """
    genie_data["variant_description"] = (
        genie_data["Chromosome"].astype(str)
        + "_"
        + genie_data["Start_Position"].astype(str)
        + "_"
        + genie_data["Reference_Allele"].astype(str)
        + "_"
        + genie_data["Tumor_Seq_Allele2"].astype(str)
    )

    return genie_data


def read_vcf_to_df(vcf_file: str) -> pd.DataFrame:
    """
    Read in VCF file to a dataframe.

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file to read in.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the relevant columns from the VCF file.
    """
    vcf_in = pysam.VariantFile(vcf_file, "r")

    # Store required data from GRCh38 VCF in a list of dictionaries then
    # convert to a DataFrame
    records = []
    for record in vcf_in:
        row = {
            "CHROM": str(record.chrom),
            "POS": int(record.pos),
            "REF": str(record.ref),
            "ALT": ",".join(str(a) for a in record.alts),
            "Genie_description": record.info.get("Genie_description", None),
            "OriginalContig": record.info.get("OriginalContig", None),
            "OriginalStart": record.info.get("OriginalStart", None),
            "SwappedAlleles": record.info.get("SwappedAlleles", None),
        }

        records.append(row)

    vcf_df = pd.DataFrame(records)

    vcf_df = vcf_df.rename(
        columns={
            "CHROM": "chrom_grch38",
            "POS": "pos_grch38",
            "REF": "ref_grch38",
            "ALT": "alt_grch38",
            "OriginalContig": "chrom_grch37",
            "OriginalStart": "pos_grch37",
        }
    )

    # If the alleles were swapped during liftover (indicated by the
    # SwappedAlleles INFO field), take the alt allele in GRCh37 as the ref
    # allele and vice versa
    vcf_df["ref_grch37"] = np.where(
        vcf_df["SwappedAlleles"], vcf_df["alt_grch38"], vcf_df["ref_grch38"]
    )
    vcf_df["alt_grch37"] = np.where(
        vcf_df["SwappedAlleles"], vcf_df["ref_grch38"], vcf_df["alt_grch38"]
    )

    # Remove the 'chr' prefix from the chromosome names
    vcf_df["chrom_grch37"] = vcf_df["chrom_grch37"].str.replace("chr", "")
    # Create a unique GRCh37 identifier for the GRCh37 normalised variant
    vcf_df["grch37_norm"] = (
        vcf_df["chrom_grch37"].astype(str)
        + "_"
        + vcf_df["pos_grch37"].astype(str)
        + "_"
        + vcf_df["ref_grch37"].astype(str)
        + "_"
        + vcf_df["alt_grch37"].astype(str)
    )
    # Reorder columns
    vcf_df = vcf_df[
        [
            "chrom_grch38",
            "pos_grch38",
            "ref_grch38",
            "alt_grch38",
            "Genie_description",
            "grch37_norm",
        ]
    ]

    return vcf_df


def merge_dataframes(
    b37_genie_data: pd.DataFrame, vcf_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge two dataframes on common columns.

    Parameters
    ----------
    b37_genie_data : pd.DataFrame
        DataFrame containing the GRCh37 Genie data
    vcf_df : pd.DataFrame
        DataFrame containing the VCF data with GRCh38 liftover information

    Returns
    -------
    liftover_rows : pd.DataFrame
        Merged DataFrame for rows where the GRCh38 liftover information has
        been added
    """
    # Merge Genie data with the GRCh38 liftover info
    merged_df = pd.merge(
        b37_genie_data,
        vcf_df,
        left_on="variant_description",
        right_on="Genie_description",
        how="left",
    )

    # Find any rows with no liftover information
    no_liftover = (
        (merged_df["chrom_grch38"].isna() | (merged_df["chrom_grch38"] == ""))
        | (merged_df["pos_grch38"].isna() | (merged_df["pos_grch38"] == ""))
        | (merged_df["ref_grch38"].isna() | (merged_df["ref_grch38"] == ""))
        | (merged_df["alt_grch38"].isna() | (merged_df["alt_grch38"] == ""))
        | (merged_df["grch37_norm"].isna() | (merged_df["grch37_norm"] == ""))
    )
    no_liftover_rows = merged_df[no_liftover]
    if len(no_liftover_rows) > 0:
        print(
            f"Warning: {len(no_liftover_rows)} rows which represent"
            f" {no_liftover_rows['variant_description'].nunique()} unique"
            " variants do not have GRCh38 liftover. These will be written out"
            " to no_liftover.tsv but not written to the output file."
        )
        no_liftover_counts = (
            no_liftover_rows["variant_description"]
            .value_counts()
            .reset_index()
        )
        no_liftover_counts.to_csv(
            "no_liftover.tsv",
            sep="\t",
            header=["Genie_Description", "Number_of_rows"],
            index=False,
        )
    else:
        print("All rows have GRCh38 liftover information.")

    # Keep only rows with liftover information
    liftover_rows = merged_df[~no_liftover]
    liftover_rows["pos_grch38"] = liftover_rows["pos_grch38"].astype(int)

    if len(b37_genie_data) != len(liftover_rows):
        print(
            "Warning: The number of rows in the original Genie data"
            f" {len(b37_genie_data)} does not match the number of rows"
            f" in the merged data with liftover: {len(liftover_rows)}"
        )

    liftover_rows["grch38_description"] = (
        liftover_rows["chrom_grch38"].astype(str)
        + "_"
        + liftover_rows["pos_grch38"].astype(str)
        + "_"
        + liftover_rows["ref_grch38"].astype(str)
        + "_"
        + liftover_rows["alt_grch38"].astype(str)
    )

    return liftover_rows


def main():
    args = parse_args()
    # Read in Genie data, stripping any whitespace from some fields
    genie_data_with_sample_info = read_in_to_df(
        args.genie_clinical,
        header=0,
        dtype={
            "Chromosome": "str",
            "Entrez_Gene_Id": "Int64",
            "Start_Position": "Int64",
        },
        converters={
            col: lambda x: x.strip() if isinstance(x, str) else x
            for col in ["Chromosome", "Reference_Allele", "Tumor_Seq_Allele2"]
        },
    )
    # Add field so we can match each variant with the Genie description
    # INFO field in the VCF file
    genie_data_sample_info_unique_key = add_unique_variant_field(
        genie_data_with_sample_info
    )

    if not os.path.exists(args.vcf):
        raise FileNotFoundError(f"VCF file {args.vcf} does not exist.")
    vcf_df = read_vcf_to_df(args.vcf)
    merged_df = merge_dataframes(genie_data_sample_info_unique_key, vcf_df)
    merged_df.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
