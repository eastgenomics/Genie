import argparse
import numpy as np
import pandas as pd
import pysam

from utils.file_io import read_in_to_df


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments

    Returns
    -------
    args : Namespace
        Namespace of passed command line argument inputs
    """
    parser = argparse.ArgumentParser(
        description=(
            "Information required to merge liftover and VEP annotations with"
            " GENIE data"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help=(
            "Path to GENIE MAF file which includes patient and sample"
            " information"
        ),
    )

    parser.add_argument(
        "--vep_vcf",
        required=True,
        type=str,
        help=(
            "Path to VCF of each variant (GRCh38) annotated with VEP with link"
            " back to its GENIE MAF representation in the Genie_description"
            " INFO field"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Name of output MAF file with annotations modified",
    )

    parser.add_argument(
        "--output_all",
        required=True,
        type=str,
        help=(
            "Name of output MAF file with both original GENIE annotations and"
            " new VEP annotations"
        ),
    )

    return parser.parse_args()


def add_unique_variant_field_to_maf(genie_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column to the GENIE data which is a unique description of each variant.

    Parameters
    ----------
    genie_data : pd.DataFrame
        DataFrame containing GENIE data with columns for Chromosome,
        Start_Position, Reference_Allele, and Tumor_Seq_Allele2.

    Returns
    -------
    pd.DataFrame
        DataFrame with new column 'Genie_description' that
        uniquely describes each variant and original columns removed
    """
    genie_data["Genie_description"] = (
        genie_data["Chromosome"].astype(str)
        + "_"
        + genie_data["Start_Position"].astype(str)
        + "_"
        + genie_data["Reference_Allele"].astype(str)
        + "_"
        + genie_data["Tumor_Seq_Allele2"].astype(str)
    )

    genie_data.drop(
        columns=[
            "Chromosome",
            "Start_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
        ],
        inplace=True,
    )

    return genie_data


def read_annotated_vcf_to_df(vcf_file: str) -> pd.DataFrame:
    """
    Read in VCF annotated by VEP to a dataframe.

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file to read in.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the relevant variants and INFO fields from the
        VCF file.
    """
    vcf_in = pysam.VariantFile(vcf_file, "r")
    try:
        records = []
        for record in vcf_in:
            records.append(
                {
                    "CHROM": str(record.chrom),
                    "POS": int(record.pos),
                    "REF": str(record.ref),
                    "ALT": ",".join(str(a) for a in record.alts),
                    "Genie_description": record.info.get("Genie_description"),
                    "OriginalContig": record.info.get("OriginalContig"),
                    "OriginalStart": record.info.get("OriginalStart"),
                    "SwappedAlleles": record.info.get("SwappedAlleles"),
                    "VEP_SYMBOL": record.info.get("CSQ_SYMBOL"),
                    "VEP_Consequence": record.info.get("CSQ_Consequence"),
                    "VEP_Feature": record.info.get("CSQ_Feature"),
                    "VEP_HGVSc": record.info.get("CSQ_HGVSc"),
                    "VEP_HGVSp": record.info.get("CSQ_HGVSp"),
                    "VEP_EXON": record.info.get("CSQ_EXON"),
                    "VEP_INTRON": record.info.get("CSQ_INTRON"),
                    "VEP_CDS_position": record.info.get("CSQ_CDS_position"),
                    "VEP_Protein_position": record.info.get(
                        "CSQ_Protein_position"
                    ),
                    "VEP_CANONICAL": record.info.get("CSQ_CANONICAL"),
                    "VEP_MANE": record.info.get("CSQ_MANE"),
                    "VEP_MANE_SELECT": record.info.get("CSQ_MANE_SELECT"),
                    "VEP_MANE_PLUS_CLINICAL": record.info.get(
                        "CSQ_MANE_PLUS_CLINICAL"
                    ),
                }
            )
    finally:
        vcf_in.close()

    return pd.DataFrame(records)


def format_fields(vcf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the fields from the VCF

    Parameters
    ----------
    vcf_df : pd.DataFrame
        Dataframe of info from the annotated VCF

    Returns
    -------
    pd.DataFrame
        Dataframe with new field grch37_norm and VEP_p, with additional fields
        added from the lfitover removed
    """
    # If the alleles were swapped during liftover (indicated by the
    # SwappedAlleles INFO field), take the alt allele in GRCh37 as the ref
    # allele and vice versa
    swapped = vcf_df["SwappedAlleles"].fillna(False).astype(bool)
    vcf_df["ref_grch37"] = np.where(swapped, vcf_df["ALT"], vcf_df["REF"])
    vcf_df["alt_grch37"] = np.where(swapped, vcf_df["REF"], vcf_df["ALT"])

    # Remove the 'chr' prefix from the chromosome names
    vcf_df["OriginalContig"] = vcf_df["OriginalContig"].str.replace("chr", "")
    # Create a unique GRCh37 identifier for the GRCh37 normalised variant
    vcf_df["grch37_norm"] = (
        vcf_df["OriginalContig"].astype(str)
        + "_"
        + vcf_df["OriginalStart"].astype(str)
        + "_"
        + vcf_df["ref_grch37"].astype(str)
        + "_"
        + vcf_df["alt_grch37"].astype(str)
    )

    # Pysam gives tuples for the VEP annotations for some reason so just
    # get the value
    cols = [
        "VEP_SYMBOL",
        "VEP_Consequence",
        "VEP_Feature",
        "VEP_HGVSc",
        "VEP_HGVSp",
        "VEP_EXON",
        "VEP_INTRON",
        "VEP_CDS_position",
        "VEP_Protein_position",
        "VEP_CANONICAL",
        "VEP_MANE",
        "VEP_MANE_SELECT",
        "VEP_MANE_PLUS_CLINICAL",
    ]

    # Replace any '.' with NaN
    vcf_df[cols] = (
        vcf_df[cols]
        .map(lambda x: x[0] if isinstance(x, tuple) and len(x) == 1 else x)
        .replace(".", np.nan)
    )

    # Split out just the p. from the HGVSp and replace URL encoding for '='
    vcf_df["VEP_p"] = (
        vcf_df["VEP_HGVSp"]
        .astype(str)
        .str.split(":", n=1)
        .str[1]
        .str.replace("%3D", "=", regex=False)
    )

    vcf_df.drop(
        ["OriginalContig", "OriginalStart", "SwappedAlleles"],
        axis=1,
        inplace=True,
    )

    return vcf_df


def remove_rows_with_no_liftover(
    merged_df: pd.DataFrame,
    b37_genie_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove any rows which have no GRCh38 liftover data

    Parameters
    ----------
    merged_df : pd.DataFrame
        GENIE data merged with VCF annotation data
    b37_genie_data : pd.DataFrame
        Original GENIE MAF data

    Returns
    -------
    pd.DataFrame
        GENIE data with any rows which did not liftover to GRCh38 removed
    """
    # Find any rows with no liftover information
    no_liftover = (
        (merged_df["CHROM"].isna() | (merged_df["CHROM"] == ""))
        | (merged_df["POS"].isna() | (merged_df["POS"] == ""))
        | (merged_df["REF"].isna() | (merged_df["REF"] == ""))
        | (merged_df["ALT"].isna() | (merged_df["ALT"] == ""))
        | (merged_df["grch37_norm"].isna() | (merged_df["grch37_norm"] == ""))
    )
    no_liftover_rows = merged_df[no_liftover]
    if len(no_liftover_rows) > 0:
        print(
            f"Warning: {len(no_liftover_rows)} rows which represent"
            f" {no_liftover_rows['Genie_description'].nunique()} unique"
            " variants do not have GRCh38 liftover. These will be written out"
            " to no_liftover.tsv but not written to the output file."
        )
        no_liftover_counts = (
            no_liftover_rows["Genie_description"].value_counts().reset_index()
        )
        no_liftover_counts.to_csv(
            "no_liftover.tsv",
            sep="\t",
            header=["Genie_description", "Number_of_rows"],
            index=False,
        )
    else:
        print("All rows have GRCh38 liftover information.")

    # Keep only rows with liftover information
    liftover_rows = merged_df[~no_liftover].copy()
    liftover_rows.loc[:, "POS"] = liftover_rows["POS"].astype("Int64")

    if len(b37_genie_data) != len(liftover_rows):
        print(
            "Warning: The number of rows in the original Genie data"
            f" {len(b37_genie_data)} does not match the number of rows"
            f" in the merged data with liftover: {len(liftover_rows)}"
        )

    liftover_rows.loc[:, "grch38_description"] = (
        liftover_rows["CHROM"].astype(str)
        + "_"
        + liftover_rows["POS"].apply(lambda x: str(int(x)))
        + "_"
        + liftover_rows["REF"].astype(str)
        + "_"
        + liftover_rows["ALT"].astype(str)
    )

    return liftover_rows


def replace_annotations(genie_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Replace specific original GENIE annotations with the new VEP annotations
    against MANE transcripts

    Parameters
    ----------
    genie_df : pd.DataFrame
        GENIE data with original and new VEP annotations

    Returns
    -------
    pd.DataFrame
        GENIE data with only the new VEP annotations retained for certain
        fields but renamed to match original GENIE fields
    """
    genie_merged.drop(
        columns=[
            "Hugo_Symbol",
            "Consequence",
            "Transcript_ID",
            "RefSeq",
            "HGVSc",
            "HGVSp",
            "Protein_position",
            "Exon_Number",
            "Variant_Classification",
            "CHROM",
            "POS",
            "REF",
            "ALT",
            "ref_grch37",
            "alt_grch37",
            "VEP_INTRON",
        ],
        inplace=True,
    )

    genie_merged.rename(
        columns={
            "VEP_SYMBOL": "Hugo_Symbol",
            "VEP_Consequence": "Consequence",
            "VEP_Feature": "RefSeq",
            "VEP_HGVSc": "HGVSc",
            "VEP_HGVSp": "HGVSp",
            "VEP_EXON": "Exon_Number",
            "VEP_CDS_position": "CDS_position",
            "VEP_Protein_position": "Protein_position",
            "VEP_CANONICAL": "CANONICAL",
            "VEP_MANE": "MANE",
            "VEP_MANE_SELECT": "MANE_SELECT",
            "VEP_MANE_PLUS_CLINICAL": "MANE_PLUS_CLINICAL",
        },
        inplace=True,
    )

    return genie_merged


def main():
    args = parse_args()
    genie_data_with_sample_info = read_in_to_df(
        args.input,
        header=0,
        dtype={
            "Entrez_Gene_Id": "Int64",
            "Start_Position": "Int64",
            "AGE_AT_SEQ_REPORT": "string",
        },
        converters={
            col: lambda x: x.strip() if isinstance(x, str) else x
            for col in ["Chromosome", "Reference_Allele", "Tumor_Seq_Allele2"]
        },
    )

    genie_data_with_sample_info = add_unique_variant_field_to_maf(
        genie_data_with_sample_info
    )
    vcf_df = read_annotated_vcf_to_df(args.vep_vcf)
    vcf_df = format_fields(vcf_df)

    merged = pd.merge(
        genie_data_with_sample_info,
        vcf_df,
        on="Genie_description",
        how="left",
    )
    merged = remove_rows_with_no_liftover(merged, genie_data_with_sample_info)

    print(
        "Writing output with original GENIE annotations and new VEP"
        f" annotations to {args.output_all}"
    )
    merged.to_csv(
        args.output_all,
        sep="\t",
        index=False,
    )
    merged = replace_annotations(merged)
    print(f"Writing output with replaced VEP annotations to {args.output}")
    merged.to_csv(
        args.output,
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
