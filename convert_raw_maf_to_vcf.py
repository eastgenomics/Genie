import argparse
import pandas as pd
import pysam
import re

from tqdm import tqdm
from utils.file_io import read_in_to_df, read_in_fasta


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
            "Information required to convert variants to VCF description"
            " and write out as VCF file"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to MAF file",
    )

    parser.add_argument(
        "--fasta",
        required=True,
        type=str,
        help="Path to FASTA file used to get ref alleles",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Name of output VCF file",
    )

    return parser.parse_args()


def get_unique_variant_rows(genie_data):
    """
    Get unique variant rows from Genie data based on chrom, pos, ref and alt

    Parameters
    ----------
    genie_data : pd.DataFrame
        Genie data with one row per variant and count info

    Returns
    -------
    genie_data_unique : pd.DataFrame
        Unique rows of Genie data based on chrom, pos, ref and alt
    """
    # Get unique rows based on chrom, pos, ref and alt
    genie_data_unique = genie_data.drop_duplicates(
        subset=[
            "Chromosome",
            "Start_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
        ],
        keep="first",
    )
    return genie_data_unique


def convert_maf_like_variant_to_vcf_description(row, fasta):
    """
    Convert a row of data from MAF-like format to VCF-like format

    Parameters
    ----------
    row : pd.Series
        Row of Genie data to convert
    fasta : pysam.FastaFile
        FASTA file as pysam object

    Returns
    -------
    vcf_chrom : str
        CHROM in VCF-like format
    vcf_pos : int
        POS in VCF-like format
    vcf_ref : str
        REF allele in VCF-like format
    vcf_alt : str
        ALT allele in VCF-like format
    """
    # Set explicit datatypes to avoid issues when querying
    chrom, pos, ref, alt = (
        str(row["Chromosome"]),
        int(row["Start_Position"]),
        str(row["Reference_Allele"]),
        str(row["Tumor_Seq_Allele2"]),
    )

    # Set VCF-like values to original by default
    vcf_chrom, vcf_pos, vcf_ref, vcf_alt = chrom, pos, ref, alt

    # Replace ref and alt '-' or other weird chars with ""
    ref = "" if re.fullmatch(r"[-?0]+", ref) else ref
    alt = "" if re.fullmatch(r"[-?0]+", alt) else alt

    # MAF coords are 1-based, pysam uses 0-based half-open intervals and
    # for deletions we want the base before the start, so needs pos - 2
    # https://pysam.readthedocs.io/en/latest/api.html#pysam.FastaFile.fetch
    ref_seq = ""
    ref_seq = fasta.fetch(
        chrom, max(0, pos - 2), pos - 1 + max(1, len(ref))
    ).upper()
    if not ref_seq:
        print(
            "No reference sequence found in FASTA for"
            f" {chrom}-{pos}-{ref}-{alt}"
        )

    # It's a simple indel
    if (len(ref) == 0) or (len(alt) == 0):
        prefix_bp = ref_seq[0]
        # For simple insertions, pos is already position of preceding bp
        if ref == "" and len(ref_seq) > 0:
            prefix_bp = ref_seq[1]
            vcf_ref = prefix_bp
            vcf_pos = pos
            vcf_alt = prefix_bp + alt
        # It's a simple deletion - we need to remove 1 from position
        else:
            vcf_ref = prefix_bp + ref
            vcf_alt = prefix_bp
            vcf_pos = pos - 1

    # For non-indels, verify ref allele
    else:
        ref_from_fasta = ref_seq[1 : len(ref) + 1]
        if ref != ref_from_fasta:
            print(
                f"Reference mismatch for {chrom}-{pos}-{ref}-{alt} -"
                f" {ref_from_fasta} found in FASTA. Removing variant."
            )
            return None

    return vcf_chrom, vcf_pos, vcf_ref, vcf_alt


def convert_to_vcf_representation(genie_data, fasta):
    """
    Make new columns with the VCF-like description for chrom, pos,
    ref and alt in GRCh37

    Parameters
    ----------
    genie_data : pd.DataFrame
        Genie data with one row per variant
    fasta : pysam.FastaFile
        FASTA file as pysam object

    Returns
    -------
    genie_count_data : pd.DataFrame
        Genie data with one row per variant and count info, with new columns
        chrom_vcf, pos_vcf, ref_vcf and alt_vcf
    """
    tqdm.pandas(desc="Converting to VCF representation")

    def conv_or_none(row):
        res = convert_maf_like_variant_to_vcf_description(row, fasta)
        if res is None:
            return pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])
        return pd.Series(res)

    genie_data[["chrom_vcf", "pos_vcf", "ref_vcf", "alt_vcf"]] = (
        genie_data.progress_apply(conv_or_none, axis=1)
    )

    return genie_data


def write_variants_to_vcf(genie_vcf_description, output_vcf, fasta):
    """
    Write out variants in the the dataframe to a VCF file

    Parameters
    ----------
    genie_vcf_description : pd.DataFrame
        Dataframe with one row per variant and columns with VCF-like
        description
    output_vcf : str
        Name of output VCF file
    fasta : pysam.FastaFile
        FASTA file as pysam object
    """
    header = pysam.VariantHeader()
    header.add_line("##fileformat=VCFv4.2")
    for contig in fasta.references:
        header.add_line(
            f"##contig=<ID={contig},length={fasta.get_reference_length(contig)}>"
        )
    header.info.add(
        "Genie_description",
        "1",
        "String",
        "Original variant description from Genie",
    )
    header.info.add(
        "Transcript_ID", "1", "String", "Ensembl Transcript ID from Genie"
    )

    vcf_out = pysam.VariantFile(output_vcf, "w", header=header)
    # For each original variant, write new variant record
    # Take 1 away from start due to differences in Pysam representation
    print("Writing variants to VCF file...")
    for row in tqdm(
        genie_vcf_description.itertuples(index=False),
        total=genie_vcf_description.shape[0],
    ):
        # Make sure there wasn't a ref mismatch
        if any(
            pd.isna(getattr(row, attr))
            for attr in ["chrom_vcf", "pos_vcf", "ref_vcf", "alt_vcf"]
        ):
            continue

        info_fields = {}
        # Add in original Genie GRCh37 chrom-pos-ref-alt
        orig_coord_str = (
            f"{getattr(row, 'Chromosome')}_{getattr(row, 'Start_Position')}_{getattr(row, 'Reference_Allele')}_{getattr(row, 'Tumor_Seq_Allele2')}"
        )
        info_fields["Genie_description"] = orig_coord_str
        transcript = getattr(row, "Transcript_ID", None)
        info_fields["Transcript_ID"] = (
            "."
            if (
                transcript is None
                or pd.isna(transcript)
                or str(transcript).strip() in {"", "."}
            )
            else str(transcript).strip()
        )

        record = vcf_out.new_record(
            contig=str(getattr(row, "chrom_vcf")),
            start=int(getattr(row, "pos_vcf")) - 1,
            alleles=(getattr(row, "ref_vcf"), getattr(row, "alt_vcf")),
            id=".",
            qual=None,
            filter=None,
            info=info_fields,
        )

        vcf_out.write(record)

    vcf_out.close()
    fasta.close()


def main():
    args = parse_args()
    genie_data = read_in_to_df(
        args.input,
        sep="\t",
        header=0,
        dtype={
            "Entrez_Gene_Id": "Int64",
            "Start_Position": "Int64",
        },
        converters={
            col: lambda x: x.strip() if isinstance(x, str) else x
            for col in ["Chromosome", "Reference_Allele", "Tumor_Seq_Allele2"]
        },
    )
    unique_genie_data = get_unique_variant_rows(genie_data)
    fasta = read_in_fasta(args.fasta)
    genie_vcf_desc = convert_to_vcf_representation(unique_genie_data, fasta)
    write_variants_to_vcf(genie_vcf_desc, args.output, fasta)


if __name__ == "__main__":
    main()
