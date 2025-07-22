import argparse
import os
import pandas as pd
import pysam
import re
import sys

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
        description=(
            "Information required to convert variants to VCF description"
            " and write out as TSV and VCF files"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to file with variant counts (one row per variant)",
    )

    parser.add_argument(
        "--fasta",
        required=True,
        type=str,
        help="Path to FASTA file used to get ref alleles",
    )

    parser.add_argument(
        "--output_vcf",
        required=True,
        type=str,
        help="Name of output VCF file",
    )

    return parser.parse_args()

    # genie_count_data["Consequence"] = genie_count_data[
    #     "Consequence"
    # ].str.replace(",", "&")


def read_in_fasta(filename):
    """
    Read in FASTA to pysam.FastaFile object

    Parameters
    ----------
    filename : str
        path to FASTA file

    Returns
    -------
    fasta : pysam.FastaFile
        FASTA file as pysam object
    Raises
    ------
    FileNotFoundError
        If FASTA file does not exist
    ValueError
        If FASTA file is not in the correct format
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FASTA file not found: {filename}")
    try:
        fasta = pysam.FastaFile(filename)
        return fasta
    except ValueError as err:
        raise ValueError(f"Invalid FASTA format: {err}") from err


def remove_disallowed_chars_from_columns(genie_counts):
    """
    Split out the chromosome, position, reference, and alternate alleles
    from the input dataframe into separate columns
    """
    # Convert . to underscore in column names
    genie_counts.columns = genie_counts.columns.str.replace(
        r"\.", "_", regex=True
    )

    # Remove commas, hyphens, whitespaces, and slashes from column names
    # to make them compatible with VCF format
    genie_counts.columns = genie_counts.columns.str.replace(
        r"[\/\s,-]", "", regex=True
    )

    return genie_counts


def camel_case_to_spaces(text):
    """
    Replace camel case in a string with spaces, while preserving 'CDS' as a whole word.

    Parameters
    ----------
    text : str
        Input string with camel case

    Returns
    -------
    str
        Input string with spaces instead of camel case
    """
    # Step 1: Protect 'CDS' with a placeholder
    placeholder = "___cds___"
    text = text.replace("CDS", placeholder)

    # Step 2: Insert spaces between camel case words
    text = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)

    # Step 3: Restore 'CDS' and ensure space before it
    text = text.replace(placeholder, "CDS")
    text = re.sub(
        r"(?<! )CDS", r" CDS", text
    )  # Add space before CDS if not already there

    return text


def generate_info_fields(genie_counts):
    """
    Generate INFO fields for the VCF file based on the genie_counts DataFrame.

    Parameters
    ----------
    genie_counts : pd.DataFrame
        DataFrame containing variant counts and descriptions

    Returns
    -------
    info_fields : list
        List of dictionaries with INFO field names and descriptions
    """
    info_fields = []
    for column in genie_counts.columns:
        if "count" in column:
            count_type_description = camel_case_to_spaces(column.split("_")[0])
            cancer_type_description = camel_case_to_spaces(
                column.split("_")[1]
            )
            info_fields.append(
                {
                    "id": column,
                    "number": 1,
                    "type": "Integer",
                    "description": (
                        f"{count_type_description} count in"
                        f" {cancer_type_description} cancer"
                    ),
                }
            )
        else:
            info_fields.append(
                {
                    "id": column,
                    "number": 1,
                    "type": "String",
                    "description": f"{column} from Genie data",
                }
            )

    return info_fields


def write_variants_to_vcf(genie_counts, output_vcf, fasta, info_fields):
    """
    Write out the dataframe (with VCF-like description) to a VCF file
    with the INFO fields specified in the JSON file

    Parameters
    ----------
    genie_counts_vcf_description : pd.DataFrame
        Dataframe with one row per variant and count info and columns with
        VCF-like description
    output_vcf : str
        Name of output VCF file
    fasta : pysam.FastaFile
        FASTA file as pysam object
    info_fields : list
        List of dictionaries with INFO field names and descriptions
        to be added to the VCF file
    """
    header = pysam.VariantHeader()
    header.add_line("##fileformat=VCFv4.2")
    for contig in fasta.references:
        header.add_line(
            f"##contig=<ID={contig},length={fasta.get_reference_length(contig)}>"
        )
    for field in info_fields:
        header.info.add(
            field["id"], field["number"], field["type"], field["description"]
        )

    vcf_out = pysam.VariantFile(output_vcf, "w", header=header)
    # For each original variant, write new variant record with all INFO
    # fields specified in the JSON file
    # Take 1 away from start due to differences in Pysam representation
    for _, row in genie_counts.iterrows():
        chrom, pos, ref, alt = row["grch38_description"].split("_")
        record = vcf_out.new_record(
            contig=str(chrom),
            start=int(pos) - 1,
            alleles=(ref, alt),
            id=".",
            qual=None,
            filter=None,
        )
        for field in info_fields:
            field_name = field["id"]
            if field_name in row:
                if pd.notna(row[field_name]):
                    field_type = field.get("type")
                    type_map = {
                        "String": str,
                        "Character": str,
                        "Integer": int,
                        "Float": float,
                        "Flag": lambda x: x,
                    }
                    converter = type_map.get(field_type)
                    if converter:
                        try:
                            field_value = converter(row[field_name])
                            record.info[field["id"]] = field_value
                        except Exception as err:
                            print(
                                f"Error converting field {field_name}: {err}"
                            )
                    else:
                        print(
                            f"Unsupported field type for field {field_name}: "
                            f"{field_type}. Skipping"
                        )

        vcf_out.write(record)

    vcf_out.close()


def main():
    args = parse_args()
    cols = list(
        pd.read_csv("Genie_v17_GRCh38_aggregated.tsv", sep="\t", nrows=1)
    )
    genie_counts = read_in_to_df(
        args.input,
        sep="\t",
        header=0,
        usecols=[i for i in cols if i != "Entrez_Gene_Id"],
        dtype={
            "Start_Position": "Int64",
        },
    )
    fasta = read_in_fasta(args.fasta)
    genie_counts_renamed = remove_disallowed_chars_from_columns(genie_counts)
    info_fields = generate_info_fields(genie_counts_renamed)
    write_variants_to_vcf(
        genie_counts_renamed, args.output_vcf, fasta, info_fields
    )


if __name__ == "__main__":
    main()
