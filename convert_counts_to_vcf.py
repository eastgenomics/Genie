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
        "--output",
        required=True,
        type=str,
        help="Name of output VCF file",
    )

    return parser.parse_args()


def remove_disallowed_chars_from_columns(genie_data):
    """
    Remove characters that are not allowed in VCF INFO field names and values.

    Parameters
    ----------
    genie_data : pd.DataFrame
        DataFrame containing variant counts and descriptions

    Returns
    -------
    genie_counts : pd.DataFrame
        DataFrame with modified column names and values for VCF compatibility
    """
    # Convert . to underscore in column names
    genie_data.columns = genie_data.columns.str.replace(r"\.", "_", regex=True)

    # Remove commas, hyphens, whitespaces, and slashes from column names
    # to make them compatible with VCF format
    genie_data.columns = genie_data.columns.str.replace(
        r"[\/\s,-]", "", regex=True
    )

    if "Consequence" in genie_data.columns:
        # Replace commas with '&' in the 'Consequence' column
        # to make it compatible with VCF format
        genie_data["Consequence"] = genie_data["Consequence"].str.replace(
            ",", "&"
        )

    return genie_data


def camel_case_to_spaces(text):
    """
    Replace camel case in a string with spaces, while preserving 'CDS' as a whole word

    Parameters
    ----------
    text : str
        Input string with camel case

    Returns
    -------
    str
        String with spaces instead of camel case
    """
    # Protect 'CDS' with a placeholder
    text = text.replace("CDS", "___cds___")

    # Insert spaces between camel case words
    text = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)

    # Restore 'CDS' and ensure space before it
    text = text.replace("___cds___", "CDS")
    text = re.sub(r"(?<! )CDS", r" CDS", text)

    return text


def generate_info_field_header_info(genie_counts):
    """
    Generate INFO field headers for the VCF file based on the genie_counts DataFrame

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
        # If it's a count, we want to add it as an int and write which
        # count type it is and whether all cancers or specific cancer type
        if "count" in column.lower():
            parts = column.split("_")
            # Skip malformed columns
            if len(parts) < 2:
                print("Skipping malformed column:", column)
                continue
            count_type_description = camel_case_to_spaces(parts[0])
            cancer_type_description = camel_case_to_spaces(parts[1]).replace(
                "Total", "All Cancers"
            )
            info_fields.append(
                {
                    "id": column,
                    "number": 1,
                    "type": "Integer",
                    "description": (
                        f"Number of patients with {count_type_description} in"
                        f" {cancer_type_description}"
                    ),
                }
            )
        # If it's the Genie description or grch37_norm then write what
        # these actually are
        elif column == "Genie_description":
            info_fields.append(
                {
                    "id": "Genie_description",
                    "number": 1,
                    "type": "String",
                    "description": (
                        "Original GRCh37 Genie description(s) of the variant"
                    ),
                }
            )
        elif column == "grch37_norm":
            info_fields.append(
                {
                    "id": "grch37_norm",
                    "number": 1,
                    "type": "String",
                    "description": (
                        "GRCh37 normalized description of the variant"
                    ),
                }
            )
        # Skip this as it's already in CHROM, POS, REF, ALT
        elif column == "grch38_description":
            continue
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


def create_field_converter_dict(info_fields):
    """
    Create a dictionary mapping field names to conversion functions

    Parameters
    ----------
    info_fields : list
        List of dictionaries with INFO field names and descriptions

    Returns
    -------
    dict
        Dictionary mapping field names to conversion functions
    """
    field_converters = {}
    for field in info_fields:
        field_name = field["id"]
        field_type = field.get("type")
        converter = {
            "String": str,
            "Character": str,
            "Integer": int,
            "Float": float,
            "Flag": lambda x: x,
        }.get(field_type)
        if converter:
            field_converters[field_name] = converter

    return field_converters


def write_variants_to_vcf(
    genie_counts, output_vcf, fasta, info_fields, field_converters
):
    """
    Write out the dataframe (with VCF-like description) to a VCF file
    with the INFO fields specified in the JSON file

    Parameters
    ----------
    genie_counts : pd.DataFrame
        Dataframe with one row per variant with counts and all other
        information aggregated per variant
    output_vcf : str
        Name of output VCF file
    fasta : pysam.FastaFile
        FASTA file as pysam object
    info_fields : list
        List of dictionaries with INFO field names and descriptions
        to be added to the VCF file
    field_converters : dict
        Dictionary mapping INFO field names to conversion functions
    """
    # Set up VCF header with the required fields
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

    # Write out variants as new records
    vcf_out = pysam.VariantFile(output_vcf, "w", header=header)

    print("Writing variants to VCF file...")
    for row in tqdm(
        genie_counts.itertuples(index=False), total=genie_counts.shape[0]
    ):
        # Split out chromosome, position, reference, and alternate alleles
        try:
            chrom, pos, ref, alt = row.grch38_description.split("_")
        except (AttributeError, ValueError) as e:
            print(f"Invalid grch38_description format: {e}")
            continue

        # Format the INFO field values according to the converters
        formatted_info_fields = {}
        for field_name, converter in field_converters.items():
            value = getattr(row, field_name, None)
            if pd.notna(value):
                try:
                    formatted_info_fields[field_name] = converter(value)
                except Exception as err:
                    print(f"Error converting field {field_name}: {err}")

        # For each variant, write new variant record with all INFO fields
        # Take 1 away from start due to differences in Pysam representation
        record = vcf_out.new_record(
            contig=str(chrom),
            start=int(pos) - 1,
            alleles=(ref, alt),
            id=".",
            qual=None,
            filter=None,
            info=formatted_info_fields,
        )

        vcf_out.write(record)

    vcf_out.close()
    try:
        fasta.close()
    except Exception as e:
        print(f"Error closing FASTA file: {e}")


def main():
    args = parse_args()
    genie_counts = read_in_to_df(
        args.input,
        sep="\t",
        header=0,
        dtype={
            "Start_Position": "Int64",
        },
    )
    fasta = read_in_fasta(filename=args.fasta)
    genie_counts_renamed = remove_disallowed_chars_from_columns(genie_counts)
    info_fields = generate_info_field_header_info(
        genie_counts=genie_counts_renamed
    )
    converter_dict = create_field_converter_dict(info_fields=info_fields)
    write_variants_to_vcf(
        genie_counts=genie_counts_renamed,
        output_vcf=args.output,
        fasta=fasta,
        info_fields=info_fields,
        field_converters=converter_dict,
    )


if __name__ == "__main__":
    main()
