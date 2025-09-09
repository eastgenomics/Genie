import argparse
import pysam
from collections import defaultdict


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
            "Information required to extract duplicate variants from VCF"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to VCF file",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Name of output VCF file",
    )

    return parser.parse_args()


def write_unique_normalisation_duplicates_to_vcf(in_vcf, out_vcf):
    """
    Extract the first instance of each duplicate variant from a VCF file.

    Parameters
    ----------
    in_vcf : str
        Path to the input VCF file.
    out_vcf : str
        Path to the output VCF file.
    """
    # Open input VCF
    vcf_in = pysam.VariantFile(in_vcf, "r")

    # Prepare output VCF with same header
    vcf_out = pysam.VariantFile(out_vcf, "w", header=vcf_in.header)

    # Count occurrences of each variant by CHROM, POS, REF, ALT
    counts = defaultdict(int)
    variants = defaultdict(list)

    for rec in vcf_in:
        key = (rec.chrom, rec.pos, rec.ref, tuple(rec.alts))
        counts[key] += 1
        variants[key].append(rec)

    # Write only the first instance of each duplicate
    for key, recs in variants.items():
        if counts[key] > 1:
            vcf_out.write(recs[0])

    vcf_in.close()
    vcf_out.close()
    print(f"First instances of duplicates written to {out_vcf}")


def main():
    args = parse_args()
    write_unique_normalisation_duplicates_to_vcf(args.input, args.output)


if __name__ == "__main__":
    main()
