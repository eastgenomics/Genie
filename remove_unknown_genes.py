import argparse
import pandas as pd

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
        description="Information required to subset MAF data"
    )

    parser.add_argument(
        "--input", required=True, type=str, help="Path to MAF file"
    )

    parser.add_argument(
        "--output", required=True, type=str, help="Name of output cleaned file"
    )

    return parser.parse_args()


def remove_unknown_genes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unknown genes from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing gene information.

    Returns
    -------
    pd.DataFrame
        DataFrame with rows containing unknown genes removed.
    """
    initial_row_count = len(df)
    filtered_df = df[
        df["Hugo_Symbol"].notna()
        & (df["Hugo_Symbol"].str.upper() != "UNKNOWN")
    ]
    removed_count = initial_row_count - len(filtered_df)
    print(
        f"Removed {removed_count} rows where Hugo_Symbol is Unknown or not"
        " present"
    )
    return filtered_df


def main():
    args = parse_args()
    genie_data = read_in_to_df(
        args.input,
        header=0,
        dtype={
            "Hugo_Symbol": "str",
            "Chromosome": "str",
            "Start_Position": "Int64",
            "Reference_Allele": "str",
            "Tumor_Seq_Allele2": "str",
            "Entrez_Gene_Id": "Int64",
        },
    )
    cleaned_data = remove_unknown_genes(genie_data)
    cleaned_data.to_csv(args.output, index=False, sep="\t")


if __name__ == "__main__":
    main()
