## Generate counts from GENIE Cohort data
This repository contains scripts to convert Genie MAF data to aggregated counts to assist with interpretation according to [UK Somatic Variant Intepretation Guidelines (S-VIG)](https://www.acgs.uk.com/media/12831/svig-uk_guidelines_v10_-_post-acgs_ratification_final_submit01.pdf) and its [supplementary information](https://www.acgs.uk.com/media/12832/svig-uk-supplementary-material-post-acgs-ratification-final.pdf).

### Convert MAF to VCF
Each unique variant in the MAF data are required to be converted to VCF description in GRCh37 to enable liftover to GRCh38. A FASTA (sourced from Ensembl) is required for the GRCh37 reference genome.

Example command:
```
python convert_raw_maf_to_vcf.py \
  --input data_mutations_extended.txt \
  --fasta Homo_sapiens.GRCh37.dna.toplevel.fa.gz \
  --output data_mutations_extended.vcf
```
Note: the VCF should then be sorted and normalised with bcftools and lifted over to GRCh38 with Picard LiftoverVcf.


### Write out normalisation duplicates
Variants in the Genie data are submitted by different institutions and are not necessarily normalised. Once the GRCh37 VCF is normalised with bcftools norm, we can extract the first instance of each duplicate from the normalised VCF:
```
python write_normalisation_duplicates_to_vcf.py \
  --input data_mutations_extended_normalised.vcf.gz \
  --output data_mutations_extended_normalised_duplicates.vcf
```
This VCF should then be annotated with VEP so that it can be used to correct the annotations for these duplicates in the Genie data.


### Merge sample info
The clinical data (patient IDs, cancer types etc.) must be merged into the MAF data using `Tumor_Sample_Barcode` from the MAF data and `SAMPLE_ID` from the clinical data in order to generate patient counts for each variant later.
Example command:
```
python merge_sample_info.py \
  --input_maf data_mutations_extended.txt \
  --clinical_info data_clinical_sample.txt \
  --output data_mutations_extended_clinical_info.txt
```


### Add GRCh38 liftover
The liftover information must now be added back to the Genie data, so that the GRCh38 description can be considered as the unique representation of each variant. This is because multiple variants in the GENIE MAF data may normalise to the same variant.
Example command:
```
python add_grch38_liftover.py \
  --genie_clinical data_mutations_extended_clinical_info.txt \
  --vcf data_mutations_extended_GRCh38_normalised_nochr_or_alt.vcf.gz \
  --output data_mutations_extended_clinical_info_GRCh38.txt
```


### Fix normalisation duplicates
The normalisation duplicates can be annotated with VEP and then these VEP annotations can be used to correct the annotations (Consequence, HGVSc, HGVSp) and be used to derive the Variant_Type and Variant_Classification fields in these rows within the Genie data prior to counting:
```
python modify_normalisation_dups.py \
  --input data_mutations_extended_clinical_info_GRCh38.txt \
  --vep_vcf data_mutations_extended_normalised_duplicates_annotated_filtered_split.vcf \
  --output data_mutations_extended_clinical_info_GRCh38_fixed.txt
```


### Generate count data
This script generates counts of how many unique patients each variant is present in for all cancers and per cancer type, with one row per unique variant in the output TSV and each count as a column. The counts types include:
- The exact nucleotide change (all variants)
- The same amino acid change (all variants with HGVSp data present)
- The number of patients with frameshift (truncating) or nonsense variants at that position or downstream in the same gene. This is only present for the following variant types, where `Ter` is present in the HGVSp string:
    - `Frame_Shift_Del`
    - `Frame_Shift_Ins`
    - `Nonsense_Mutation`
- The number of patients with inframe deletions which cover the same CDS positions or are nested within the deletion.
    - These counts are only present for the `In_Frame_Del` variant type.

This requires a file (`--columns_to_aggregate.txt`) where each Genie annotation to be kept in the final VCF are provided, one per line. A file of cancer types which are classified as haemonc-related cancer types are (`--haemonc_cancer_types.txt`) can also be provided in order to generate counts in all haemonc cancer types. 

Example command:
```
python generate_count_data.py \
  --input data_mutations_extended_clinical_info_GRCh38_fixed.txt \
  --columns_to_aggregate columns_to_aggregate.txt \
  --haemonc_cancer_types haemonc_cancer_types.txt (optional) \
  --output Genie_v17_GRCh38_counts.tsv
```

### Write counts to VCF
The aggregated existing Genie fields and the count data are written out to VCF as INFO fields, with the GRCh38 liftover data used for CHROM, POS, REF and ALT.
Example command:
```
python convert_counts_to_vcf.py \
  --input Genie_v17_GRCh38_counts.tsv \
  --fasta Homo_sapiens.GRCh38.dna.toplevel.fa.gz \
  --output Genie_v17_GRCh38_counts.vcf
```
