## Generate counts from GENIE Cohort data
This repository contains scripts to convert Genie MAF data to aggregated counts to assist with interpretation according to UK Somatic Variant Intepretation Guidelines (S-VIG).

### Merge sample info
The clinical data (patient IDs, cancer types etc.) must be merged into the MAF data using `Tumor_Sample_Barcode` from the MAF data and `SAMPLE_ID` from the clinical data in order to generate patient counts for each variant later.
Example command:
```
python merge_sample_info.py \
  --input_maf data_mutations_extended.txt \
  --clinical_info data_clinical_sample.txt \
  --output data_mutations_extended_clinical_info.txt
```

### Convert MAF to VCF
Each unique variant in the MAF data are required to be converted to VCF description in GRCh37 to enable liftover to GRCh38. A FASTA (sourced from Ensembl) is required for the GRCh37 reference genome.

Example command:
```
python convert_raw_maf_to_vcf.py \
  --input data_mutations_extended.txt \
  --fasta Homo_sapiens.GRCh37.dna.toplevel.fa.gz \
  --output_vcf data_mutations_extended.vcf
```
Note: the VCF should then be sorted and normalised with bcftools and lifted over to GRCh38 with Picard LiftoverVcf.


### Add GRCh38 liftover
The liftover information must now be added back to the Genie data, so that the GRCh38 description can be considered as the unique representation of each variant. This is because multiple variants in the GENIE MAF data may normalise to the same variant.
Example command:
```
python add_grch38_liftover.py \
  --genie_clinical data_mutations_extended_clinical_info.txt \
  --vcf data_mutations_extended.vcf \
  --output data_mutations_extended_clinical_info_GRCh38.txt
```
