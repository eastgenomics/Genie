## Generate counts from GENIE Cohort data
This repository contains scripts to convert Genie MAF data to aggregated counts to assist with interpretation according to [UK Somatic Variant Interpretation Guidelines (S-VIG)](https://www.acgs.uk.com/media/12831/svig-uk_guidelines_v10_-_post-acgs_ratification_final_submit01.pdf) and its [supplementary information](https://www.acgs.uk.com/media/12832/svig-uk-supplementary-material-post-acgs-ratification-final.pdf).

### Convert MAF to VCF
Each unique variant in the MAF data is required to be converted to VCF description in GRCh37 to enable liftover to GRCh38. A FASTA (sourced from Ensembl) is required for the GRCh37 reference genome.
These can be retrieved from the Ensembl FTP site and then bgzipped and indexed:
```
wget https://ftp.ensembl.org/pub/grch37/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.toplevel.fa.gz
gzip -d Homo_sapiens.GRCh37.dna.toplevel.fa.gz
bgzip Homo_sapiens.GRCh37.dna.toplevel.fa
samtools faidx Homo_sapiens.GRCh37.dna.toplevel.fa.gz
```

Example command:
```
python convert_raw_maf_to_vcf.py \
  --input data_mutations_extended.txt \
  --fasta Homo_sapiens.GRCh37.dna.toplevel.fa.gz \
  --output data_mutations_extended.vcf
```
The script will print out any variants where the reference allele does not match the given FASTA file and remove them.

Note: the VCF should then be sorted and normalised with bcftools:
```
bcftools sort data_mutations_extended.vcf \
  | bcftools norm -f Homo_sapiens.GRCh37.dna.toplevel.fa.gz \
  -Oz -o data_mutations_extended_normalised.vcf.gz
```


### Write out normalisation duplicates
Variants in the Genie data are submitted by different institutions and are not normalised, meaning multiple descriptions of a variant in Genie may map to the same GRCh37 description after normalisation. Once the GRCh37 VCF is normalised with bcftools norm, we can extract the first instance of each normalisation duplicate from the normalised VCF:
```
python write_normalisation_duplicates_to_vcf.py \
  --input data_mutations_extended_normalised.vcf.gz \
  --output data_mutations_extended_normalised_duplicates.vcf
```
This VCF of duplicates should then be annotated with VEP to obtain the `Consequence,Feature,HGVSc,HGVSp` fields so that it can be used to correct the annotations (`Consequence, Feature, HGVSc, HGVSp, Variant_Type, Variant_Classification`) for the rows which contain these duplicates in the Genie data. This is required because these annotations are used for counting. Example VEP command:
```
docker run -v /home/Genie:/data -w /data <vep-image-id>  \
  vep -i /data/data_mutations_extended_normalised_duplicates.vcf \
  -o /data/data_mutations_extended_normalised_duplicates_annotated.vcf.gz  \
  --dir /data   \
  --vcf --cache --exclude_predicted --hgvs --hgvsg    \
  --check_existing --numbers --format vcf  \
  --offline --exclude_null_alleles --assembly GRCh37 \
  --fields Consequence,Feature,HGVSc,HGVSp   \
  --buffer_size 500    \
  --no_stats --compress_output bgzip --shift_3prime 1
```
A list of Ensembl transcripts can be obtained from the duplicates, one per line:
```
bcftools query -f '%Transcript_ID\n' data_mutations_extended_normalised_duplicates.vcf \
  | grep -v '^\.$' | sort -u > transcripts.tsv
```
Then these transcripts can be used to filter the results:
```
docker run -v /home/Genie:/data -w /data <vep-image-id> \
  filter_vep -i /data/data_mutations_extended_normalised_duplicates_annotated.vcf.gz \
  -o /data/data_mutations_extended_normalised_duplicates_annotated_filtered.vcf \
  --only_matched --filter "Feature in /data/transcripts.tsv" --force_overwrite
```
Then the VEP CSQ string can be split to separate INFO fields:
```
bcftools +split-vep --columns - -a CSQ -Ou -p 'CSQ_' -d \
  data_mutations_extended_normalised_duplicates_annotated_filtered.vcf \
  | bcftools annotate -x INFO/CSQ \
  -o data_mutations_extended_normalised_duplicates_annotated_filtered_split.vcf
```

## Liftover variants to GRCh38
1. Add `chr` prefix to the VCF to allow liftover:
```
# Create file with no chr to chr mapping for contigs in the VCF
bcftools view -h \
  data_mutations_extended_normalised.vcf.gz \
  | grep '^##contig' | sed 's/##contig=<ID=/ /' | cut -d ',' -f 1 | \
  awk '{print $1, "chr"$1}' > nochr_to_chr.txt

# Rename chrMT to chrM otherwise this doesn't liftover
sed -i -e 's/chrMT/chrM/g' nochr_to_chr.txt

# Rename chrs in VCF with bcftools annotate using this file
bcftools annotate \
  --rename-chrs nochr_to_chr.txt \
  data_mutations_extended_normalised.vcf.gz \
  -Oz -o data_mutations_extended_normalised_withchr.vcf.gz
```
2. Download the GRCh38 reference FASTA and chain file:
```
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz
```
3. Create a sequence dictionary with Picard:
```
docker run -it \
  -v $(pwd):/usr/working/ \
  --entrypoint /bin/bash \
  broadinstitute/picard:3.3.0

java -Xmx60g -jar /usr/picard/picard.jar CreateSequenceDictionary \
  -R hg38.fa.gz \
  -O hg38.dict
```
4. Run Picard LiftoverVcf:
```
java -Xmx60g -jar /usr/picard/picard.jar LiftoverVcf \
  --INPUT data_mutations_extended_normalised_withchr.vcf.gz \
  --OUTPUT data_mutations_extended_GRCh38_normalised_withchr.vcf.gz \
  --CHAIN hg19ToHg38.over.chain.gz \
  --REFERENCE_SEQUENCE hg38.fa.gz \
  --REJECT rejected.vcf \
  --RECOVER_SWAPPED_REF_ALT \
  --WRITE_ORIGINAL_POSITION \
  --WRITE_ORIGINAL_ALLELES
```
5. Remove `chr` info from the lifted over VCF:
```
# Create file with chr to no chr mapping for GRCh38
bcftools view -h \
  data_mutations_extended_GRCh38_normalised_withchr.vcf.gz \
  | grep '^##contig' | sed 's/##contig=<ID=/ /' | cut -d ',' -f 1 | \
  awk '{print $1, substr($1, 4)}' > chr_to_nochr.txt

# Manually rename chrM to MT
sed 's/^chrM[[:space:]]\+M$/chrM MT/' chr_to_nochr.txt > chr_to_nochr_fixed.txt

# Rename chrs using this file
bcftools annotate \
  --rename-chrs chr_to_nochr_fixed.txt \
  data_mutations_extended_GRCh38_normalised_withchr.vcf.gz \
  -Oz -o data_mutations_extended_GRCh38_normalised_nochr.vcf.gz
```

### Merge sample info
The clinical data (patient IDs, cancer types etc.) must be merged into the MAF data using `Tumor_Sample_Barcode` from the MAF data and `SAMPLE_ID` from the clinical data in order to generate patient counts for each variant later.
Example command:
```
python merge_sample_info.py \
  --input_maf data_mutations_extended.txt \
  --clinical_info data_clinical_sample.txt \
  --output data_mutations_extended_clinical_info.txt
```


### Add GRCh38 liftover to Genie data
The liftover information can now be added back to the Genie data.
Example command:
```
python add_grch38_liftover.py \
  --genie_clinical data_mutations_extended_clinical_info.txt \
  --vcf data_mutations_extended_GRCh38_normalised_nochr.vcf.gz \
  --output data_mutations_extended_clinical_info_GRCh38.txt
```

### Fix normalisation duplicates
We can use the VEP annotations for the normalisation duplicates generated earlier to correct the annotations (Consequence, HGVSc, HGVSp) for the relevant rows in the Genie data which can be used to derive the Variant_Type and Variant_Classification prior to counting:
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

This requires a file (`--columns_to_aggregate.txt`) where each Genie annotation to be kept in the final VCF are provided, one per line, and (`--column_for_inframe_deletions`) which specifies whether to extract positions of an inframe deletion from the HGVSc or HGVSp notation. A file of cancer types which are classified as haemonc-related cancer types are (`--haemonc_cancer_types.txt`) and a file of cancer types which are classified as solid cancer types (`--solid_cancer_types.txt`) can also be provided in order to generate grouped counts.

Example command:
```
python generate_count_data.py \
  --input data_mutations_extended_clinical_info_GRCh38_fixed.txt \
  --columns_to_aggregate columns_to_aggregate.txt \
  --haemonc_cancer_types haemonc_cancer_types.txt (optional) \
  --solid_cancer_types solid_cancer_types.txt (optional) \
  --column_for_inframe_deletions HGVSp [HGVSc or HGVSp] \
  --output Genie_v17_GRCh38_counts.tsv
```

### Write counts to VCF
The variants are then written out in GRCh38 with the existing Genie fields and the count data are written out to VCF as INFO fields.
Example command:
```
python convert_counts_to_vcf.py \
  --input Genie_v17_GRCh38_counts.tsv \
  --fasta Homo_sapiens.GRCh38.dna.toplevel.fa.gz \
  --output Genie_v17_GRCh38_counts.vcf
```
