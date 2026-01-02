## Generate counts from GENIE Cohort data
This repository contains scripts to convert Genie MAF data to aggregated counts to assist with interpretation according to [UK Somatic Variant Interpretation Guidelines (S-VIG)](https://www.acgs.uk.com/media/12831/svig-uk_guidelines_v10_-_post-acgs_ratification_final_submit01.pdf) and its [supplementary information](https://www.acgs.uk.com/media/12832/svig-uk-supplementary-material-post-acgs-ratification-final.pdf).

### Merge patient and sample information
The clinical data (patient IDs, cancer types etc.) must be merged into the MAF data using `Tumor_Sample_Barcode` from the MAF data and `SAMPLE_ID` from the clinical data in order to generate patient counts for each variant later.
Example command:
```
python merge_sample_info.py \
  --input_maf data_mutations_extended.txt \
  --clinical_info data_clinical_sample.txt \
  --output data_mutations_extended_clinical_info.txt
```

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
  --input data_mutations_extended_clinical_info.txt \
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

### Reannotate in GRCh38 against MANE transcripts
Get the latest VEP Docker image:
```
docker pull ensemblorg/ensembl-vep:release_115.2
```
Get the VEP GRCh38 RefSeq cache and FASTA:
```
wget https://ftp.ensembl.org/pub/release-115/variation/indexed_vep_cache/homo_sapiens_refseq_vep_115_GRCh38.tar.gz
wget https://ftp.ensembl.org/pub/release-115/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gzip -d Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
bgzip Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

```
docker run -v /home/Genie:/data -w /data <vep-image-id> vep \
  -i /data/data_mutations_extended_GRCh38_normalised_nochr_or_alt.vcf.gz \
  -o /data/data_mutations_extended_GRCh38_normalised_nochr_or_alt_annotated.vcf.gz  \
  --dir /data \
  --vcf --cache --refseq --exclude_predicted --hgvs --symbol --check_existing \
  --canonical --mane --numbers --pick --format vcf --offline --exclude_null_alleles \
  --assembly GRCh38 \
  --fields SYMBOL,Consequence,Feature,HGVSc,HGVSp,EXON,INTRON,CDS_position,Protein_position,CANONICAL,MANE,MANE_SELECT,MANE_PLUS_CLINICAL \
  --shift_3prime 1 \
  --buffer_size 500 \
  --no_stats \
  --compress_output bgzip
```
Split out VEP annotations to separate INFO fields. prefixed with `_CSQ`:
```
bcftools +split-vep --columns - -a CSQ -Ou -p 'CSQ_' -d \
  data_mutations_extended_GRCh38_normalised_nochr_or_alt_annotated.vcf.gz \
  | bcftools annotate -x INFO/CSQ \
  -o data_mutations_extended_GRCh38_normalised_nochr_or_alt_annotated_split.vcf.gz
```

### Merge liftover and new VEP annotations into existing GENIE data
The information from the lifted over and annotated VCF can now be used to overwrite the existing GENIE fields for:
- Hugo_Symbol
- Consequence
- RefSeq
- HGVSc
- HGVSp
- Exon_Number
- Protein_position
Example command:
```
python update_annotations.py \
  --input data_mutations_extended_clinical_sample.txt \
  --vep_vcf data_mutations_extended_GRCh38_normalised_nochr_or_alt_annotated_split.vcf.gz \
  --output_all data_mutations_extended_clinical_sample_reannotated.txt \
  --output data_mutations_extended_clinical_sample_annotations_replaced.txt
```
Note: this outputs a version with both the original GRCh37 GENIE annotations (`--output_all`) and a version with the fields overwritten; the latter should be used downstream. 

### Generate count data
This script generates counts of how many unique patients each variant is present in for all cancers and per cancer type, with one row per unique variant in the output TSV and each count as a column. The counts types include:
- The exact nucleotide change (all variants)
- The same amino acid change (all variants with HGVSp data present)
- The number of patients with frameshift (truncating) or nonsense variants at that position or downstream in the same gene. This is only present for variants with `frameshift_variant` or `stop_gained` in the Consequence field, where `Ter` (and not `ext`) is present in the HGVSp string.
- The number of patients with inframe deletions which cover the same Protein positions or are nested within the deletion.
    - These counts are only present for variants with the `inframe_deletion` Consequence.

This requires a file (`--columns_to_aggregate.txt`) where each GENIE annotation to be kept in the final VCF are provided, one per line, and (`--column_for_inframe_deletions`) which specifies whether to extract positions of an inframe deletion from the HGVSc or HGVSp notation. A file of cancer types which are classified as haemonc-related cancer types are (`--haemonc_cancer_types.txt`) and a file of cancer types which are classified as solid cancer types (`--solid_cancer_types.txt`) can also be provided in order to generate grouped counts.

Example command:
```
python generate_count_data.py \
  --input data_mutations_extended_clinical_info_GRCh38_fixed.txt \
  --columns_to_aggregate columns_to_aggregate.txt \
  --haemonc_cancer_types haemonc_cancer_types.txt (optional) \
  --solid_cancer_types solid_cancer_types.txt (optional) \
  --output Genie_v17_GRCh38_counts.tsv
```

### Write counts to VCF
The variants are then written out in GRCh38 with the existing GENIE fields and the count data are written out to VCF as INFO fields.
Example command:
```
python convert_counts_to_vcf.py \
  --input Genie_v17_GRCh38_counts.tsv \
  --fasta Homo_sapiens.GRCh38.dna.toplevel.fa.gz \
  --output Genie_v17_GRCh38_counts.vcf
```

You may want to do some sorting and removing of Unknown genes:
```
bcftools filter -e 'INFO/Hugo_Symbol=="Unknown"' Genie_v17_GRCh38_counts.vcf -Oz \
  | bcftools sort -Oz -o Genie_v17_GRCh38_counts_v1.0.0.vcf.gz
```

## Additional scripts
### Generate cancer types CSV for input to Genie NHS website
This script takes the final VCF and generates a file which maps each cancer type from the VCF INFO fields to its display name (original name from the Genie data), its cancer grouping and the total patients count. This is required as input to the [NHS Genie website](https://github.com/eastgenomics/genie_nhs_website).
Example command:
```
python generate_cancer_types_csv.py \
  --input Genie_v17_GRCh38_counts_v1.0.0.vcf.gz \
  --all_cancer_types all_cancer_types.txt \
  --haemonc_cancer_types haemonc_cancer_types.txt \
  --solid_cancer_types solid_cancer_types.txt \
  --output genie_cancer_types.csv
```

Example output format:
```
"display_name","vcf_name","is_haemonc","is_solid","total_patient_count"
"All Cancers","All_Cancers",0,0,180561
"Haemonc Cancers","Haemonc_Cancers",0,0,15095
"Solid Cancers","Solid_Cancers",0,0,157909
"Choroid Plexus Tumor","ChoroidPlexusTumor",0,1,73
```