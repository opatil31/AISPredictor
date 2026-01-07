# Patient-Specific Genomic Attribution for Adolescent Idiopathic Scoliosis

## Implementation Plan: HyenaDNA-Based Disease Prediction with Interpretable Gene-Region Attribution

**Version:** 1.0  
**Date:** January 2026  
**Status:** Proposed

---

## Executive Summary

This document outlines the implementation of a deep learning system that predicts Adolescent Idiopathic Scoliosis (AIS) risk from chromosome 6 genetic variants while providing patient-specific explanations at the gene and genomic region level.

**Key Innovation:** By combining pretrained DNA language model embeddings (HyenaDNA) with a hierarchical attention architecture, we can answer: *"For this specific patient, which genes and which regulatory regions (promoter, exon, intron, etc.) most strongly drive their AIS risk prediction?"*

**Clinical Relevance:** AIS affects 2-3% of adolescents, with chromosome 6 harboring well-replicated susceptibility loci including GPR126/ADGRG6. Understanding patient-specific genetic drivers could inform personalized screening and treatment strategies.

| Metric | Target |
|--------|--------|
| Sample Size | ~3,000 AIS cases + ~12,000 matched controls |
| Data Source | UK Biobank (GRCh38) |
| Primary Outcome | Per-patient gene and region attribution |
| Validation | Replication of known GPR126 locus signal |

---

## Table of Contents

1. [Scientific Background](#1-scientific-background)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Design](#4-model-design)
5. [Training Strategy](#5-training-strategy)
6. [Attribution & Interpretation](#6-attribution--interpretation)
7. [Validation Framework](#7-validation-framework)
8. [Implementation Timeline](#8-implementation-timeline)
9. [Technical Requirements](#9-technical-requirements)
10. [Risk Assessment](#10-risk-assessment)
11. [Appendices](#appendices)

---

## 1. Scientific Background

### 1.1 Adolescent Idiopathic Scoliosis

Adolescent Idiopathic Scoliosis (AIS) is a complex spinal deformity affecting 2-3% of adolescents, with a strong genetic component (heritability ~38%). Genome-wide association studies have identified multiple susceptibility loci, with chromosome 6 harboring one of the most robustly replicated signals.

### 1.2 The GPR126/ADGRG6 Locus

The rs6570507 variant at 6q24.1, located within an intron of GPR126 (now ADGRG6), has been replicated across multiple ancestries:

| Study | Population | P-value | OR |
|-------|------------|---------|-----|
| Kou et al. 2013 | Japanese | 1.6×10⁻¹⁴ | 1.26 |
| Londono et al. 2014 | European | 3.1×10⁻⁸ | 1.17 |
| Sharma et al. 2015 | Multi-ethnic | 4.0×10⁻¹¹ | 1.21 |

This locus serves as our primary validation target—a well-performing model should recover this signal.

### 1.3 Why DNA Language Models?

Traditional GWAS identifies associated variants but struggles to:
- Capture complex epistatic interactions
- Provide patient-specific risk decomposition
- Integrate sequence context around variants

HyenaDNA, a long-range genomic foundation model, encodes DNA sequence context at single-nucleotide resolution. By computing "delta embeddings" (the difference between reference and alternate allele embeddings), we capture the functional impact of variants in their sequence context.

### 1.4 Project Goals

| Goal | Description |
|------|-------------|
| **Primary** | Predict AIS case/control status from chr6 variants |
| **Secondary** | Provide per-patient attribution at gene and region level |
| **Validation** | Recover known GPR126 intronic signal |
| **Deliverable** | Patient-specific reports identifying top contributing genes/regions |

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  UK Biobank        Ancestry QC       Case-Control      Variant             │
│  .pgen files  ───► & Filtering  ───► Matching     ───► Extraction          │
│                                                                              │
│  Reference         VEP                Gene-Region                           │
│  Genome (hg38) ───► Annotation    ───► Index                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EMBEDDING COMPUTATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   For each variant v:                                                        │
│                                                                              │
│   ┌──────────────┐      ┌──────────────┐                                    │
│   │  REF window  │      │  ALT window  │     8kb sequence windows           │
│   │  ...ACGT...  │      │  ...ACTT...  │     centered on variant            │
│   └──────┬───────┘      └──────┬───────┘                                    │
│          │                     │                                             │
│          ▼                     ▼                                             │
│   ┌──────────────┐      ┌──────────────┐                                    │
│   │   HyenaDNA   │      │   HyenaDNA   │     Pretrained DNA LM              │
│   │   Encoder    │      │   Encoder    │                                    │
│   └──────┬───────┘      └──────┬───────┘                                    │
│          │                     │                                             │
│          ▼                     ▼                                             │
│      e(REF)                e(ALT)            Pooled embeddings               │
│          │                     │             (±50bp around variant)          │
│          └─────────┬───────────┘                                             │
│                    │                                                         │
│                    ▼                                                         │
│              Δe = e(ALT) - e(REF)            Delta embedding                 │
│                                              (precomputed once)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PATIENT REPRESENTATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   For patient i with dosages g_iv ∈ {0, 1, 2}:                              │
│                                                                              │
│   x_iv = g_iv × Δe_v        Dosage-scaled variant embedding                 │
│                                                                              │
│   Patient input = { (x_iv, position, region_type) for all variants v }      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Variant Embeddings                              │   │
│   │            (batch, n_variants, embedding_dim)                        │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Input Projection + Position                       │   │
│   │                      Linear → LayerNorm → GELU                       │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                   Region-Type Attention Pooling                      │   │
│   │                                                                      │   │
│   │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│   │   │ Promoter │ │   UTR5   │ │   Exon   │ │  Intron  │ │  Splice  │  │   │
│   │   │ Variants │ │ Variants │ │ Variants │ │ Variants │ │ Variants │  │   │
│   │   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│   │        │            │            │            │            │         │   │
│   │        ▼            ▼            ▼            ▼            ▼         │   │
│   │   ┌──────────────────────────────────────────────────────────────┐  │   │
│   │   │     Learned Query per Region → Attention → Region Embedding  │  │   │
│   │   └──────────────────────────────────────────────────────────────┘  │   │
│   │                                                                      │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Region Combination Layer                          │   │
│   │              Concatenate → MLP → Patient Embedding                   │   │
│   └───────────────────────────┬─────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      Classification Head                             │   │
│   │                   LayerNorm → MLP → P(AIS)                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ATTRIBUTION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Integrated Gradients w.r.t. variant embeddings:                           │
│                                                                              │
│   • Per-variant attribution scores                                           │
│   • Aggregated to region level (sum within promoter/exon/intron/etc.)       │
│   • Aggregated to gene level (sum across regions)                           │
│                                                                              │
│   Output: Ranked list of genes and regions driving each patient's risk      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Design Rationale

| Design Choice | Rationale |
|---------------|-----------|
| **Delta embeddings** | Captures variant effect in sequence context; precomputable for efficiency |
| **Dosage scaling** | Incorporates copy number; handles imputed genotypes |
| **Region-first pooling** | Enables direct region-level attribution without post-hoc mapping |
| **Vectorized operations** | Efficient GPU utilization; avoids Python loops |
| **Reduced model size** | Appropriate for ~15k sample cohort; prevents overfitting |

---

## 3. Data Pipeline

### 3.1 Phase 0: Cohort Definition

#### 3.1.1 Case Identification

AIS cases identified via ICD-10 codes from UK Biobank hospital inpatient records:

| Code | Description | Expected Count |
|------|-------------|----------------|
| M41.1 | Juvenile idiopathic scoliosis | ~500 |
| M41.2 | Other idiopathic scoliosis | ~2,500 |
| **Total** | | **~3,000** |

#### 3.1.2 Ancestry Quality Control

```
UK Biobank Samples (~500k)
         │
         ▼
┌─────────────────────────────┐
│  Self-reported ancestry     │  Filter to European (field 21000)
│  (field 21000)              │  Codes: 1, 1001, 1002, 1003
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  PCA-based outlier removal  │  Remove samples >6 SD from mean
│  (first 4 PCs)              │  on PC1-PC4
└─────────────┬───────────────┘
              │
              ▼
       Ancestry-matched
         cohort (~450k)
```

#### 3.1.3 Control Selection

**Strategy:** Nearest-neighbor matching without replacement

**Matching Variables:**
- Age at recruitment
- Sex
- PC1-PC4 (ancestry)
- Genotyping batch (optional)

**Ratio:** 4 controls per case (~12,000 controls)

**Exclusions:** 
- Any scoliosis diagnosis (M41.x)
- Related individuals (KING kinship > 0.0884)

#### 3.1.4 Final Cohort

| Group | N | Female % | Mean Age |
|-------|---|----------|----------|
| Cases | ~3,000 | ~70% | ~55 |
| Controls | ~12,000 | ~70% | ~55 |
| **Total** | **~15,000** | | |

### 3.2 Phase 1: Variant Extraction

#### 3.2.1 Quality Control Filters

| Filter | Threshold | Rationale |
|--------|-----------|-----------|
| Minor Allele Frequency | ≥ 0.01 | Sufficient power for common variants |
| Imputation INFO | ≥ 0.8 | High-quality imputed variants |
| Variant Type | SNPs only | Simplifies embedding computation |
| Hardy-Weinberg | P > 1×10⁻⁶ | Remove genotyping errors |

#### 3.2.2 Expected Variant Counts

| Stage | Approximate Count |
|-------|-------------------|
| All chr6 variants | ~2,000,000 |
| After MAF filter | ~500,000 |
| After INFO filter | ~400,000 |
| SNPs only | ~350,000 |
| Final (with HWE) | **~300,000** |

#### 3.2.3 Optional LD Pruning

For computational efficiency or model simplification:
- Window: 500kb
- Step: 1 variant
- r² threshold: 0.8

Reduces to ~50,000-100,000 tag variants.

### 3.3 Phase 2: Variant Annotation

#### 3.3.1 VEP Configuration

```bash
vep \
  --input_file variants.txt \
  --output_file annotated.tsv \
  --cache --assembly GRCh38 \
  --canonical --pick \
  --distance 2000,500 \           # Promoter: 2kb upstream, 500bp downstream
  --fields "Gene,Consequence,DISTANCE"
```

#### 3.3.2 Region Type Mapping

| VEP Consequence | Region Type |
|-----------------|-------------|
| missense_variant, synonymous_variant, stop_gained, etc. | Exon |
| splice_acceptor_variant, splice_donor_variant, splice_region_variant | Splice |
| 5_prime_UTR_variant | UTR5 |
| 3_prime_UTR_variant | UTR3 |
| intron_variant | Intron |
| upstream_gene_variant (≤2kb) | Promoter |
| downstream_gene_variant | Downstream |
| intergenic_variant | Intergenic |

#### 3.3.3 Multi-Gene Resolution

Some variants map to multiple overlapping genes. 

**Strategy:** Primary gene assignment based on:
1. Canonical transcript priority
2. Consequence severity (coding > non-coding)
3. Biotype (protein_coding > pseudogene)

### 3.4 Phase 3: Delta Embedding Computation

#### 3.4.1 HyenaDNA Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `hyenadna-small-32k-seqlen` | Balance of context and speed |
| Window size | 8,192 bp | Captures local regulatory context |
| Pooling window | ±50 bp | Captures binding site disruption |
| Precision | FP16 | Memory efficiency |

#### 3.4.2 Computation Strategy

```
For each variant v at position p:

1. Extract REF window: hg38[chr6, p-4096 : p+4096]
2. Create ALT window: substitute ALT allele at position p
3. Tokenize both sequences (character-level: A,C,G,T,N)
4. Forward pass through HyenaDNA
5. Pool hidden states at positions [p-50 : p+50]
6. Δe_v = mean_pool(ALT) - mean_pool(REF)
```

#### 3.4.3 Output Specifications

| Output | Shape | Storage |
|--------|-------|---------|
| Delta embeddings | (n_variants, 256) | ~300MB |
| Variant positions | (n_variants,) | ~1.2MB |
| Region types | (n_variants,) | ~300KB |

#### 3.4.4 Parallelization

- Batch size: 32 sequences
- GPU memory: ~16GB per worker
- Estimated time: ~24 hours on 4× A100

### 3.5 Data Storage Schema

```
data/
├── cohort/
│   ├── cohort.parquet              # Patient metadata + labels
│   ├── matching_info.parquet       # Case-control pairs
│   └── ancestry_pcs.parquet        # PC loadings
├── variants/
│   ├── variants.parquet            # Variant metadata
│   ├── dosages.zarr/               # (n_patients, n_variants) dosages
│   └── sample_ids.txt              # Patient ID order
├── annotations/
│   ├── vep_annotations.parquet     # VEP output
│   └── gene_region_index.json      # Gene → variant mapping
└── embeddings/
    └── delta_embeddings.zarr/      # (n_variants, 256) embeddings
```

---

## 4. Model Design

### 4.1 Architecture Specifications

#### 4.1.1 Model Dimensions

| Component | Dimension | Rationale |
|-----------|-----------|-----------|
| HyenaDNA embedding | 256 | Fixed by pretrained model |
| Model dimension (d_model) | 64 | Reduced for sample size |
| Attention heads | 2 | Minimal for regularization |
| Region types | 8 | Biological categories |
| Hidden dimension | 128 | 2× d_model |

#### 4.1.2 Parameter Count

| Component | Parameters |
|-----------|------------|
| Input projection | 256 × 64 = 16,384 |
| Position embedding | 64 + 16 = 80 |
| Region queries | 8 × 64 = 512 |
| Key/Value projections | 2 × 64 × 64 = 8,192 |
| Region combination | (8 × 64) × 128 + 128 × 64 = 73,728 |
| Classifier | 64 × 32 + 32 × 2 = 2,112 |
| **Total** | **~100,000** |

This is 10-100× smaller than typical transformer models, appropriate for ~15k samples.

### 4.2 Layer Specifications

#### 4.2.1 Input Projection

```
Input: (batch, n_variants, 256)  [dosage-scaled delta embeddings]
       ↓
Linear(256 → 64)
       ↓
LayerNorm(64)
       ↓
GELU
       ↓
Dropout(0.3)
       ↓
Output: (batch, n_variants, 64)
```

#### 4.2.2 Positional Encoding

Sinusoidal encoding based on genomic position:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Position normalized by 10⁸ to bring into reasonable range.

#### 4.2.3 Region Attention Pooling

For each region type r ∈ {promoter, utr5, exon, intron, utr3, splice, downstream, other}:

```
Query: q_r ∈ R^64  [learned per region]
Keys:  K = W_k × variants_in_region  ∈ R^(n_r × 64)
Values: V = W_v × variants_in_region  ∈ R^(n_r × 64)

Attention: α = softmax(q_r × K^T / √64)
Output: region_embedding_r = α × V ∈ R^64
```

Empty regions (no variants) receive a learned "empty" embedding.

#### 4.2.4 Region Combination

```
Input: [region_embeddings]  ∈ R^(8 × 64) = R^512
       ↓
Linear(512 → 128)
       ↓
LayerNorm(128)
       ↓
GELU
       ↓
Dropout(0.3)
       ↓
Linear(128 → 64)
       ↓
Output: patient_embedding ∈ R^64
```

#### 4.2.5 Classification Head

```
Input: patient_embedding ∈ R^64
       ↓
LayerNorm(64)
       ↓
Dropout(0.3)
       ↓
Linear(64 → 32)
       ↓
GELU
       ↓
Dropout(0.3)
       ↓
Linear(32 → 2)
       ↓
Output: logits ∈ R^2  [control, case]
```

### 4.3 Regularization Strategy

| Technique | Value | Location |
|-----------|-------|----------|
| Dropout | 0.3 | All layers |
| Weight decay | 0.1 | AdamW optimizer |
| Gradient clipping | 1.0 | Global norm |
| Early stopping | 15 epochs | Validation AUROC |
| Class weighting | Inverse frequency | Loss function |

### 4.4 Baseline Models

For benchmarking, we implement two simpler baselines:

#### 4.4.1 Mean Pooling + Logistic Regression

```
Patient embedding = mean(dosage_i × Δe_i for all variants i)
Prediction = LogisticRegression(patient_embedding)
```

Expected AUROC: ~0.55-0.60

#### 4.4.2 Simple Attention (No Hierarchy)

```
Single attention layer over all variants → prediction
Parameters: ~50k
```

Expected AUROC: ~0.58-0.63

The full hierarchical model should exceed both baselines.

---

## 5. Training Strategy

### 5.1 Cross-Validation Design

**Strategy:** Stratified 5-fold cross-validation

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Dataset (~15k)                       │
├─────────────────────────────────────────────────────────────┤
│  Fold 1: [████████████████] Train  [████] Val               │
│  Fold 2: [████] Val  [████████████████] Train               │
│  Fold 3: [████████] Train  [████] Val  [████████]           │
│  Fold 4: [████████████] Train  [████] Val  [████]           │
│  Fold 5: [████] Train  [████] Val  [████████████]           │
└─────────────────────────────────────────────────────────────┘

Each fold: ~12,000 train / ~3,000 validation
Stratified by case/control status
```

### 5.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1×10⁻³ |
| LR schedule | Cosine annealing |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | 15 epochs patience |
| Mixed precision | FP16 |

### 5.3 Loss Function

Weighted cross-entropy to handle class imbalance:

```
L = -Σ w_c × y_c × log(p_c)

where w_control = n_case / n_total
      w_case = n_control / n_total
```

### 5.4 Evaluation Metrics

| Metric | Primary | Rationale |
|--------|---------|-----------|
| **AUROC** | ✓ | Standard for binary classification |
| AUPRC | | Important for imbalanced data |
| Accuracy | | Interpretability |
| Sensitivity/Specificity | | Clinical relevance |

### 5.5 Hyperparameter Search Space

If resources permit:

| Parameter | Search Range |
|-----------|--------------|
| d_model | {32, 64, 128} |
| n_heads | {1, 2, 4} |
| dropout | {0.2, 0.3, 0.4} |
| learning_rate | {5×10⁻⁴, 1×10⁻³, 2×10⁻³} |
| weight_decay | {0.05, 0.1, 0.2} |

Search strategy: Random search with 20 trials, optimizing validation AUROC.

---

## 6. Attribution & Interpretation

### 6.1 Attribution Method

**Method:** Layer Integrated Gradients (Captum)

**Target Layer:** Input projection (after dosage scaling)

**Baseline:** Zero embedding (no variant effect)

**Steps:** 50 interpolation steps

### 6.2 Attribution Levels

```
┌─────────────────────────────────────────────────────────────┐
│                    VARIANT LEVEL                             │
│  Integrated Gradients → attribution per variant              │
│  a_v = IG(model, x_v, baseline=0)                           │
└─────────────────────────────┬───────────────────────────────┘
                              │ Aggregate by region
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    REGION LEVEL                              │
│  a_region = Σ |a_v| for v in region                         │
│  Regions: promoter, exon, intron, UTR, splice               │
└─────────────────────────────┬───────────────────────────────┘
                              │ Aggregate by gene
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GENE LEVEL                                │
│  a_gene = Σ a_region for all regions in gene                │
│  Provides ranked list of contributing genes                  │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Patient Report Structure

For each patient, generate:

```json
{
  "patient_id": "UKB_XXXXX",
  "prediction": {
    "ais_probability": 0.73,
    "risk_category": "elevated"
  },
  "top_genes": [
    {
      "rank": 1,
      "gene_id": "ENSG00000112414",
      "gene_name": "GPR126",
      "total_attribution": 0.234,
      "region_breakdown": {
        "promoter": 0.012,
        "exon": 0.031,
        "intron": 0.178,
        "splice": 0.013
      },
      "top_variants": [
        {
          "variant_id": "rs6570507",
          "position": 142723757,
          "consequence": "intron_variant",
          "attribution": 0.089
        }
      ]
    }
  ],
  "region_summary": {
    "intronic_contribution": 0.45,
    "exonic_contribution": 0.22,
    "promoter_contribution": 0.18,
    "other_contribution": 0.15
  }
}
```

### 6.4 Visualization Outputs

1. **Genome Browser Track:** BED file of variant attributions for IGV/UCSC
2. **Gene Ranking Plot:** Bar chart of top 20 genes by attribution
3. **Region Pie Chart:** Contribution breakdown by region type
4. **Variant Manhattan Plot:** Attribution vs. position within top genes

---

## 7. Validation Framework

### 7.1 Model Performance Validation

| Criterion | Minimum | Target | Method |
|-----------|---------|--------|--------|
| CV AUROC | > baseline | > 0.65 | 5-fold CV |
| CV AUROC variance | < 0.05 | < 0.03 | Std across folds |
| AUROC improvement | > 0 | > 0.05 | vs. mean pooling baseline |

### 7.2 Known Locus Validation

**Primary Target:** GPR126/ADGRG6 (rs6570507)

| Criterion | Expected | Validation Method |
|-----------|----------|-------------------|
| GPR126 in top genes | Top 5% | Rank across all annotated genes |
| Intronic signal | > 30% | Region attribution within GPR126 |
| rs6570507 detected | Present | Top 100 variants in GPR126 |

**Statistical Test:** Permutation test (n=1000) comparing GPR126 rank to null distribution.

### 7.3 Biological Plausibility Checks

| Check | Method |
|-------|--------|
| Pathway enrichment | Top 100 genes → Gene Ontology/KEGG |
| Tissue specificity | Overlap with spine/bone expression QTLs |
| Regulatory enrichment | Overlap with ENCODE enhancer annotations |

### 7.4 Confounding Assessment

| Confounder | Check |
|------------|-------|
| Ancestry | PCs alone should not predict (AUROC ~0.5) |
| Batch effects | No batch clustering in embeddings |
| Age/Sex | Attribution patterns consistent across strata |

### 7.5 Validation Checklist

| # | Check | Pass Criterion | Status |
|---|-------|----------------|--------|
| 1 | Model converges | Loss decreases | ⬜ |
| 2 | No overfitting | Val AUROC ≥ Train AUROC - 0.05 | ⬜ |
| 3 | Beats baseline | AUROC > mean pooling | ⬜ |
| 4 | CV stability | AUROC std < 0.05 | ⬜ |
| 5 | GPR126 recovered | Top 5% of genes | ⬜ |
| 6 | Intronic signal | > 30% in GPR126 | ⬜ |
| 7 | No ancestry confounding | PCs AUROC ~0.5 | ⬜ |
| 8 | Biologically plausible | Pathway enrichment p < 0.05 | ⬜ |

---

## 8. Implementation Timeline

### 8.1 Gantt Chart

```
Stage    1    2    3    4    5    6    7    8    9    10
        ─────────────────────────────────────────────────
Phase 0 ████                                              Cohort Definition
Phase 1      ████                                         Variant Extraction
Phase 2      ░░░░████                                     VEP Annotation
Phase 3           ████████                                Embedding Computation
Phase 4                ░░░░████                           Dataset Implementation
Phase 5                     ████                          Baseline Models
Phase 6                          ████                     Full Model v1
Phase 7                               ████████            Training & Tuning
Phase 8                                    ░░░░████       Attribution Pipeline
Phase 9                                         ████      Validation & Reports
        ─────────────────────────────────────────────────
```

### 8.2 Detailed Phase Breakdown

| Phase | Stage | Description | Deliverable | Dependencies |
|-------|------|-------------|-------------|--------------|
| **0** | 1 | Cohort definition | `cohort.parquet` | UK Biobank access |
| | | - Ancestry QC | | |
| | | - Case identification | | |
| | | - Control matching | | |
| **1** | 2 | Variant extraction | `variants.parquet`, `dosages.zarr` | Phase 0 |
| | | - QC filtering | | |
| | | - Dosage extraction | | |
| **2** | 2-3 | VEP annotation | `vep_annotations.parquet` | Phase 1 |
| | | - Run VEP | | |
| | | - Build gene index | | |
| **3** | 3-4 | Delta embeddings | `delta_embeddings.zarr` | Phase 1 |
| | | - HyenaDNA setup | | |
| | | - Parallel computation | | |
| **4** | 4-5 | Dataset implementation | Working DataLoaders | Phases 1-3 |
| | | - PyTorch Dataset | | |
| | | - Collate function | | |
| **5** | 5 | Baseline models | Baseline AUROC | Phase 4 |
| | | - Mean pooling + LR | | |
| | | - Simple attention | | |
| **6** | 6 | Full model | Model code | Phase 4 |
| | | - Vectorized encoder | | |
| | | - Parameter verification | | |
| **7** | 7-8 | Training | Best checkpoint | Phases 5-6 |
| | | - 5-fold CV | | |
| | | - Hyperparameter tuning | | |
| **8** | 9 | Attribution | Attribution pipeline | Phase 7 |
| | | - Layer IG implementation | | |
| | | - GPR126 validation | | |
| **9** | 10 | Reports | Final deliverable | Phase 8 |
| | | - Patient reports | | |
| | | - Documentation | | |

### 8.3 Milestones

| Milestone | Target Stage | Success Criterion |
|-----------|-------------|-------------------|
| **M1:** Cohort ready | Stage 1 | ~15k samples with labels |
| **M2:** Data pipeline complete | Stage 5 | DataLoader returns valid batches |
| **M3:** Baseline established | Stage 5 | Mean pooling AUROC computed |
| **M4:** Model trained | Stage 8 | CV AUROC > baseline |
| **M5:** Validation complete | Stage 9 | GPR126 in top 5% |

---

## 9. Technical Requirements

### 9.1 Compute Resources

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **Embedding computation** | | |
| GPU | 1× V100 (32GB) | 4× A100 (40GB) |
| Time | ~96 hours | ~24 hours |
| **Model training** | | |
| GPU | 1× V100 (16GB) | 1× A100 (40GB) |
| Time per fold | ~2 hours | ~30 minutes |
| **Storage** | | |
| Raw data | 50 GB | 50 GB |
| Embeddings | 500 MB | 500 MB |
| Checkpoints | 5 GB | 5 GB |

### 9.2 Software Dependencies

```
# Core
python>=3.9
pytorch>=2.0
pytorch-lightning>=2.0
torchmetrics>=1.0

# Genomics
pgenlib>=0.90
pyfaidx>=0.7
ensembl-vep>=110

# HyenaDNA
transformers>=4.30
safetensors>=0.3

# Attribution
captum>=0.6

# Data
pandas>=2.0
numpy>=1.24
zarr>=2.14
pyarrow>=12.0

# ML utilities
scikit-learn>=1.2
scipy>=1.10

# Visualization
matplotlib>=3.7
seaborn>=0.12
```

### 9.3 Data Requirements

| Data | Source | Format |
|------|--------|--------|
| Genotypes | UK Biobank | .pgen/.pvar/.psam |
| Phenotypes | UK Biobank | .csv (field 41270, 21000) |
| Reference genome | Ensembl | hg38.fa |
| VEP cache | Ensembl | GRCh38 cache |

### 9.4 UK Biobank Application

Required fields:

| Field ID | Description |
|----------|-------------|
| 22009 | Genetic PCs |
| 22006 | Genetic ethnic grouping |
| 21000 | Ethnic background |
| 41270 | Diagnoses - ICD10 |
| 31 | Sex |
| 21003 | Age at recruitment |
| 22000 | Genotype batch |

---

## 10. Risk Assessment

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HyenaDNA embedding computation too slow | Medium | High | Pre-filter variants by LD; use multiple GPUs |
| Model overfits | Medium | High | Strong regularization; cross-validation |
| Insufficient cases | Low | High | Broaden to all idiopathic scoliosis |
| Memory issues with large tensors | Medium | Medium | Zarr chunking; gradient checkpointing |

### 10.2 Scientific Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| No signal above baseline | Medium | High | Validate embeddings capture known eQTLs first |
| GPR126 not recovered | Medium | Medium | Check variant coverage at locus; consider LD |
| Confounding by ancestry | Low | High | Strict matching; include PCs as covariates |
| Attributions not interpretable | Medium | Medium | Compare IG with attention weights; use counterfactuals |

### 10.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| UK Biobank access delays | Low | High | Apply early; have backup dataset |
| GPU availability | Medium | Medium | Book compute in advance; cloud fallback |
| Staff turnover | Low | Medium | Comprehensive documentation |

---

## Appendices

### A. Known AIS GWAS Loci

| Locus | Lead SNP | Gene | Chr | Region | Reference |
|-------|----------|------|-----|--------|-----------|
| 6q24.1 | rs6570507 | GPR126 | 6 | Intron | Kou 2013 |
| 10q24.31 | rs11190870 | LBX1 | 10 | Near gene | Takahashi 2011 |
| 20p11.22 | rs6137473 | PAX1 | 20 | Intergenic | Sharma 2011 |
| 9p22.2 | rs3904778 | BNC2 | 9 | Intron | Ogura 2015 |

### B. Region Type Definitions

| Region | Definition |
|--------|------------|
| Promoter | TSS - 2000bp to TSS + 500bp |
| 5' UTR | 5' untranslated region |
| Exon | Protein-coding exonic sequence |
| Intron | Intronic sequence |
| 3' UTR | 3' untranslated region |
| Splice | Within 8bp of exon boundary |
| Downstream | TSS + 500bp to TES + 500bp (non-coding) |

### C. Glossary

| Term | Definition |
|------|------------|
| AIS | Adolescent Idiopathic Scoliosis |
| AUROC | Area Under Receiver Operating Characteristic |
| AUPRC | Area Under Precision-Recall Curve |
| CV | Cross-Validation |
| Delta embedding | Difference between ALT and REF sequence embeddings |
| IG | Integrated Gradients |
| LD | Linkage Disequilibrium |
| MAF | Minor Allele Frequency |
| PC | Principal Component |
| VEP | Variant Effect Predictor |

### D. References

1. Kou I, et al. (2013). Genetic variants in GPR126 are associated with adolescent idiopathic scoliosis. *Nature Genetics*, 45(6), 676-679.

2. Nguyen E, et al. (2023). HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution. *arXiv:2306.15794*.

3. McLaren W, et al. (2016). The Ensembl Variant Effect Predictor. *Genome Biology*, 17(1), 122.

4. Sundararajan M, et al. (2017). Axiomatic Attribution for Deep Networks. *ICML*.

5. Bycroft C, et al. (2018). The UK Biobank resource with deep phenotyping and genomic data. *Nature*, 562(7726), 203-209.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | January 2026 | - | Initial release |

---

*This document is intended for research planning purposes. Implementation details may be adjusted based on preliminary results and resource availability.*
