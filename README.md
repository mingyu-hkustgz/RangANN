# ElasticGraph: Elastic Graphs for Range-Filtering Approximate k Nearest Neighbor Search

## Prerequisites
* C++ require:
* OpenMP
* AVX512

## Dataset
* We recommend start from datasets available online (SIFT and DEEP)
  * The following links may be useful https://github.com/KGLab-HDU/TKDE-under-review-Native-Hybrid-Queries-via-ANNS
* The WIT dataset need  manually crawl and generate using ResNet
  * The link provide image url https://www.kaggle.com/c/wikipedia-image-caption/overview

## Reproduction
1. set the **store_path** and dataset in set.sh
2. run ```bash run.sh```

## Note
1. All dataset stoe in fvecs format.
2. The dataset path need arrange as
--{store_path}/{dataset_name}
   --{dataset_name}_base.fvecs
   --{dataset_name}_query.fvecs

## Baseline Notice
* HBI1D--ESG1D in our paper
* HBI2D--ESG2D in our paper
* SEG  Segment Tree-based method

## Index Size Note
* For HBI1D-ESG1D we redundant store base vectors to keep with the origin hnswlib memory layout
* For HBI2D-ESG2D we provide a static version of hnswlib that allow the share of same base vectors 
