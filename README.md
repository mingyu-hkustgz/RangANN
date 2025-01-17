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

## Data Path
1. All dataset stoe in fvecs format.
2. The dataset path need arrange as
--{store_path}/{dataset_name}/
   --{dataset_name}_base.fvecs
   --{dataset_name}_query.fvecs

## RANGE
* we use range from [500000,250000,125000,62500,31250,15625,7812,3906,0] to evaluate 1 million scale dataset
  * 0 is the mix range set that random set left and right bound

## Baseline Notice
* HBI1D--ESG1D in our paper
* HBI2D--ESG2D in our paper
* HBI2D{%d}--ESG2D with large fanout
* SEG  Segment Tree-based method
* HBI2DS SuperPostFiltering with elastic factor of 1/2
  * The elastic factor is fixed

## Index Size Note
* For HBI1D--ESG1D we redundant store base vectors to keep with the origin hnswlib memory layout
* For HBI2D--ESG2D we provide a static version of hnswlib that allow the share of same base vectors 
* The HBI2D{%d}--ESG2D build the top layer, and HBI2D--ESG2D remove the top layer for fast index
    * Interestingly, splitting the top layer into two parts sometimes works better and sometimes doesn't
## Index Parameter Set
* The efconstruct and M are set by definition in ./include/Index*.h
  * Default parameter is M=16 and efconstruct=200

## Evaluation Note
* We found that there are duplicate points in data sets such as sift, so we judge that if the distance between the returned nearest neighbor (Top 1) and the groundtruth relative to the query is almost the same (relative error < 0.01%), we also consider that the nearest neighbor has been found
  * (we also use the same strategy for SERF and IRANGE).
* For Top k, we still use the original recall calculation method

