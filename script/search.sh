source set.sh

for data in "${datasets[@]}"; do
  for L in {500000,250000,125000,10000,5000}; do

  ./cmake-build-debug/src/test_search_hnsw1D -d ${data} -s "${store_path}/${data}/" -l 100000 -k 1 -e 50

  ./cmake-build-debug/src/test_search_hnsw2D -d ${data} -s "${store_path}/${data}/" -l 100000 -k 1 -e 50
  done

done
