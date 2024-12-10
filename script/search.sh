source set.sh

for data in "${datasets[@]}"; do
  for L in {1000,5000,10000,50000,100000,250000,500000}; do
    if [ $data == "sift" ]; then
      E=1
    elif [ $data == "deep1M" ]; then
      E=5
    elif [ $data == "deep100M" ]; then
      E=20
    elif [ $data == "gist" ]; then
      E=50
    elif [ $data == "msong" ]; then
      E=10
    fi
  ./build/src/test_search_hnsw1D -d ${data} -s "${store_path}/${data}/" -l $L -k 1 -e $E
  ./build/src/test_search_hnsw2D -d ${data} -s "${store_path}/${data}/" -l $L -k 1 -e $E
done
done
