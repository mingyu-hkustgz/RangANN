source set.sh

for data in "${datasets[@]}"; do
for L in {800000,600000,400000,200000,100000,50000}; do
    if [ $data == "sift" ]; then
      E=1
    elif [ $data == "deep" ]; then
      E=5
    elif [ $data == "msong" ]; then
      E=10
    elif [ $data == "msmarc-small" ]; then
      E=10
    elif [ $data == "deep100M" ]; then
      E=20
    elif [ $data == "sift100m" ]; then
      E=20
    fi
  ./build/src/test_search_hnsw1D -d ${data} -s "${store_path}/${data}/" -l $L -k 1 -e $E
done
done

for data in "${datasets[@]}"; do
  for L in {500000,250000,125000,62500,31250,15625,3906,1953}; do
    if [ $data == "sift" ]; then
      E=1
    elif [ $data == "msong" ]; then
      E=10
    elif [ $data == "deep" ]; then
      E=5
    elif [ $data == "msmarc-small" ]; then
      E=10
    elif [ $data == "deep100M" ]; then
      E=20
    elif [ $data == "sift100m" ]; then
      E=20
    fi
  ./build/src/test_search_hnsw2D_SEG -d ${data} -s "${store_path}/${data}/" -l $L -k 1 -e $E
  ./build/src/test_search_hnsw2D_HALF -d ${data} -s "${store_path}/${data}/" -l $L -k 1 -e $E
done
done
