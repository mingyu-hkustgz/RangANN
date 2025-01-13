source set.sh

for data in "${datasets[@]}"; do
  for L in {500000,250000,125000,62500,31250,15625,7812,3906}; do
    if [ $data == "sift" ]; then
      E=1
    elif [ $data == "deep" ]; then
      E=5
    elif [ $data == "paper" ]; then
      E=10
    elif [ $data == "WIT" ]; then
      E=10
    elif [ $data == "glove100d" ]; then
      E=100
    elif [ $data == "deep100M" ]; then
      E=20
    elif [ $data == "sift100m" ]; then
      E=20
    fi
  ./build/src/test_search_hnsw2DS -d ${data} -s "${store_path}/${data}/" -l $L -k 10 -e $E
done
done
