source set.sh

for data in "${datasets[@]}"; do
  for L in {500000,250000,31250,15625,3906,1953}; do
    if [ $data == "sift" ]; then
      E=1
    elif [ $data == "deep" ]; then
      E=5
    elif [ $data == "WIT" ]; then
      E=10
    elif [ $data == "audio" ]; then
      E=10
    elif [ $data == "glove100d" ]; then
      E=100
    elif [ $data == "deep100M" ]; then
      E=20
    elif [ $data == "sift100m" ]; then
      E=20
    fi

#  ./build/src/test_search_hnsw2DF -d ${data} -s "${store_path}/${data}/" -l $L -k 10 -e $E -f 2
#
#  ./build/src/test_search_hnsw2DF -d ${data} -s "${store_path}/${data}/" -l $L -k 10 -e $E -f 4
#
#  ./build/src/test_search_hnsw2DF -d ${data} -s "${store_path}/${data}/" -l $L -k 10 -e $E -f 8

  ./build/src/test_search_hnsw2DF -d ${data} -s "${store_path}/${data}/" -l $L -k 10 -e $E -f 16
done
done
