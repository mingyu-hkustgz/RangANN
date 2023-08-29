for data in gist; do

  DATA=/home/DATA/vector_data
  data_path="${DATA}/${data}/${data}_base.fvecs"
  query_path="${DATA}/${data}/${data}_query.fvecs"
  index_path="./DATA/faiss_${data}.hnsw"
  segment_path="./DATA/${data}_segment.log"

  for bound in {100000,200000,400000}; do
    for efSearch in {32,64,96,128,160,192,256}; do
      ./cmake-build-debug/main -n ${data_path} -q ${query_path} -i ${index_path} -r ${segment_path} -b ${bound} -s ${efSearch} -l "./logger/${data}_result_${bound}.log"
    done

  done

done
