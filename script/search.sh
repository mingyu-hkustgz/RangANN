cd ..

for data in {sift,gist,deep1M}; do
  for index_type in {hnsw,ivf}; do

    DATA=/home/DATA/vector_data
    data_path="${DATA}/${data}/${data}_base.fvecs"
    query_path="${DATA}/${data}/${data}_query.fvecs"
    index_path="./DATA/faiss_${data}.${index_type}"
    segment_path="./DATA/${data}_${index_type}_segment.log"
    for bound in {100000,200000,400000}; do
      for efSearch in {8,32,64,96,128,256}; do
        ./cmake-build-debug/test_segment_${index_type} -n ${data_path} -q ${query_path} -i ${index_path} -r ${segment_path} -b ${bound} -s ${efSearch} -l "./logger/${data}_${index_type}_result_${bound}.log" &
      done
      wait
    done
  done
done
