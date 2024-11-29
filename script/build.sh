cd ..

for index_type in hnsw; do
  for data in {sift,gist,deep1M}; do
    DATA=/DATA
    data_path="${DATA}/${data}/${data}_base.fvecs"
    query_path="${DATA}/${data}/${data}_query.fvecs"
    index_path="./DATA/faiss_${data}.${index_type}"
    segment_path="./DATA/${data}_${index_type}_segment.log"
    ./cmake-build-debug/test_segment_${index_type} -n ${data_path} -i ${index_type} -r ${segment_path} -q ${query_path}
    python ./data/${index_type}.py -e 200 -s ${segment_path} -d ${data}
  done
done
