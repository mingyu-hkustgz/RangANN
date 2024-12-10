source set.sh

for data in "${datasets[@]}"; do

  log_file="./results/time-log/${data}/HNSW1D-Index-time.log"
  start_time=$(date +%s)
  ./build/src/test_build_hnsw1D -d ${data} -s "${store_path}/${data}/" -i "./DATA/${data}/"
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW1D Index time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/HNSW2D-Index-time.log"
  start_time=$(date +%s)
  ./build/src/test_build_hnsw2D -d ${data} -s "${store_path}/${data}/" -i "./DATA/${data}/"
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW2D Index time: ${duration}(s)" | tee -a ${log_file}

done
