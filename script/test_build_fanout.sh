source set.sh

for data in "${datasets[@]}"; do

  log_file="./results/time-log/${data}/HNSW2D-Index-time-32.log"
  start_time=$(date +%s)
  ./build/src/test_build_hnsw2DF -d ${data} -s "${store_path}/${data}/" -i "./DATA/${data}/" -f 16
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW2D Index time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/HNSW2D-Index-time-16.log"
  start_time=$(date +%s)
  ./build/src/test_build_hnsw2DF -d ${data} -s "${store_path}/${data}/" -i "./DATA/${data}/" -f 16
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW2D Index time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/HNSW2D-Index-time-8.log"
  start_time=$(date +%s)
  ./build/src/test_build_hnsw2DF -d ${data} -s "${store_path}/${data}/" -i "./DATA/${data}/" -f 8
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW2D Index time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/HNSW2D-Index-time-4.log"
  start_time=$(date +%s)
  ./build/src/test_build_hnsw2DF -d ${data} -s "${store_path}/${data}/" -i "./DATA/${data}/" -f 4
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW2D Index time: ${duration}(s)" | tee -a ${log_file}

  log_file="./results/time-log/${data}/HNSW2D-Index-time-4.log"
  start_time=$(date +%s)
  ./build/src/test_build_hnsw2DF -d ${data} -s "${store_path}/${data}/" -i "./DATA/${data}/" -f 2
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW2D Index time: ${duration}(s)" | tee -a ${log_file}

done
