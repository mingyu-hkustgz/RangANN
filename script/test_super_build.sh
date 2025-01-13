source set.sh

for data in "${datasets[@]}"; do

  log_file="./results/time-log/${data}/HNSW2D-Index-time-Super-2.log"
  start_time=$(date +%s)
  ./build/src/test_build_hnsw2DS -d ${data} -s "${store_path}/${data}/" -i "./DATA/${data}/"
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "HNSW2D Super Index time: ${duration}(s)" | tee -a ${log_file}

done
