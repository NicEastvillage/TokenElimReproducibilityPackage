#!/bin/bash
cd "$(dirname "$0")"

# Run all queries of all models with a 30 minute timeout
./run_pipeline_parallel.sh ae_tapaal ../bin/verifypn-tokelim-linux64 tapaal 30 1 16 &
./run_pipeline_parallel.sh ae_dynamic ../bin/verifypn-tokelim-linux64 dynamic 30 1 16 &
./run_pipeline_parallel.sh ae_static ../bin/verifypn-tokelim-linux64 static 30 1 16 &

wait

echo "DONE"
