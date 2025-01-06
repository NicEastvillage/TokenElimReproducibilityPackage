#!/bin/bash
cd "$(dirname "$0")"

# Run 1 query for every 15th model using timeout of 10
./run_pipeline_parallel.sh ae_tapaal ../bin/verifypn-tokelim-linux64 tapaal 10 15 1 &
./run_pipeline_parallel.sh ae_dynamic ../bin/verifypn-tokelim-linux64 dynamic 10 15 1 &
./run_pipeline_parallel.sh ae_static ../bin/verifypn-tokelim-linux64 static 10 15 1 &

wait

echo "DONE"
