#!/bin/bash
cd "$(dirname "$0")"

# Run 1 query for every 20000th model using timeout of 1
./run_pipeline_parallel.sh ae_tapaal ../bin/verifypn-tokelim-linux64 tapaal 1 20000 1 &
./run_pipeline_parallel.sh ae_dynamic ../bin/verifypn-tokelim-linux64 dynamic 1 20000 1 &
./run_pipeline_parallel.sh ae_static ../bin/verifypn-tokelim-linux64 static 1 20000 1 &

wait

echo "DONE"
