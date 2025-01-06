#!/bin/bash

NAME="$1"
BIN="$2"
METHOD="$3"
TIMEOUT="$4"
STEPSIZE="$5"
QUERY_LIMIT="$6"

OPTIONS=" "

if [ -z "$NAME" ] ; then
	echo "Missing benchmark name"
	exit
fi

if [ -z "$BIN" ] ; then
	echo "Missing binary"
	exit
fi

if [ ! -f "$BIN" ] ; then
	echo "Binary does not exist"
	exit
fi

if [ -z "$METHOD" ] ; then
	echo "Missing method"
	exit
elif ! [[ "$METHOD" =~ ^(tapaal|dynamic|static)$ ]] ; then
  echo "Unknown method: '$METHOD', expected tapaal, dynamic, or static"
  exit
fi

if [ -z "$TIMEOUT" ] ; then
	echo "Missing timeout"
	exit
fi

if [ -z "$STEPSIZE" ] ; then
  STEPSIZE=1
fi

if [ -z "$QUERY_LIMIT" ] ; then
  QUERY_LIMIT=16
fi

MODELS_DIR="../MCC2023-CTL"
LOGS_DIR="../logs/$NAME"

chmod u+x "$(dirname "$BIN")/"
rm -rf $LOGS_DIR
mkdir -p $LOGS_DIR

function process_queries() {
  local offset=$1
  local step=$2
  local models=($MODELS_DIR/*)

  for (( i=offset; i<${#models[@]}; i+=step )); do
    MODEL=$(basename "${models[$i]}")
    for CATEGORY in "CTLCardinality" "CTLFireability"; do
      mkdir -p "$LOGS_DIR/$MODEL/$CATEGORY"
      for Q in $(seq 1 $QUERY_LIMIT) ; do
        ./run_single.sh $NAME $BIN "$OPTIONS" $METHOD $MODEL $CATEGORY $Q $TIMEOUT
      done
    done
  done
}

let "STEP_ONE=$STEPSIZE*1"
let "STEP_TWO=$STEPSIZE*2"
let "STEP_THREE=$STEPSIZE*3"

# Every $STEPSZIE'th model, 3 processes in parallel
process_queries 0         $STEP_THREE &
process_queries $STEP_ONE $STEP_THREE &
process_queries $STEP_TWO $STEP_THREE &

wait

./extract.sh $NAME
