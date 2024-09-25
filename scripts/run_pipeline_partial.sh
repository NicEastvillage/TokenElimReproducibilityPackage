#!/bin/bash

NAME=$1
BIN=$2
METHOD=$3

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

MODELS_DIR="../MCC2023-CTL"
LOGS_DIR="../logs/$NAME"

chmod u+x "$(dirname "$BIN")/"
rm -rf $LOGS_DIR
mkdir -p $LOGS_DIR

function process_queries() {
  local offset=$1
  local step=$2
  local models=$(ls $MODELS_DIR)

  for (( i=offset; i<${#models[@]}; i+=step )); do
    MODEL="${models[$i]}"
    for CATEGORY in "CTLCardinality" "CTLFireability"; do
      mkdir -p "$LOGS_DIR/$MODEL/$CATEGORY"
      ./run_single.sh $NAME $BIN "$OPTIONS" $METHOD $MODEL $CATEGORY 1 10
    done
  done
}

# Every 15th model, 3 processes in parallel
process_queries 0 45 &
process_queries 15 45 &
process_queries 30 45 &

wait

./extract.sh $NAME
