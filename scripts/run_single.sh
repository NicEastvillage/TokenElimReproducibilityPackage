#!/bin/bash

NAME=$1
BIN=$2
OPTIONS="$3"
METHOD=$4
MODEL=$5
CAT=$6
QUERY=$7
TIMEOUT=$8

if [ -z "$TIMEOUT" ] ; then
	TIMEOUT=30
fi

MODELS_DIR="../MCC2023-CTL"
LOGS_DIR="../logs/$NAME/$MODEL/$CAT"
OUT="$LOGS_DIR/$QUERY.log"
mkdir -p $LOGS_DIR

ENVS=""
if [[ "$METHOD" == "tapaal" ]]; then
  ENVS=""
elif [[ "$METHOD" == "dynamic" ]]; then
  ENVS="env TAPAAL_TOKEN_ELIM=on "
elif [[ "$METHOD" == "static" ]]; then
  ENVS="env TAPAAL_TOKEN_ELIM=on TAPAAL_TOKEN_ELIM_STATIC=on "
else
  echo "Unknown mode: Expect 'tapaal', 'dynamic', or 'static'"
  exit
fi

let "TIME=$TIMEOUT*60"
let "MEM=15*1024*1024"
ulimit -v $MEM

echo "Processing $MODEL, $CAT query $QUERY ..."

CMD="$ENVS./"$BIN" -s RDFS "$OPTIONS" $MODELS_DIR/$MODEL/model.pnml $MODELS_DIR/$MODEL/$CAT.xml -x $QUERY"
RAW=$(eval "/usr/bin/time -f \"@@@%e,%M@@@\" timeout $TIME $CMD" 2>&1)

echo "$RAW" > $OUT
