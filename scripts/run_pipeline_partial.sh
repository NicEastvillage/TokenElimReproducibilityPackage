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

# Run cardinality/fireability query 1 of every 10th model with a timeout of 15 minutes
for MODEL in $(ls $MODELS_DIR | awk 'NR % 10 == 0') ; do

  for CATEGORY in "CTLCardinality" "CTLFireability"; do

    mkdir -p "$LOGS_DIR/$MODEL/$CATEGORY"
    ./run_single.sh $NAME $BIN "$OPTIONS" $METHOD $MODEL $CATEGORY 1 15
  done
done

./extract.sh $NAME
