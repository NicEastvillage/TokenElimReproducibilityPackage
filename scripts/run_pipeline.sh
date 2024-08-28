#!/bin/bash

NAME=$1
BIN=$2
METHOD=$3

OPTIONS=" "
CAT="CTL"

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

for MODEL in $(ls $MODELS_DIR) ; do

  for SUBCAT in "Cardinality" "Fireability"; do

    CATEGORY="${CAT}${SUBCAT}"

    mkdir -p "$LOGS_DIR/$MODEL/$CATEGORY"
    Q_NUM=$(egrep -c "<property>" "$MODELS_DIR/$MODEL/$CATEGORY.xml")
    for QUERY in $(seq 1 $Q_NUM) ; do
      ./run_single.sh $NAME $BIN "$OPTIONS" $METHOD $MODEL $CATEGORY $QUERY
    done
  done
done

./extract.sh $NAME
