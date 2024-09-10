#!/bin/bash

NAME=$1

if [ -z "$NAME" ] ; then
	echo "Missing benchmark name"
	exit
fi

MODELS_DIR="../MCC2023-CTL"
LOGS_DIR="../logs/$NAME"
CSV="../data/$NAME.csv"
rm -f $CSV
mkdir -p $(dirname "$CSV")

# Write header
echo "model;query;formula;category;subcategory;satisfied;time;verify time;memory;simplification;reachability;discovered;explored;tokens extrapolated;max tokens" > $CSV

for MODEL in $(ls $MODELS_DIR) ; do

  for CAT in "CTL" "Reachability"; do
    for SUBCAT in "Cardinality" "Fireability"; do

      CATEGORY="${CAT}${SUBCAT}"

      if ! [[ -d "$LOGS_DIR/$MODEL/$CATEGORY" ]]; then
        continue
      fi

      Q_NUM=$(egrep -c "<property>" "$MODELS_DIR/$MODEL/$CATEGORY.xml")
      for QUERY in $(seq 1 $Q_NUM) ; do

        LOG="$LOGS_DIR/$MODEL/$CATEGORY/$QUERY.log"
        echo "Extracting data from $LOG"

        RAW=$([[ -f $LOG ]] && cat "$LOG" | grep -v "^<" || echo "")

        # Extract fields
        FORMULA=$(grep -oP "Query after reduction: \K.*" <<< "$RAW" || echo "unknown")
        FORMULA=$([[ "${#FORMULA}" -gt 200 ]] && echo "too long" || echo "$FORMULA")

        if grep -q "Query is satisfied." <<< "$RAW"; then
          SATISFIED="true"
        elif grep -q "Query is NOT satisfied." <<< "$RAW"; then
          SATISFIED="false"
        elif grep -qP "(segmentation|exception|^ERROR|^error)" <<< "$RAW"; then
          SATISFIED="error"
        else
          SATISFIED="unknown"
        fi
        TIME=$(grep -oP "Spent \K[^\s]*(?= in total)" <<< "$RAW" || echo "-1")
        VERIFY_TIME=$(grep -oP "Spent \K[^\s]*(?= on verification)" <<< "$RAW" || echo "-1")
        MEMORY=$(grep -oP "@@@[^,]*,\K[^@]*(?=@@@)" <<< "$RAW" | awk '{res=$1*1000}END{print res}')
        SIMPLIFICATION=$(grep -q "Query solved by Query Simplification." <<< "$RAW" && echo "true" || echo "false")
        REACH=$(grep -q "discovered states:" <<< "$RAW" && echo "true" || echo "false")
        DISCOVERED=$(grep -oP "Configurations\s*: \K.*" <<< "$RAW" || grep -oP "discovered states:\s*\K.*" <<< "$RAW" || echo "0")
        EXPLORED=$(grep -oP "Explored Configs\s*: \K.*" <<< "$RAW" || grep -oP "explored states:\s*\K.*" <<< "$RAW" || echo "0")
        EXTRAP=$(grep -oP "Tokens Extrapolated\s*:\s*\K.*" <<< "$RAW" || echo "0")
        MAX_TOKENS=$(grep -oP "max tokens:\s*: \K.*" <<< "$RAW" || echo "0")

        # Add entry to CSV
        ENTRY="$MODEL;$QUERY;$FORMULA;$CAT;$SUBCAT;$SATISFIED;$TIME;$VERIFY_TIME;$MEMORY;$SIMPLIFICATION;$REACH;$DISCOVERED;$EXPLORED;$EXTRAP;$MAX_TOKENS"
        echo "$ENTRY" >> $CSV

      done
    done
  done
done

echo "Done"
