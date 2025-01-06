#!/bin/bash
cd "$(dirname "$0")"

chmod +x *.sh
chmod +x ../bin/*

if [ ! -d "MCC2023-CTL" ]; then
  tar -xvf MCC2023-CTL.tar.gz
else
  echo "MCC2023-CTL.tar.gz has already been unpacked"
fi

pip install -r requirements.txt
