#!/bin/sh
python make_list.py chars chars/chars --recursive=True
im2rec chars/chars.lst chars/ chars/train.rec
python generatesynsetwords.py --datadir=chars
