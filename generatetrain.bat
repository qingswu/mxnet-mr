python make_list.py chars chars/chars --recursive=True
"x64/release/im2rec" chars/chars.lst chars/ chars/train.rec
python generatesynsetwords.py --datadir=chars
pause