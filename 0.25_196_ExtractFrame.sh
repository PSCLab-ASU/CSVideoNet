#!/bin/bash
mkdir -p "0.25_196/"
for f in *.avi
      do

      ffmpeg -ss 1 -i "$f" -r 0.25 -s 196x196 "0.25_196/${f%.avi}"_%02d.jpg
done
