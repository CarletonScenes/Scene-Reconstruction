#!/bin/bash
#
# ToSift.sh
# Create a script for extracting sift features from a set of images

# Set this variable to your base install path (e.g., /home/foo/bundler)
BIN_PATH=$(dirname $(which $0));

IMAGE_DIR="."

if [ $# -eq 1 ]
then
    IMAGE_DIR=$1
fi

SIFT=$BIN_PATH/sift

if ! [ -e $SIFT ] ; then
    echo "[ToSift] Error: SIFT not found.  Please install SIFT to $BIN_PATH" > /dev/stderr
    exit 1
fi

for d in `ls -1 $IMAGE_DIR | egrep "jpg$"`
do 
    pgm_file=$IMAGE_DIR/`echo $d | sed 's/jpg$/pgm/'`
    key_file=$IMAGE_DIR/`echo $d | sed 's/jpg$/key/'`
    echo "wc -l $key_file.vlfeat | awk '{ print \$1 \" 128\" }' > $key_file"
    echo "awk 'BEGIN { split("4 24 44 64 84 104 124 132", offsets); } { i1 = 0; tmp = $1; $1 = $2; $2 = tmp; for (i=1; i<9; i++) { i2 = offsets[i]; out = ""; for (j=i1+1; j<=i2; j++) { if (j != i1+1) { out = out " " }; out = out $j }; i1 = i2; print out } }' >> $key_file.key”
    echo "rm $key_file.vlfeat"
done
