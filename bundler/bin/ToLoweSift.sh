#!/bin/sh

for file in "$@"; do
    echo "Treating $file" 
    if [ "${file##*.}" == "gz" ]; then
        echo "Uncompressing to ${file%.key.gz}.key"
        gzip -d "$file"
        echo "Converting ${file%.key.gz}.sift"
        cat "${file%.key.gz}.key" | wc -l | awk '{ print $1 " 128" }' > "${file%.key.gz}.sift"
        cat "${file%.key.gz}.key" | awk 'BEGIN { split("4 24 44 64 84 104 124 132", offsets); } { i1 = 0; tmp = $1; $1 = $2; $2 = tmp; for (i=1; i<9; i++) { i2 = offsets[i]; out = ""; for (j=i1+1; j<=i2; j++) { if (j != i1+1) { out = out " " }; out = out $j }; i1 = i2; print out } }' >> "${file%.key.gz}.sift"
        echo "Replacing ${file%.key.gz}.sift for ${file%.key.gz}.key"
        rm "${file%.key.gz}.key"
        mv ${file%.key.gz}.sift ${file%.key.gz}.key
        echo "Compressing ${file%.key.gz}.key"
        gzip -f ${file%.key.gz}.key
    else
        echo "Converting to ${file%.key}.sift"
        cat "$file" | wc -l | awk '{ print $1 " 128" }' > "${file%.key}.sift"
        cat "$file" | awk 'BEGIN { split("4 24 44 64 84 104 124 132", offsets); } { i1 = 0; tmp = $1; $1 = $2; $2 = tmp; for (i=1; i<9; i++) { i2 = offsets[i]; out = ""; for (j=i1+1; j<=i2; j++) { if (j != i1+1) { out = out " " }; out = out $j }; i1 = i2; print out } }' >> "${file%.key}.sift"
        echo "Replacing ${file%.key}.sift for ${file%.key}.key"
        rm "${file%.key}.sift"
        mv ${file%.key}.sift ${file%.key}.key
        echo "Compressing ${file%.key}.key"
        gzip -f ${file%.key}.key
    fi
done
