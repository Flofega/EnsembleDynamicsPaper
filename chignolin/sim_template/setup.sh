#!/bin/bash

# List of files to copy (edit this with your files)
FILES=("plumed.dat" "md.mdp" "topol.top")

# Create 10 directories (0–9) and copy files
for i in $(seq 0 19); do
    mkdir -p "$i"
    cp -r "charmm22star.ff" "$i/"
    # create folder if it doesn’t exist
    cp "$i.gro" "$i/"
    for f in "${FILES[@]}"; do
        if [[ -f "$f" ]]; then
            cp "$f" "$i/"
        else
            echo "Warning: $f not found, skipping..."
        fi
    done
done

echo "Done!"

