#!/bin/bash
while IFS= read -r url; do
   
    if [ -z "$url" ]; then
        continue
    fi
    

    path=$(echo "$url" | sed 's|.*/codonrl-checkpoints/||')
    
    if [ -n "$path" ]; then
  
        full_path="checkpoints/$path"
        dir=$(dirname "$full_path")
        mkdir -p "$dir"
        

        echo "Downloading to $full_path..."
        wget -O "$full_path" "$url"
    fi
done < checkpoint_urls.txt
