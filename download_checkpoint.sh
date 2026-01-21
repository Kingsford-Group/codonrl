#!/bin/bash


LINKS_URL="https://datarnadesign.blob.core.windows.net/codonrl-checkpoints/download_links.txt"
BASE_DIR="checkpoints" 

echo "========================================"
echo "Downloading from public links"
echo "========================================"
echo "Downloading from $LINKS_URL"
echo "Downloading to $BASE_DIR"
echo ""


mkdir -p "$BASE_DIR"

# Download link list file
echo "Getting download link list..."
TEMP_LINKS=$(mktemp)
wget -q -O "$TEMP_LINKS" "$LINKS_URL"

if [ $? -ne 0 ]; then
    echo "✗ Failed to get download link list"
    exit 1
fi

echo "✓ Successfully got download link list"
echo ""

total=0
success=0
failed=0


total_lines=$(grep -c "^https://" "$TEMP_LINKS")
echo "Found $total_lines files to download"
echo ""

# Read each URL and download
while IFS= read -r url; do
    # Skip empty lines
    if [ -z "$url" ]; then
        continue
    fi
    
    # Skip lines that are not https
    if [[ ! "$url" =~ ^https:// ]]; then
        continue
    fi
    
    ((total++))
    
    # Extract path from URL
    # URL : https://datarnadesign.blob.core.windows.net/codonrl-checkpoints/{subdir}/{filename}
    # Extract {subdir}/{filename} part
    path=$(echo "$url" | sed 's|https://datarnadesign.blob.core.windows.net/codonrl-checkpoints/||')
    
    # Extract subdirectory and filename
    subdir=$(dirname "$path")
    filename=$(basename "$path")
    
    LOCAL_FILE="${BASE_DIR}/${path}"
    LOCAL_DIR="${BASE_DIR}/${subdir}"
    
 
    mkdir -p "$LOCAL_DIR"
    
 
    if [[ "$filename" == *.pth ]]; then
        file_type="checkpoint"
    else
        file_type="summary"
    fi
    
    echo "[$total/$total_lines] Downloading ${path} (${file_type}) ..."
    
    # Download using wget
    wget -q --show-progress -O "$LOCAL_FILE" "$url" 2>&1 | grep -v "^$"
    
    if [ $? -eq 0 ] && [ -f "$LOCAL_FILE" ]; then
        file_size=$(du -h "$LOCAL_FILE" | cut -f1)
        echo "  ✓ Completed (${file_size})"
        ((success++))
    else
        echo "  ✗ Failed"
        ((failed++))
   
        rm -f "$LOCAL_FILE"
    fi
    

    if [ $((total % 10)) -eq 0 ]; then
        echo ""
        echo "--- Progress: $total/$total_lines (Success: $success, Failed: $failed) ---"
        echo ""
    fi
    
done < "$TEMP_LINKS"


rm -f "$TEMP_LINKS"

echo ""
echo "========================================"
echo "Download completed!"
echo "========================================"
echo "Total: $total files"
echo "Success: $success files"
echo "Failed: $failed files"
echo "Download directory: $(pwd)/$BASE_DIR"
echo "========================================"


echo ""
echo "Directory structure:"
ls -lh "$BASE_DIR" | head -20
if [ $(ls "$BASE_DIR" | wc -l) -gt 20 ]; then
    echo "... more directories"
fi
