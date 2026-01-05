#!/bin/bash

# Configuration
JSON_FILE="papers/papers.json"
PDF_OUT_DIR="pdf-output"
ROOT_DIR=$(pwd)

mkdir -p "$PDF_OUT_DIR"

# 1. Extract the list of slugs exactly like the YAML matrix did
MAP=$(jq -r '
  .sections[].items[] as $item |
  $item.slug as $slug |
  (
    if ($item.children? | length > 0) then
      $item.children[] | "\($slug)/\(.slug)"
    else
      $slug
    end
  )
' "$JSON_FILE")

# 2. Iterate through each slug and compile
for SLUG in $MAP; do
    echo "------------------------------------------------"
    echo "Building PDFs for slug: $SLUG"
    echo "------------------------------------------------"

    SLUG_DIR="$ROOT_DIR/papers/$SLUG"
    
    if [ ! -d "$SLUG_DIR" ]; then
        echo "WARNING: Folder '$SLUG_DIR' does not exist. Skipping."
        continue
    fi

    # Find all .tex files excluding common/glossary
    find "$SLUG_DIR" -maxdepth 1 -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | while read -r texfile; do
        DIR_PATH=$(dirname "$texfile")
        BASE_NAME=$(basename "$texfile" .tex)
        
        echo "Compiling $BASE_NAME in $DIR_PATH..."

        # Navigate to the tex directory to handle relative includes/bib
        pushd "$DIR_PATH" > /dev/null
        
        # Run latexmk
        latexmk -pdf -interaction=nonstopmode -halt-on-error "$(basename "$texfile")"
        
        # Prepare the output filename (flattening slashes to dashes)
        SAFE_SLUG="${SLUG//\//-}"
        FINAL_PDF_NAME="$SAFE_SLUG-$BASE_NAME.pdf"

        # Copy resulting PDF to the global output folder
        if [ -f "$BASE_NAME.pdf" ]; then
            cp "$BASE_NAME.pdf" "$ROOT_DIR/$PDF_OUT_DIR/$FINAL_PDF_NAME"
            echo "Successfully created $FINAL_PDF_NAME"
        else
            echo "ERROR: Failed to generate $BASE_NAME.pdf"
        fi

        popd > /dev/null
    done
done
