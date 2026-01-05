#!/bin/bash

# 1. SETUP PATHS
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

JSON_FILE="$PROJECT_ROOT/papers/papers.json"
HTML_ROOT="$PROJECT_ROOT/html"
STYLE_CSS="$PROJECT_ROOT/style.css"
BIB_PATH="$PROJECT_ROOT/papers/references.bib"

# Check PDF directory - update this name if your build-pdf script uses something else
PDF_DIR="$PROJECT_ROOT/pdf-output" 
[ ! -d "$PDF_DIR" ] && PDF_DIR="$PROJECT_ROOT/all-pdfs"

mkdir -p "$HTML_ROOT"
cp "$STYLE_CSS" "$HTML_ROOT/"
touch "$HTML_ROOT/.nojekyll"

# 2. PRE-PROCESS NAVIGATION MAP
FLAT_JSON=$(jq -c '[ .sections[].items[] | .slug as $pslug | {title, slug} , (select(.children) | .children[] | {title, slug: "\($pslug)/\(.slug)"}) ]' "$JSON_FILE")
TOTAL_ITEMS=$(echo "$FLAT_JSON" | jq 'length')

# 3. START MAIN INDEX.HTML
TITLE=$(jq -r '.title' "$JSON_FILE")
INDEX_FILE="$HTML_ROOT/index.html"
echo "<html><head><title>$TITLE</title><link rel='stylesheet' href='style.css'></head><body class='bodytext'><h1>$TITLE</h1>" > "$INDEX_FILE"

# 4. CORE PROCESSING FUNCTION
process_entry() {
    local title="$1"
    local full_slug="$2"
    local is_child="$3"

    TARGET_DIR="$HTML_ROOT/$full_slug"
    mkdir -p "$TARGET_DIR"

    # --- Fixed Depth Logic ---
    # Even depth 0 (top level) needs to go up 1 level to reach the root 'html/'
    DEPTH=$(echo "$full_slug" | tr -cd '/' | wc -c)
    TO_ROOT=".."
    for ((i=0; i<DEPTH; i++)); do TO_ROOT="../$TO_ROOT"; done
    

# --- Corrected Navigation Footer Logic ---
    IDX=$(echo "$FLAT_JSON" | jq -r "map(.slug == \"$full_slug\") | index(true)")
    PREV_NAV="[Start]"
    NEXT_NAV="[End]"
    
    if [ "$IDX" != "null" ]; then
        if [ "$IDX" -gt 0 ]; then
            P_DATA=$(echo "$FLAT_JSON" | jq -c ".[$IDX-1]")
            P_SLUG=$(echo "$P_DATA" | jq -r .slug)
            P_TITLE=$(echo "$P_DATA" | jq -r .title)
            # Use ${TO_ROOT}/ to reset the path to the root 'html/' folder
            PREV_NAV="<a href=\"${TO_ROOT}/${P_SLUG}/index.html\">&larr; ${P_TITLE}</a>"
        fi
        if [ "$IDX" -lt $((TOTAL_ITEMS - 1)) ]; then
            N_DATA=$(echo "$FLAT_JSON" | jq -c ".[$IDX+1]")
            N_SLUG=$(echo "$N_DATA" | jq -r .slug)
            N_TITLE=$(echo "$N_DATA" | jq -r .title)
            # Use ${TO_ROOT}/ to reset the path to the root 'html/' folder
            NEXT_NAV="<a href=\"${TO_ROOT}/${N_SLUG}/index.html\">${N_TITLE} &rarr;</a>"
        fi
    fi
    FOOTER_HTML="<hr><div class='nav-footer'><a href=\"${TO_ROOT}/index.html\">&uarr; Up (TOC)</a> | $PREV_NAV | $NEXT_NAV <br><br><i>&copy; 2026 The Abstract Universe Project.</i></div>"

    SRC_DIR="$PROJECT_ROOT/papers/$full_slug"
    TEX_INPUT=$(find "$SRC_DIR" -maxdepth 1 -name "*.tex" ! -name "common.tex" ! -name "glossary.tex" | head -n 1)

    if [ -f "$TEX_INPUT" ]; then
        [ -d "$SRC_DIR/figures" ] && cp -r "$SRC_DIR/figures" "$TARGET_DIR/"
        [ -d "$SRC_DIR/simulations" ] && cp -r "$SRC_DIR/simulations" "$TARGET_DIR/"
        find "$SRC_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.svg" \) -exec cp {} "$TARGET_DIR/" \;

        # We write the footer to a temporary file to ensure UTF-8 compliance
        TEMP_FOOTER=$(mktemp)
        echo "$FOOTER_HTML" > "$TEMP_FOOTER"

        pushd "$SRC_DIR" > /dev/null
        pandoc "$(basename "$TEX_INPUT")" -s --citeproc --bibliography="$BIB_PATH" \
            --mathjax="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" \
            -c "${TO_ROOT}/style.css" -A "$TEMP_FOOTER" \
            -o "$TARGET_DIR/index.html" --metadata title="$title"
        popd > /dev/null
        rm "$TEMP_FOOTER"
    fi

    # Handle PDF
    PDF_LINK_TOC=""
    if [ -d "$PDF_DIR" ]; then
        SAFE_SLUG="${full_slug//\//-}"
        PDF_FILE=$(find "$PDF_DIR" -type f -name "*$SAFE_SLUG*.pdf" | head -n 1)
        if [ -n "$PDF_FILE" ]; then
            cp "$PDF_FILE" "$TARGET_DIR/"
            PDF_BASENAME=$(basename "$PDF_FILE")
            PDF_LINK_TOC=" | <a href='./$full_slug/$PDF_BASENAME' target='_blank'>[PDF]</a>"
        fi
    fi

    if [ "$is_child" = true ]; then
        echo "  <li style='margin-left:20px;'><a href='./$full_slug/index.html'>$title</a>$PDF_LINK_TOC</li>" >> "$INDEX_FILE"
    else
        echo "<li><a href='$full_slug/index.html'>$title</a>$PDF_LINK_TOC</li>" >> "$INDEX_FILE"
    fi
}

# 5. MAIN LOOP
jq -c '.sections[]' "$JSON_FILE" | while read -r section; do
    S_TITLE=$(echo "$section" | jq -r .title)
    echo "<h2>$S_TITLE</h2><ul>" >> "$INDEX_FILE"
    
    echo "$section" | jq -c '.items[]' | while read -r item; do
        I_TITLE=$(echo "$item" | jq -r .title)
        I_SLUG=$(echo "$item" | jq -r .slug)
        process_entry "$I_TITLE" "$I_SLUG" false
        
        echo "$item" | jq -c '.children[]?' | while read -r child; do
            [[ -z "$child" || "$child" == "null" ]] && continue
            C_TITLE=$(echo "$child" | jq -r .title)
            C_SLUG=$(echo "$child" | jq -r .slug)
            process_entry "$C_TITLE" "$I_SLUG/$C_SLUG" true
        done
    done
    echo "</ul>" >> "$INDEX_FILE"
done

echo "</body></html>" >> "$INDEX_FILE"
