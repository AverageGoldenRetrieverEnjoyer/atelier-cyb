#!/usr/bin/env bash
# Merge split compressed data files into single files
# - Merges all .lz4 PCAP files into one merged.pcap (memory-efficient streaming)
# - Merges all .gz eve.json files into one merged_eve.json (streaming)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default paths (can be overridden with env vars)
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/logs}"
PCAP_DIR="${PCAP_DIR:-$DATA_ROOT/pcap}"
EVE_DIR="${EVE_DIR:-$DATA_ROOT/eve}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data}"

# Use system temp directory (often faster, may be tmpfs)
TEMP_DIR="${TMPDIR:-./dontlookatme}/merge_data_$$"
mkdir -p "$TEMP_DIR"
trap "rm -rf '$TEMP_DIR'" EXIT INT TERM

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Merge split compressed data files:
  - Merges all .lz4 PCAP files into merged.pcap (streaming, low memory)
  - Merges all .gz eve.json files into merged_eve.json (streaming)

Options:
  -h, --help          Show this help message
  -p, --pcap-dir DIR  PCAP source directory (default: data/logs/pcap)
  -e, --eve-dir DIR   EVE source directory (default: data/logs/eve)
  -o, --output DIR    Output directory (default: data/)
  --pcap-only         Only merge PCAP files
  --eve-only          Only merge EVE files
  --temp-dir DIR      Custom temporary directory (default: system temp)

Environment variables:
  DATA_ROOT           Root data directory
  PCAP_DIR            PCAP source directory
  EVE_DIR             EVE source directory
  OUTPUT_DIR          Output directory
  TMPDIR              Temporary directory

Examples:
  $0                                    # Merge both PCAP and EVE
  $0 --pcap-only                        # Only merge PCAP files
  $0 -o /tmp/merged                     # Custom output directory
EOF
}

# Check for required tools
check_dependencies() {
    local missing=()
    
    if ! command -v lz4 &> /dev/null; then
        missing+=("lz4")
    fi
    
    if ! command -v mergecap &> /dev/null; then
        missing+=("mergecap (from wireshark-common)")
    fi
    
    if ! command -v gunzip &> /dev/null && ! command -v zcat &> /dev/null; then
        missing+=("gunzip or zcat")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}Error: Missing required tools:${NC} ${missing[*]}" >&2
        echo "Install with:" >&2
        echo "  sudo apt-get install lz4 wireshark-common gzip" >&2
        exit 1
    fi
}

# Test if lz4 file is valid (quick check)
test_lz4_file() {
    local file="$1"
    # Try to get file info without full decompression
    lz4 -t "$file" &>/dev/null || return 1
    return 0
}

merge_pcaps() {
    local pcap_dir="$1"
    local output_file="$2"
    
    echo -e "${YELLOW}=== Merging PCAP files (streaming, low memory) ===${NC}"
    
    if [ ! -d "$pcap_dir" ]; then
        echo -e "${RED}Error: PCAP directory not found: $pcap_dir${NC}" >&2
        return 1
    fi
    
    # Find all .lz4 files and sort them
    local lz4_files=($(find "$pcap_dir" -type f -name "*.lz4" | sort))
    
    if [ ${#lz4_files[@]} -eq 0 ]; then
        echo -e "${YELLOW}No .lz4 files found in $pcap_dir${NC}"
        return 0
    fi
    
    echo "Found ${#lz4_files[@]} .lz4 files"
    echo "Using temporary directory: $TEMP_DIR"
    
    # Start with empty output or first file
    local current_merged="$TEMP_DIR/merged_current.pcap"
    local next_merged="$TEMP_DIR/merged_next.pcap"
    local temp_pcap="$TEMP_DIR/temp.pcap"
    local first_file=true
    local success_count=0
    local fail_count=0
    
    for lz4_file in "${lz4_files[@]}"; do
        local basename=$(basename "$lz4_file" .lz4)
        echo -n "  Processing: $(basename "$lz4_file") ... "
        
        # Test file integrity first (fast check)
        if ! test_lz4_file "$lz4_file"; then
            echo -e "${RED}FAILED (corrupted)${NC}"
            ((fail_count++))
            continue
        fi
        
        # Decompress to temp file (one at a time)
        # lz4 -dc outputs to stdout, redirect stderr to check for real errors
        lz4 -dc "$lz4_file" > "$temp_pcap" 2>"$TEMP_DIR/lz4_error.log"
        lz4_exit=$?
        
        # Check if decompression actually worked (file exists and has content)
        if [ $lz4_exit -ne 0 ] || [ ! -s "$temp_pcap" ]; then
            echo -e "${RED}FAILED (decompression error)${NC}"
            rm -f "$temp_pcap"
            ((fail_count++))
            continue
        fi
        
        # Quick sanity check: PCAP files should start with magic numbers
        # Check for common PCAP magic: 0xa1b2c3d4 (big-endian) or 0xd4c3b2a1 (little-endian) or pcapng 0x0a0d0d0a
        if command -v od &> /dev/null; then
            local magic=$(head -c 4 "$temp_pcap" 2>/dev/null | od -An -tx4 2>/dev/null | tr -d ' ' || echo "")
            if [ -n "$magic" ] && [ "$magic" != "a1b2c3d4" ] && [ "$magic" != "d4c3b2a1" ] && [ "$magic" != "0a0d0d0a" ]; then
                # Not a standard PCAP magic, but might still be valid - continue anyway
                echo -e "${YELLOW}WARNING (unusual magic: $magic, continuing anyway)${NC}"
            fi
        fi

        # Validate using capinfos or tshark
        if command -v capinfos &> /dev/null; then
            if ! capinfos "$temp_pcap" >/dev/null 2>&1; then
                echo -e "${RED}ERROR: PCAP appears corrupted (capinfos failed)${NC}"
                ((fail_count++))
                continue
            fi
        elif command -v tshark &> /dev/null; then
            if ! tshark -r "$temp_pcap" -q >/dev/null 2>&1; then
                echo -e "${RED}ERROR: PCAP appears corrupted (tshark failed)${NC}"
                ((fail_count++))
                continue
            fi
        else
            echo -e "${YELLOW}WARNING: No deep PCAP validator (capinfos/tshark) found, magic check only${NC}"
        fi
        
        # Merge with current result
        if [ "$first_file" = true ]; then
            # First file: just copy it
            mv "$temp_pcap" "$current_merged"
            first_file=false
        else
            # Merge with existing
            # mergecap can be verbose, capture errors separately
            # Try merge first, if that fails try concatenation
            mergecap -w "$next_merged" "$current_merged" "$temp_pcap" --log-level=error --log-file="$TEMP_DIR/mergecap_error.log"
            mergecap_exit=$?
            echo -e "${YELLOW}mergecap exit code: $mergecap_exit${NC}"
            if [ -s "$TEMP_DIR/mergecap_error.log" ]; then
                echo -e "${YELLOW}mergecap log:${NC}"
                echo "$(cat "$TEMP_DIR/mergecap_error.log")"
                cp "$temp_pcap" "$TEMP_DIR/corrupted_pcap.pcap"
                cp "$TEMP_DIR/mergecap_error.log" "./corrupted_mergecap_error.log"
                exit 1
            fi

            # Check if merge succeeded - file should exist and be larger than the smaller input
            local current_size=$(stat -c%s "$current_merged" 2>/dev/null || echo 0)
            local temp_size=$(stat -c%s "$temp_pcap" 2>/dev/null || echo 0)
            
            if [ $mergecap_exit -eq 0 ] && [ -f "$next_merged" ] && [ -s "$next_merged" ]; then
                local merged_size=$(stat -c%s "$next_merged" 2>/dev/null || echo 0)
                # Merged file should be at least as large as the larger input (with some tolerance)
                local min_expected=$((current_size > temp_size ? temp_size : current_size))
                if [ "$merged_size" -ge "$min_expected" ]; then
                    mv "$next_merged" "$current_merged"
                    rm -f "$temp_pcap"
                else
                    # Try concatenation as fallback
                    echo -e "${YELLOW}Merge produced small file, trying concatenation...${NC}"
                    if mergecap -w "$next_merged" -a "$current_merged" "$temp_pcap" 2>/dev/null && [ -f "$next_merged" ] && [ -s "$next_merged" ]; then
                        mv "$next_merged" "$current_merged"
                        rm -f "$temp_pcap"
                    else
                        echo -e "${RED}FAILED (both merge and concatenation failed)${NC}"
                        rm -f "$temp_pcap" "$next_merged"
                        ((fail_count++))
                        continue
                    fi
                fi
            else
                # Try concatenation as fallback
                echo -e "${YELLOW}Merge failed, trying concatenation...${NC}"
                if mergecap -w "$next_merged" -a "$current_merged" "$temp_pcap" 2>/dev/null && [ -f "$next_merged" ] && [ -s "$next_merged" ]; then
                    mv "$next_merged" "$current_merged"
                    rm -f "$temp_pcap"
                else
                    echo -e "${RED}FAILED (merge error)${NC}"
                    # Show error if available
                    if [ -s "$TEMP_DIR/mergecap_error.log" ]; then
                        local error_msg=$(head -1 "$TEMP_DIR/mergecap_error.log" 2>/dev/null || echo "Unknown error")
                        echo "    Error: $error_msg"
                    fi
                    rm -f "$temp_pcap" "$next_merged"
                    ((fail_count++))
                    continue
                fi
            fi
        fi
        
        echo -e "${GREEN}OK${NC}"
        ((success_count++))
        
        # Clean up temp file immediately
        rm -f "$temp_pcap"
    done
    
    if [ "$first_file" = true ]; then
        echo -e "${RED}No PCAP files successfully processed${NC}" >&2
        return 1
    fi
    
    # Move final result to output
    mv "$current_merged" "$output_file"
    
    local output_size=$(du -h "$output_file" | cut -f1)
    echo -e "${GREEN}✓ Merged PCAP created: $output_file (${output_size})${NC}"
    echo "  Successfully merged: $success_count files"
    [ $fail_count -gt 0 ] && echo -e "  ${YELLOW}Failed/Skipped: $fail_count files${NC}"
}

merge_eve() {
    local eve_dir="$1"
    local output_file="$2"
    
    echo -e "${YELLOW}=== Merging EVE JSON files (streaming) ===${NC}"
    
    if [ ! -d "$eve_dir" ]; then
        echo -e "${RED}Error: EVE directory not found: $eve_dir${NC}" >&2
        return 1
    fi
    
    # Find all .gz files and sort them
    local gz_files=($(find "$eve_dir" -type f -name "*.gz" | sort))
    
    if [ ${#gz_files[@]} -eq 0 ]; then
        echo -e "${YELLOW}No .gz files found in $eve_dir${NC}"
        return 0
    fi
    
    echo "Found ${#gz_files[@]} .gz files"
    echo "Streaming decompression (low memory)..."
    
    # Create empty output file
    > "$output_file"
    
    local success_count=0
    local fail_count=0
    
    # Process files one at a time, streaming directly to output
    for gz_file in "${gz_files[@]}"; do
        echo -n "  Processing: $(basename "$gz_file") ... "
        
        # Stream decompress directly to output (no temp file)
        if command -v zcat &> /dev/null; then
            if zcat "$gz_file" >> "$output_file" 2>/dev/null; then
                echo -e "${GREEN}OK${NC}"
                ((success_count++))
            else
                echo -e "${RED}FAILED${NC}"
                ((fail_count++))
            fi
        else
            if gunzip -c "$gz_file" >> "$output_file" 2>/dev/null; then
                echo -e "${GREEN}OK${NC}"
                ((success_count++))
            else
                echo -e "${RED}FAILED${NC}"
                ((fail_count++))
            fi
        fi
    done
    
    if [ $success_count -eq 0 ]; then
        echo -e "${RED}No EVE files successfully processed${NC}" >&2
        rm -f "$output_file"
        return 1
    fi
    
    local output_size=$(du -h "$output_file" | cut -f1)
    local line_count=$(wc -l < "$output_file" 2>/dev/null || echo "?")
    echo -e "${GREEN}✓ Merged EVE JSON created: $output_file (${output_size}, ${line_count} lines)${NC}"
    echo "  Successfully merged: $success_count files"
    [ $fail_count -gt 0 ] && echo -e "  ${YELLOW}Failed/Skipped: $fail_count files${NC}"
}

main() {
    local pcap_only=false
    local eve_only=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -p|--pcap-dir)
                PCAP_DIR="$2"
                shift 2
                ;;
            -e|--eve-dir)
                EVE_DIR="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --pcap-only)
                pcap_only=true
                shift
                ;;
            --eve-only)
                eve_only=true
                shift
                ;;
            --temp-dir)
                TEMP_DIR="$2/merge_data_$$"
                mkdir -p "$TEMP_DIR"
                shift 2
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}" >&2
                usage
                exit 1
                ;;
        esac
    done
    
    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"
    
    # Check dependencies
    check_dependencies
    
    # Merge PCAP files (streaming, one at a time)
    if [ "$eve_only" = false ]; then
        merge_pcaps "$PCAP_DIR" "$OUTPUT_DIR/merged.pcap" || {
            echo -e "${YELLOW}PCAP merge had some errors (check output above)${NC}" >&2
            # Don't exit - partial success is better than nothing
        }
    fi
    
    # Merge EVE files (streaming, direct to output)
    if [ "$pcap_only" = false ]; then
        merge_eve "$EVE_DIR" "$OUTPUT_DIR/merged_eve.json" || {
            echo -e "${YELLOW}EVE merge had some errors (check output above)${NC}" >&2
            # Don't exit - partial success is better than nothing
        }
    fi
    
    echo -e "${GREEN}=== Merge complete ===${NC}"
    echo "Output directory: $OUTPUT_DIR"
    [ "$eve_only" = false ] && echo "  - merged.pcap"
    [ "$pcap_only" = false ] && echo "  - merged_eve.json"
}

main "$@"
