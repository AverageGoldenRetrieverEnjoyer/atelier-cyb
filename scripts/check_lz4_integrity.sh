#!/usr/bin/env bash
# Check integrity of lz4 files and report which are corrupted

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PCAP_DIR="${PCAP_DIR:-$REPO_ROOT/data/gdrive/tpot-backup/logs/pcap}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! command -v lz4 &> /dev/null; then
    echo -e "${RED}Error: lz4 not found${NC}" >&2
    exit 1
fi

if [ ! -d "$PCAP_DIR" ]; then
    echo -e "${RED}Error: Directory not found: $PCAP_DIR${NC}" >&2
    exit 1
fi

echo "Checking lz4 file integrity in: $PCAP_DIR"
echo ""

lz4_files=($(find "$PCAP_DIR" -type f -name "*.lz4" | sort))
total=${#lz4_files[@]}

if [ $total -eq 0 ]; then
    echo -e "${YELLOW}No .lz4 files found${NC}"
    exit 0
fi

echo "Found $total .lz4 files"
echo ""

good_count=0
bad_count=0
bad_files=()

for file in "${lz4_files[@]}"; do
    basename=$(basename "$file")
    echo -n "Checking: $basename ... "
    
    # Test integrity (fast check)
    if lz4 -t "$file" &>/dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        good_count=$((good_count + 1))
    else
        echo -e "${RED}CORRUPTED${NC}"
        bad_count=$((bad_count + 1))
        bad_files+=("$file")
        
        # Try to get file size for diagnostics
        if command -v stat &> /dev/null; then
            size=$(stat -c%s "$file" 2>/dev/null || echo "?")
            echo "    Size: $size bytes"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "  Total files: $total"
echo -e "  ${GREEN}Good: $good_count${NC}"
echo -e "  ${RED}Corrupted: $bad_count${NC}"

if [ $bad_count -gt 0 ]; then
    echo ""
    echo -e "${RED}Corrupted files:${NC}"
    for file in "${bad_files[@]}"; do
        echo "  - $(basename "$file")"
    done
    
    echo ""
    echo "Possible causes:"
    echo "  1. Incomplete download from Google Drive"
    echo "  2. Network interruption during transfer"
    echo "  3. Disk errors"
    echo "  4. File system corruption"
    echo ""
    echo "Solutions:"
    echo "  1. Re-download corrupted files:"
    echo "     ./scripts/gdrive_import.sh"
    echo "  2. Check disk health:"
    echo "     sudo smartctl -a /dev/sdX"
    echo "  3. Check file system:"
    echo "     sudo fsck /dev/sdX"
fi

if [ $bad_count -gt 0 ]; then
    echo "Re-downloading corrupted files..."
    ./scripts/redownload_corrupted.sh
fi

exit 0
