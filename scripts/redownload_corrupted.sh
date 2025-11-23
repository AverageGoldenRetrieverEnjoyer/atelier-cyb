#!/usr/bin/env bash
# Re-download corrupted lz4 files from Google Drive

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PCAP_DIR="${PCAP_DIR:-$REPO_ROOT/data/gdrive/tpot-backup/logs/pcap}"
REMOTE="${REMOTE:-gdrive:tpot-backup/logs/pcap}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! command -v rclone &> /dev/null; then
    echo -e "${RED}Error: rclone not found${NC}" >&2
    echo "Install with: sudo apt-get install rclone" >&2
    exit 1
fi

if ! command -v lz4 &> /dev/null; then
    echo -e "${RED}Error: lz4 not found${NC}" >&2
    exit 1
fi

if [ ! -d "$PCAP_DIR" ]; then
    echo -e "${RED}Error: Directory not found: $PCAP_DIR${NC}" >&2
    exit 1
fi

echo "Finding corrupted lz4 files..."
echo ""

# Find corrupted files
corrupted_files=()
lz4_files=($(find "$PCAP_DIR" -type f -name "*.lz4" | sort))

for file in "${lz4_files[@]}"; do
    if ! lz4 -t "$file" &>/dev/null; then
        corrupted_files+=("$(basename "$file")")
    fi
done

if [ ${#corrupted_files[@]} -eq 0 ]; then
    echo -e "${GREEN}No corrupted files found!${NC}"
    exit 0
fi

echo -e "${YELLOW}Found ${#corrupted_files[@]} corrupted files:${NC}"
for f in "${corrupted_files[@]}"; do
    echo "  - $f"
done

echo ""
read -p "Re-download these files from $REMOTE? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

echo ""
echo "Re-downloading corrupted files..."
echo ""

# Create temp directory for downloads
TEMP_DIR=$(mktemp -d)
trap "rm -rf '$TEMP_DIR'" EXIT

success=0
failed=0

for filename in "${corrupted_files[@]}"; do
    echo "Downloading $filename ..."
    echo "  (This may take a while for large files - be patient!)"
    
    # Download to temp location first
    temp_file="$TEMP_DIR/$filename"
    
    # Download with progress (rclone will show progress automatically)
    # We capture exit code separately since progress output goes to stderr
    rclone copy "$REMOTE/$filename" "$TEMP_DIR" --no-check-dest --progress
    rclone_exit=$?
    
    # Check if download succeeded and file exists
    if [ $rclone_exit -eq 0 ] && [ -f "$temp_file" ]; then
        # Verify downloaded file
        echo -n "  Verifying integrity ... "
        if lz4 -t "$temp_file" &>/dev/null 2>&1; then
            # Move to final location
            mv "$temp_file" "$PCAP_DIR/$filename"
            echo -e "${GREEN}✓ OK${NC}"
            success=$((success + 1))
        else
            echo -e "${RED}✗ Still corrupted after download${NC}"
            rm -f "$temp_file"
            failed=$((failed + 1))
        fi
    else
        if [ $rclone_exit -ne 0 ]; then
            echo -e "${RED}✗ Download failed (rclone exit code: $rclone_exit)${NC}"
        else
            echo -e "${RED}✗ File not found after download${NC}"
        fi
        failed=$((failed + 1))
    fi
    echo ""
done

echo ""
echo "=========================================="
echo "Re-download summary:"
echo -e "  ${GREEN}Successfully fixed: $success${NC}"
echo -e "  ${RED}Still corrupted: $failed${NC}"

if [ $failed -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Some files are still corrupted. Possible issues:${NC}"
    echo "  1. Source files on Google Drive are corrupted"
    echo "  2. Network issues during download"
    echo "  3. Disk problems"
fi

# Always exit successfully - we've done our best
exit 0

