#!/usr/bin/env bash
# Connect to WireGuard VPN and then SSH to remote host

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ACCESS_KEYS_DIR="${ACCESS_KEYS_DIR:-$REPO_ROOT/access_keys}"

# SSH connection details
SSH_KEY="${SSH_KEY:-}"
SSH_PORT="${SSH_PORT:-64295}"
SSH_USER="${SSH_USER:-azureuser}"
SSH_HOST="${SSH_HOST:-10.0.0.4}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Connect to WireGuard VPN and SSH to remote host.

Options:
  -h, --help          Show this help message
  -k, --key PATH      Path to SSH .pem key file (auto-detected if not specified)
  -p, --port PORT     SSH port (default: 64295)
  -u, --user USER     SSH user (default: azureuser)
  --host HOST         SSH host (default: 10.0.0.4)
  --wg-config PATH    Path to WireGuard config file (auto-detected if not specified)
  --skip-vpn          Skip VPN connection, just SSH
  --skip-ssh          Skip SSH connection, just VPN

Environment variables:
  ACCESS_KEYS_DIR     Directory containing VPN configs and SSH keys
  SSH_KEY             Path to SSH .pem key
  SSH_PORT            SSH port
  SSH_USER            SSH user
  SSH_HOST            SSH host

Examples:
  $0                                    # Auto-detect everything
  $0 -k access_keys/key.pem             # Specify SSH key
  $0 --skip-vpn                        # Skip VPN, just SSH
EOF
}

# Check if running as root (needed for WireGuard)
check_root() {
    if [ "$EUID" -ne 0 ] && [ "$1" != "skip" ]; then
        echo -e "${YELLOW}Note: WireGuard operations may require sudo${NC}"
        return 1
    fi
    return 0
}

# Check if WireGuard is installed
check_wireguard() {
    if command -v wg &> /dev/null && command -v wg-quick &> /dev/null; then
        return 0
    fi
    return 1
}

# Install WireGuard
install_wireguard() {
    echo -e "${YELLOW}WireGuard not found. Installing...${NC}"
    
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y wireguard wireguard-tools
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS
        sudo yum install -y epel-release
        sudo yum install -y wireguard-tools
    elif command -v dnf &> /dev/null; then
        # Fedora
        sudo dnf install -y wireguard-tools
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        sudo pacman -S --noconfirm wireguard-tools
    else
        echo -e "${RED}Error: Cannot determine package manager. Please install WireGuard manually.${NC}" >&2
        echo "Visit: https://www.wireguard.com/install/" >&2
        exit 1
    fi
    
    if check_wireguard; then
        echo -e "${GREEN}✓ WireGuard installed successfully${NC}"
    else
        echo -e "${RED}Error: WireGuard installation failed${NC}" >&2
        exit 1
    fi
}

# Find WireGuard config file
find_wg_config() {
    if [ -n "${WG_CONFIG:-}" ] && [ -f "$WG_CONFIG" ]; then
        echo "$WG_CONFIG"
        return 0
    fi
    
    # Look for .conf files in access_keys directory
    if [ -d "$ACCESS_KEYS_DIR" ]; then
        local configs=($(find "$ACCESS_KEYS_DIR" -type f -name "*.conf" 2>/dev/null))
        if [ ${#configs[@]} -gt 0 ]; then
            echo "${configs[0]}"
            return 0
        fi
    fi
    
    return 1
}

# Connect to WireGuard VPN
connect_vpn() {
    local wg_config="$1"
    
    if [ ! -f "$wg_config" ]; then
        echo -e "${RED}Error: WireGuard config file not found: $wg_config${NC}" >&2
        return 1
    fi
    
    echo -e "${YELLOW}Connecting to WireGuard VPN...${NC}"
    echo "  Config: $wg_config"
    
    # Check if already connected (interface name is usually wg0 or from config)
    local ifname=$(grep -E "^\[Interface\]" -A 5 "$wg_config" | grep -i "Name" | cut -d'=' -f2 | tr -d ' ' || echo "wg0")
    
    if ip link show "$ifname" &>/dev/null; then
        echo -e "${YELLOW}VPN interface $ifname already exists. Bringing it up...${NC}"
        sudo wg-quick up "$wg_config" || {
            echo -e "${YELLOW}Interface exists but may be down. Trying to bring it up...${NC}"
            sudo wg-quick down "$wg_config" 2>/dev/null || true
            sudo wg-quick up "$wg_config"
        }
    else
        sudo wg-quick up "$wg_config"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ VPN connected${NC}"
        # Show connection status
        sudo wg show
        return 0
    else
        echo -e "${RED}Error: Failed to connect to VPN${NC}" >&2
        return 1
    fi
}

# Find SSH key file
find_ssh_key() {
    if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
        echo "$SSH_KEY"
        return 0
    fi
    
    # Look for .pem files in access_keys directory
    if [ -d "$ACCESS_KEYS_DIR" ]; then
        local keys=($(find "$ACCESS_KEYS_DIR" -type f -name "*.pem" 2>/dev/null))
        if [ ${#keys[@]} -gt 0 ]; then
            echo "${keys[0]}"
            return 0
        fi
    fi
    
    return 1
}

# Connect via SSH
connect_ssh() {
    local ssh_key="$1"
    
    if [ ! -f "$ssh_key" ]; then
        echo -e "${RED}Error: SSH key file not found: $ssh_key${NC}" >&2
        return 1
    fi
    
    # Set proper permissions on key
    chmod 600 "$ssh_key" 2>/dev/null || true
    
    echo -e "${YELLOW}Connecting via SSH...${NC}"
    echo "  Host: $SSH_USER@$SSH_HOST"
    echo "  Port: $SSH_PORT"
    echo "  Key: $ssh_key"
    echo ""
    
    ssh -i "$ssh_key" -p "$SSH_PORT" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$SSH_USER@$SSH_HOST"
}

# Disconnect VPN on exit
cleanup() {
    if [ -n "${WG_CONFIG_FILE:-}" ] && [ -f "$WG_CONFIG_FILE" ]; then
        echo ""
        echo -e "${YELLOW}Disconnecting VPN...${NC}"
        sudo wg-quick down "$WG_CONFIG_FILE" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

main() {
    local skip_vpn=false
    local skip_ssh=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -k|--key)
                SSH_KEY="$2"
                shift 2
                ;;
            -p|--port)
                SSH_PORT="$2"
                shift 2
                ;;
            -u|--user)
                SSH_USER="$2"
                shift 2
                ;;
            --host)
                SSH_HOST="$2"
                shift 2
                ;;
            --wg-config)
                WG_CONFIG="$2"
                shift 2
                ;;
            --skip-vpn)
                skip_vpn=true
                shift
                ;;
            --skip-ssh)
                skip_ssh=true
                shift
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}" >&2
                usage
                exit 1
                ;;
        esac
    done
    
    # Check/install WireGuard
    if [ "$skip_vpn" = false ]; then
        if ! check_wireguard; then
            install_wireguard
        else
            echo -e "${GREEN}✓ WireGuard is installed${NC}"
        fi
        
        # Find and connect VPN
        local wg_config=$(find_wg_config)
        if [ -z "$wg_config" ]; then
            echo -e "${RED}Error: No WireGuard config file found in $ACCESS_KEYS_DIR${NC}" >&2
            echo "Looking for .conf files..." >&2
            exit 1
        fi
        
        WG_CONFIG_FILE="$wg_config"
        connect_vpn "$wg_config" || {
            echo -e "${RED}Failed to connect to VPN${NC}" >&2
            exit 1
        }
        
        # Wait a moment for VPN to stabilize
        echo "Waiting for VPN to stabilize..."
        sleep 2
    fi
    
    # SSH connection
    if [ "$skip_ssh" = false ]; then
        local ssh_key=$(find_ssh_key)
        if [ -z "$ssh_key" ]; then
            echo -e "${RED}Error: No SSH key file found in $ACCESS_KEYS_DIR${NC}" >&2
            echo "Looking for .pem files..." >&2
            exit 1
        fi
        
        connect_ssh "$ssh_key"
    fi
}

main "$@"

