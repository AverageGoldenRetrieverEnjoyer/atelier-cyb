#!/bin/bash
echo "Ce telechargement va prendre un peu de temps, car le fichier est assez gros."
echo "Pense a verifier que tu as assez de place sur ton disque."
echo "La commande pour la taille de ton disque est : df -h"
echo "Ton disque doit avoir au moins 10GB de place libre."
read -p "Appuie sur entrer pour commencer le telechargement, sinon appuie sur Ctrl+C pour quitter."

//Check if the file exists
if [ ! -f "data/monday_morning.pcap" ]; then
    echo "Le fichier existe deja donc n'a pas ete telecharge."
    exit 1
fi

echo ">>> Downloading Benin dataset"
curl -L -o data/monday_morning.pcap --retry 5 --continue-at - -H "Accept-Encoding: gzip, deflate" "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/PCAPs/Monday-WorkingHours.pcap"

SUM=$(sha256sum data/monday_morning.pcap | awk '{print $1}')
if [ "$SUM" != "8525733283c2c5f98891a0dca036e7c7" ]; then
    echo "Le checksum du fichier n'est pas correct."
    exit 1
fi