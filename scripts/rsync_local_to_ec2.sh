rsync -avc -e "ssh -i ~/.ssh/nkalafut-work.pem" --exclude ".*/" .. "$1":~/repos/celltrip/.
