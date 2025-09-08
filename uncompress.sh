if ! command -v unrar &> /dev/null; then
  apt update
  apt install unrar -y
fi

unrar x data/data.rar data
