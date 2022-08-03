sudo groupadd docker || true

sudo usermod -aG docker $USER || true

echo "Current user added to docker supergroup."

newgrp docker || true
