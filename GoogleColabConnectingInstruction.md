# Please run the code below and copy the ip and paste it into jupiter

# Install jupyterlab and ngrok
!pip install jupyterlab pyngrok -q

# Run jupyterlab in background
!nohup jupyter lab --ip=0.0.0.0 &

# Make jupyterlab accessible via ngrok
from pyngrok import ngrok
print(ngrok.connect(8888))