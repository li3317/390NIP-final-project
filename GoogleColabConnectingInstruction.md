# Please run the code below
# then copy the resulted ip and paste it into jupiter notebook in vscode

# Install jupyterlab and ngrok
!pip install jupyterlab pyngrok -q

# Run jupyterlab in background
!nohup jupyter lab --ip=0.0.0.0 &

# Make jupyterlab accessible via ngrok
from pyngrok import ngrok
print(ngrok.connect(8888))