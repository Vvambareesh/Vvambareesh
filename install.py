import subprocess

# Install python dependencies
subprocess.run(['pip', 'install','numpy', 'scipy', 'sounddevice', 
                'python-speech-features']) 

# Install PocketSphinx
subprocess.run(['pip', 'install', 'pocketsphinx'])

# Download pre-trained models
import urllib.request
model_url = "https://github.com/cmusphinx/pocketsphinx-data/blob/master/en-us/en-us.tar.gz"
urllib.request.urlretrieve(model_url, 'en-us.tar.gz')  
subprocess.run(['tar', '-xzvf', 'en-us.tar.gz'])

print('PocketSphinx installed!')