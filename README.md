# Django OTMapOnto OAEI 2021
This repository contains the sources and results of the ontology matching 
tool, OTMaponto, participating in OAEI 2021. The tool is wrapped as a Web service implemented in Django+gunicorn+nginx

## Installation Instructions
* OS: Ubuntu LTS 20.04
* user account: ubuntu
* Installation path: /home/ubuntu/django

**step 1**: Install anaconda3 under /home/ubuntu.

**step 2**: Clone this repository under /home/ubuntu.

**step 3**: Rename the cloned folder name from "otmaponto_django" to "django".

**step 4**: Change directory to /home/ubuntu/django by "cd ~/django".

**step 5**: Run "conda activate".

**step 6**: Install the required Python packages in requirements.txt by "pip install -r requirements.txt".

**step 7**: Download the fasttext pre-trained model: crawl-300d-2M-subword.zip, 2 million word vectors trained with subword information on Common Crawl (600B tokens) from 
https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip under /home/ubuntu/django/otmaponto/match/ontology_mapping/model/ .

**step 8**: Run 'sudo cp gunicorn.service /etc/systemd/system/' 

**step 9**: Run 'sudo systemctl enable gunicorn.service'

**step 10**: Run 'sudo systemctl start gunicorn.service' Since the matching application needs to load the fasttext pre-trained model, the gunicorn service may take about 10-15 mins to be ready.

**step 11**: Install nginx by 'sudo install nginx'.

**step 12**: Run 'sudo cp etc.nginx.sites-available.otmaponto /etc/nginx/sites-available/otmaponto'

**step 13**: Delete the default nginx app by 'sudo rm /etc/nginx/sites-enabled/default'

**step 14**: Run 'sudo ln -s /etc/nginx/sites-available/otmaponto /etc/nginx/sites-enabled/'

**step 15**: Create a 'logs' subfolder under ~/django by 'mkdir logs'

**step 16**: Start nginx server by 'sudo systemctl start nginx'

**step 17**: Access the matching from a browser: http://<your ip>/match
  
**step 18**: Web service endpoint: http://<your ip>/match/runmatcher_web_file .


