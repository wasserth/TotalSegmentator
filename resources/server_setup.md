## Helpful commands for setting up the google cloud server


Updating code on server:
``` 
cd ~/dev/totalsegmentator
git pull
docker build -t totalsegmentator:master .
``` 

# todo: put right starting command here (depends if we use nginx)
Run docker on server for test
``` 
docker run -p 5000:5000 --gpus 'device=0' --ipc=host -v /home/jakob/dev/TotalSegmentator/store:/app/store totalsegmentator:master /app/run_server.sh
``` 
Can only be killed via docker
``` 
docker kill $(docker ps -q)
``` 

Run docker on server for production
``` 
docker run -d --restart always -p 5000:80 --gpus 'device=0' --ipc=host --name totalsegmentator-server-job -v /mnt/data/server-store:/app/store totalsegmentator:master /app/run_server.sh
``` 

Stop docker
```
docker stop totalsegmentator-server-job
docker rm $(sudo docker ps -a -q -f status=exited)  
```

See stdout of running docker container
```
docker logs totalsegmentator-server-job
```


## Other commands

Upload docker to server
``` 
docker save totalsegmentator:master | ssh -C <username>@<URL_TODO> docker load
``` 

Backup to local harddrive
``` 
rsync -avz <username>@<URL_TODO>:/mnt/data/server-store /mnt/jay_hdd/backup
``` 
