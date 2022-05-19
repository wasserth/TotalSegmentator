## Terraform commands
```
cd resources
terraform init
terraform validate
terraform apply -auto-approve
```

## Helpful commands for setting up the cloud server
Updating code on server:
``` 
cd ~/dev/totalsegmentator
git pull
docker build -t totalsegmentator:master .
``` 

# todo: put right starting command here (depends if we use nginx)

Run docker TotalSegmentator for test
``` 
docker run --gpus 'device=0' --ipc=host -v /home/ubuntu/test:/workspace totalsegmentator:master TotalSegmentator -i /workspace/ct3mm_0000.nii.gz -o /workspace/test_output --fast --preview
``` 

Run docker flask server for test locally
``` 
docker run -p 80:5000 --gpus 'device=0' --ipc=host -v /home/jakob/dev/TotalSegmentator/store:/app/store totalsegmentator:master /app/run_server.sh
``` 
Can only be killed via docker
``` 
docker kill $(docker ps -q)
``` 

Run docker on server for production  
(will automatically start after reboot)
``` 
docker run -d --restart always -p 80:5000 --gpus 'device=0' --ipc=host --name totalsegmentator-server-job -v /home/ubuntu/store:/app/store totalsegmentator:master /app/run_server.sh
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

Backup to local harddrive
``` 
rsync -avz <username>@<URL_TODO>:/mnt/data/server-store /mnt/jay_hdd/backup
``` 
