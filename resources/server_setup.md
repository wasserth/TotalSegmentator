## Terraform commands
```
cd resources
terraform init
terraform validate
terraform apply -auto-approve

terraform destroy -auto-approve
```

## Helpful commands for setting up the cloud server
Updating code on server:
```
cd ~/dev/TotalSegmentator
git pull

docker stop totalsegmentator-server-job
docker rm $(sudo docker ps -a -q -f status=exited)

docker build -t totalsegmentator:master .

docker run -d --restart always -p 80:5000 --gpus 'device=0' --ipc=host --name totalsegmentator-server-job -v /home/ubuntu/store:/app/store totalsegmentator:master /app/run_server.sh
```

Run docker TotalSegmentator for test locally
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
Have to setup docker.service once so docker will be available at system start
```
systemctl enable docker.service
```
Then this docker container will always run
```
docker run -d --restart always -p 80:5000 --gpus 'device=0' --ipc=host --name totalsegmentator-server-job -v /home/ubuntu/store:/app/store totalsegmentator:master /app/run_server.sh
```

See running containers
```
docker container ls
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

Remove finished containers
```
docker rm $(docker ps -a -q -f status=exited)
```

Remove all untagged images
```
docker rmi $(docker images | grep "<none>" | awk '{print $3}')
```

Restart docker (e.g. if crashed)
```
docker restart totalsegmentator-server-job
```


## Other commands

Backup to local harddrive
```
rsync -avz <username>@<URL_TODO>:/mnt/data/server-store /mnt/jay_hdd/backup
```

Systemd commands

Start or stop only once
```
systemctl start/stop/restart totalsegmentator_server
```
Permanently start program (automatic restart on error or reboot):
```
systemctl enable/disable totalsegmentator_server
```
Check status
```
systemctl status totalsegmentator_server
```


## Old commands
