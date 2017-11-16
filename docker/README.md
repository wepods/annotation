To build the docker:
```
docker build -t annotaion .
```

Run this command only once when you started your computer.
```
xhost +local:docker
```

For making your container:

```
nvidia-docker run -it  --name annotation -v /media/your_data/:/root/data   --workdir="/root"    --env="DISPLAY"    --env="XDG_RUNTIME_DIR"    --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" annotation bash
```

For reuse it:
```
nvidia-docker start annotation
nvidia-docker exec -it annotation bash
```

For more information on how to work with a docker, you can check my short description
[here](https://medium.com/@fkariminejadasl/getting-started-docker-b9ef6d90f979).
