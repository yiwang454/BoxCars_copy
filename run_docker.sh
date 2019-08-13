#!/bin/bash
export XSOCK=/tmp/.X11-unix
export XAUTH=/tmp/.docker.xauth
sudo xauth nlist :0 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run -it --rm \
--name boxcar \
-v $XSOCK:$XSOCK \
-v $XAUTH:$XAUTH \
-e XAUTHORITY=$XAUTH \
-e DISPLAY=$DISPLAY \
-v /home/vivacityserver6/:/home/vivacityserver6/ \
--runtime=nvidia \
--entrypoint=/bin/bash \
--init \
tf-keras-cv2:latest
