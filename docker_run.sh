export FLINGBOT_PATH=${PWD}
docker run \
    --name flingbot-container \
	-v $FLINGBOT_PATH:/workspace/flingbot \
	--gpus all \
    --shm-size=64gb  \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it flingbot