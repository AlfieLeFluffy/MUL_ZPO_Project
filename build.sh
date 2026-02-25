#!/bin/bash

clean() {
    docker rmi mulzpo:latest -f
}

build() {
    docker build -t mulzpo:latest .
}

run() {
    build
    docker run -u=$(id -u $USER):$(id -g $USER) -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw --rm mulzpo:latest
}

case "$1" in
    clean)
        clean
        ;;
    build)
        build
        ;;
    run)
        run
        ;;
    *)  
        run
        ;;
esac

