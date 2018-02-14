#!/bin/bash

# from https://hub.docker.com/r/lballabio/quantlib-notebook/
#
docker run -ti -p 8888:8888 -v `pwd`:/notebooks lballabio/quantlib-notebook

