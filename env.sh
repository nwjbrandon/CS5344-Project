# docker build -t aiqqia/spark:2.2.1 . -f Dockerfile
docker run --name ws --rm -v ./:/data -it nwjbrandon/spark:2.2.1 bash
