docker build -t nwjbrandon/spark:3.0.1 . -f Dockerfile
docker run --name ws1 --rm -v ./:/data -it nwjbrandon/spark:3.0.1 bash
