# docker build -t nwjbrandon/spark:2.2.1 . -f Dockerfile
# docker run --name ws --rm -v ./:/data -it nwjbrandon/spark:2.2.1 bash
docker run -v C:\Users\khilr\Desktop\CS5344-Project:/data --name spark-container -it cs5344project /bin/bash