
# docker build -f Dockerfile-python-3.6 -t mypy3.6 .
if [ 1 -eq 1 ]; then
docker run -it \
  -v $HOME:$HOME \
  mypy3.6
else
docker run -it \
  -v /:/tmp/abc \
  mypy3.6
fi
