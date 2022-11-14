# virtyal_try-on



support windows and linux

install virtual camera

windows:[obs](https://obsproject.com/) 

linux:[v4l2loopback](https://github.com/umlaeute/v4l2loopback)

```
apt-get install v4l2loopback-utils
```
### download model
```
googledownload.sh 
```
### Usage

```sh
#only linux
#asigin camera number
sudo modprobe v4l2loopback video_nr=3 
#common
python inference.py


#generate custom cloth
python gen_edge.py
```
