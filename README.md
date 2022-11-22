# virtyal_try-on
This system can be installed on Windows and Linux.

#Installation
##Install virtual camera

Windows: [obs](https://obsproject.com/) 

Linux: [v4l2loopback](https://github.com/umlaeute/v4l2loopback)
```
apt-get install v4l2loopback-utils
```
### Download model
```
python download_model.sh 
```
### Usage
```sh
#only linux
#asigin camera number
sudo modprobe v4l2loopback video_nr=3 

#common
python inference.py


