## Steps to run
---

```shell
cd pixel_mapping_test
catkin_make
source devel/setup.bash
roslaunch pixel_mapping image_parser.launch
```
To run the scripts that publishes points

```shell
roslaunch image_parser image_parser.launch
```
To view the points from the node go to RVIZ:

* Click on ADD
* Select dropdown "By Topic"
* Click on "Marker"