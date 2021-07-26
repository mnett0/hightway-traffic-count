# hightway-traffic-count

The aim of the project is to count the number of cars passing in both directions on the Millau viaduct in France, from a
live video broadcast
here: [Viewsurf](https://www.viewsurf.com/univers/trafic/vue/17348-france-midi-pyrenees-millau-a75-viaduc-de-millau-sud)

Then to display the collected data on a dashboard.

The master branch uses the background subtraction method with OpenCV library. While the yolo branch, uses the YOLOv3 (or
v4) library it allows to differentiate the type of vehicles but is very heavy to execute. Also, it allows to count the
vehicles in both directions.

**The architecture of the project:**

Live video <-- Python script (OpenCV) --> Prometheus <-- Grafana



