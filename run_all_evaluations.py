import os

os.system("python eval_road_shape.py -d semantickitti -p '/path/to/SemanticKITTI/dataset/' -s 00 --start 100 --stop 200")
os.system("python eval_road_shape.py -d semantickitti -p '/path/to/SemanticKITTI/dataset/' -s 03 --start 0 --stop 100")

os.system("python eval_road_shape.py -d pandaset -p '/path/to/pandaset' -s 001 --start 0 --stop 80")
os.system("python eval_road_shape.py -d pandaset -p '/path/to/pandaset' -s 027 --start 0 --stop 80")

os.system("python eval_road_shape.py -d nuscenes -p '/path/to/nuscenes' --start 0 --stop 38 -s 8")
os.system("python eval_road_shape.py -d nuscenes -p '/path/to/nuscenes' --start 0 --stop 38 -s 22")

# randlanet
os.system("python eval_road_shape.py -d semantickitti -p '/path/to/SemanticKITTI/dataset/' -s 00 --start 100 --stop 200 -m randlanet")
os.system("python eval_road_shape.py -d semantickitti -p '/path/to/SemanticKITTI/dataset/' -s 03 --start 0 --stop 100 -m randlanet")

os.system("python eval_road_shape.py -d pandaset -p '/path/to/pandaset' -s 001 --start 0 --stop 80 -m randlanet")
os.system("python eval_road_shape.py -d pandaset -p '/path/to/pandaset' -s 027 --start 0 --stop 80 -m randlanet")

os.system("python3 eval_road_shape.py -d nuscenes -p '/path/to/nuscenes' -s 8 --start 0 --stop 38 -m randlanet")
os.system("python3 eval_road_shape.py -d nuscenes -p '/path/to/nuscenes' -s 22 --start 0 --stop 38 -m randlanet")