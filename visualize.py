import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

sys.path.append("MTR/mtr/datasets/waymo/")

from waymo_types import polyline_type

# fichiers importants
# https://github.com/tudoroancea/civil_459_project/blob/main/MTR/mtr/datasets/waymo/waymo_dataset.py#L335
# https://github.com/tudoroancea/civil_459_project/blob/main/MTR/mtr/datasets/waymo/data_preprocess.py
# https://github.com/tudoroancea/civil_459_project/blob/main/MTR/mtr/datasets/waymo/waymo_types.py

# file = "../data/processed_scenarios_training/sample_2c25255b942748dc.pkl"
file = "../data/processed_scenarios_training/sample_867dd000677d389.pkl"

with open(file, 'rb') as f:
    data = pickle.load(f)
    print(data.keys())
    print(data["map_infos"].keys())
    print(data["map_infos"]["all_polylines"].shape)
    print(data["map_infos"]["road_line"][0])
    # polyline is an array of N array of size 7
    # polyline : [x, y, z, dir_x, dir_y, dir_z, global_type]

    fig, ax = plt.subplots()

    for a in data["map_infos"]["road_line"]:
        color = 'black'
        linestyle = 'solid'
        if a["type"] == "TYPE_BROKEN_SINGLE_WHITE":
            color = 'lightgray'
            linestyle = 'dashed'
        elif a["type"] == "TYPE_SOLID_SINGLE_WHITE" or a["type"] == "TYPE_SOLID_DOUBLE_WHITE":
            color = 'lightgray'
        elif a["type"] == "TYPE_SOLID_SINGLE_YELLOW" or a["type"] == "TYPE_SOLID_DOUBLE_YELLOW":
            color = 'yellow'
        elif a["type"] == "TYPE_BROKEN_SINGLE_YELLOW" or a["type"] == "TYPE_BROKEN_DOUBLE_YELLOW" or a["type"] == "TYPE_PASSING_DOUBLE_YELLOW":
            color = 'yellow'
            linestyle = 'dashed'

        indices = a["polyline_index"]
        road_line_polylines = data["map_infos"]["all_polylines"][indices[0]:indices[1]]
        ax.plot(road_line_polylines[:, 0], road_line_polylines[:, 1], color=color, linestyle=linestyle)

    for a in data["map_infos"]["road_edge"]:
        color = 'black'
        if a["type"] == "TYPE_ROAD_EDGE_MEDIAN":
            color = 'gray'
        indices = a["polyline_index"]
        road_edge_polylines = data["map_infos"]["all_polylines"][indices[0]:indices[1]]
        ax.plot(road_edge_polylines[:, 0], road_edge_polylines[:, 1], color=color)


    # load the road lane
    for a in data["map_infos"]["lane"]:
        indices = a["polyline_index"]
        road_line_polylines = data["map_infos"]["all_polylines"][indices[0]:indices[1]]
        ax.plot(road_line_polylines[:, 0], road_line_polylines[:, 1], color='gray')

    print(data["map_infos"]["lane"])
    print(road_line_polylines[0])
    # plot the polylines
    
    # ax.plot(road_line_polylines[:, 0], road_line_polylines[:, 1])


    # load the trajectories
    # trajs : x, y, z, length, width, height, heading, velocity_x, velocity_y, valid
    type = data["track_infos"]["object_type"]
    trajs = data["track_infos"]["trajs"]
    print(trajs.shape)
    for i in range(len(type)):
        print(type[i])
        angle = trajs[i, 0, 6]
        if type[i] == "TYPE_VEHICLE":
            # plot a rectangle at the first position of the trajectory and rotate it
            rect = Rectangle((trajs[i, 0, 0] - 1, trajs[i, 0, 1]-2), 2, 4, angle=90 + angle*180/np.pi, color='blue', rotation_point='center')
            ax.add_patch(rect)
        elif type[i] == "TYPE_PEDESTRIAN":
            rect = Rectangle((trajs[i, 0, 0] - 0.25, trajs[i, 0, 1]-0.25), 1, 1, angle=90 + angle*180/np.pi, color='black', rotation_point='center')
            ax.add_patch(rect)
        elif type[i] == "TYPE_CYCLIST":
            rect = Rectangle((trajs[i, 0, 0] - 0.25, trajs[i, 0, 1]-0.5), 1, 1, angle=90 + angle*180/np.pi, color='green', rotation_point='center')
            ax.add_patch(rect)
        ax.scatter(trajs[i, :, 0], trajs[i, :, 1], color='red', s=0.5)

    ax.set_aspect('equal')
    ax.set_box_aspect(1)    
    plt.show()
