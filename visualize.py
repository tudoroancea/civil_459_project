import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import sys

sys.path.append("MTR/mtr/datasets/waymo/")

from waymo_types import polyline_type

from apply_model import apply

# launch command : python visualize.py --cfg_file MTR/tools/cfgs/waymo/mtr_single_file_vis.yaml

# fichiers importants
# https://github.com/tudoroancea/civil_459_project/blob/main/MTR/mtr/datasets/waymo/waymo_dataset.py#L335
# https://github.com/tudoroancea/civil_459_project/blob/main/MTR/mtr/datasets/waymo/data_preprocess.py
# https://github.com/tudoroancea/civil_459_project/blob/main/MTR/mtr/datasets/waymo/waymo_types.py

# file = "../data/processed_scenarios_training/sample_2c25255b942748dc.pkl"
path = "../data/processed_scenarios_training/"
filename = "sample_867dd000677d389.pkl"
file = path + filename
# file = "../data/processed_scenarios_training/sample_867dd000677d389.pkl"


def plot_frame(ax, data, frame):
    # load the trajectories
    # trajs : x, y, z, length, width, height, heading, velocity_x, velocity_y, valid
    type = data["track_infos"]["object_type"]
    trajs = data["track_infos"]["trajs"]


    for i in range(len(type)):
        # remove 0,0 points from the trajectories
        zeros = np.where(trajs[i, :, 0] == 0)[0]
        first_zero = trajs.shape[1]
        if len(zeros) > 0:
            first_zero = np.where(trajs[i, :, 0] == 0)[0][0]        
        # print(first_zero)
        index = min(frame, first_zero-1)
        if (index >= 0):
            angle = trajs[i, index, 6]
            if type[i] == "TYPE_VEHICLE":
                # plot a rectangle at the first position of the trajectory and rotate it
                rect = Rectangle((trajs[i, index, 0] - 1, trajs[i, index, 1]-2), 2, 4, angle=90 + angle*180/np.pi, color='blue', rotation_point='center')
                ax.add_patch(rect)
            elif type[i] == "TYPE_PEDESTRIAN":
                rect = Rectangle((trajs[i, index, 0] - 0.25, trajs[i, index, 1]-0.25), 1, 1, angle=90 + angle*180/np.pi, color='black', rotation_point='center')
                ax.add_patch(rect)
            elif type[i] == "TYPE_CYCLIST":
                rect = Rectangle((trajs[i, index, 0] - 0.25, trajs[i, index, 1]-0.5), 1, 1, angle=90 + angle*180/np.pi, color='green', rotation_point='center')
                ax.add_patch(rect)

            ax.scatter(trajs[i, :first_zero, 0], trajs[i, :first_zero, 1], color='red', s=0.1, alpha=0.5)

    for a in data["map_infos"]["road_line"]:
        color = 'black'
        linestyle = 'solid'
        if a["type"] == "TYPE_BROKEN_SINGLE_WHITE":
            color = 'gray'
            linestyle = 'dashed'
        elif a["type"] == "TYPE_SOLID_SINGLE_WHITE" or a["type"] == "TYPE_SOLID_DOUBLE_WHITE":
            color = 'gray'
        elif a["type"] == "TYPE_SOLID_SINGLE_YELLOW" or a["type"] == "TYPE_SOLID_DOUBLE_YELLOW":
            color = 'yellow'
        elif a["type"] == "TYPE_BROKEN_SINGLE_YELLOW" or a["type"] == "TYPE_BROKEN_DOUBLE_YELLOW" or a["type"] == "TYPE_PASSING_DOUBLE_YELLOW":
            color = 'yellow'
            linestyle = 'dashed'

        indices = a["polyline_index"]
        road_line_polylines = data["map_infos"]["all_polylines"][indices[0]:indices[1]]
        ax.plot(road_line_polylines[:, 0], road_line_polylines[:, 1], color=color, linestyle=linestyle, linewidth=0.2)

    for a in data["map_infos"]["road_edge"]:
        color = 'black'
        if a["type"] == "TYPE_ROAD_EDGE_MEDIAN":
            color = 'gray'
        indices = a["polyline_index"]
        road_edge_polylines = data["map_infos"]["all_polylines"][indices[0]:indices[1]]
        ax.plot(road_edge_polylines[:, 0], road_edge_polylines[:, 1], color=color, linewidth=0.2)


    # load the road lane
    for a in data["map_infos"]["lane"]:
        indices = a["polyline_index"]
        road_line_polylines = data["map_infos"]["all_polylines"][indices[0]:indices[1]]
        ax.plot(road_line_polylines[:, 0], road_line_polylines[:, 1], color='lightgray', linewidth=0.2)


def get_prediction():
    return apply(filename)

def add_prediction_to_plot(ax, pred):
    for i in pred:
        trajs = i["pred_trajs"]
        num_trajs = trajs.shape[0]
        for j in range(num_trajs):
            plt.scatter(trajs[j, :, 0], trajs[j, :, 1], color='green', s=0.1, alpha=i["pred_scores"][j])

with open(file, 'rb') as f:
    data = pickle.load(f)
    print(data.keys())
    print(data["map_infos"].keys())
    print(data["map_infos"]["all_polylines"].shape)
    print(data["map_infos"]["road_line"][0])
    # polyline is an array of N array of size 7
    # polyline : [x, y, z, dir_x, dir_y, dir_z, global_type]

    fig, ax = plt.subplots()

    pred = get_prediction()

    def animate_func(j):
        ax.clear()
        plot_frame(ax, data, j)
        add_prediction_to_plot(ax, pred)

    # animate the trajectories with rectangles    
    # the animation should last 8 seconds 
    interval = 1000/(data["track_infos"]["trajs"].shape[1] / 8)
    anim = FuncAnimation(fig, animate_func, frames=range(0, data["track_infos"]["trajs"].shape[1]), interval=interval)
    
    ax.set_aspect('equal')
    ax.set_box_aspect(1)
    # play the animation
    anim.save('animation.gif', writer='pillow', fps=30, dpi=750)
    
    # plot_frame(ax, data, 0)
    # plt.show()
