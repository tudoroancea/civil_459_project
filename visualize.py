import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
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
init_filename = "sample_867dd000677d389.pkl"
filename = init_filename
file = path + filename
# file = "../data/processed_scenarios_training/sample_867dd000677d389.pkl"


def plot_frame(ax, data, frame):
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


    # load the trajectories
    # trajs : x, y, z, length, width, height, heading, velocity_x, velocity_y, valid
    type = data["track_infos"]["object_type"]
    trajs = data["track_infos"]["trajs"]
    # print(data)


    for i in range(len(type)):
        # remove 0,0 points from the trajectories
        color = 'blue'
        if (i in data["tracks_to_predict"]["track_index"]):
            color = 'red'
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
                rect = Rectangle((trajs[i, index, 0] - 1, trajs[i, index, 1]-2), 2, 4, angle=90 + angle*180/np.pi, color=color, rotation_point='center', zorder=2)
                ax.add_patch(rect)
            elif type[i] == "TYPE_PEDESTRIAN":
                rect = Rectangle((trajs[i, index, 0] - 0.25, trajs[i, index, 1]-0.25), 0.5, 0.5, angle=90 + angle*180/np.pi, color='black', rotation_point='center')
                ax.add_patch(rect)
            elif type[i] == "TYPE_CYCLIST":
                rect = Rectangle((trajs[i, index, 0] - 0.25, trajs[i, index, 1]-0.5), 0.5, 1, angle=90 + angle*180/np.pi, color='green', rotation_point='center')
                ax.add_patch(rect)

            c = np.linspace(0.2, 1.0, first_zero)
            ax.scatter(trajs[i, :first_zero, 0], trajs[i, :first_zero, 1], color="red", s = 0.01, alpha=0.9, zorder=2)

    # build the legend
    legend_elements = [Line2D([0], [0], color='gray', lw=2, label='white Road lines'),
                        Line2D([0], [0], color='yellow', lw=2, label='yellow Road lines'),
                        Line2D([0], [0], color='black', lw=2, label='Road edges'),
                        Line2D([0], [0], color='lightgray', lw=2, label='Lanes'),
                        Line2D([0], [0], color='red', lw=2, label='GT trajectory', ls=':'),
                        Line2D([0], [0], color='green', lw=2, label='Predicted trajectory', ls=':')]
    
    return legend_elements
                                   

def get_prediction(checkpoint_path):
    return apply(filename, checkpoint_path=checkpoint_path)

def add_prediction_to_plot(ax, pred):
    for i in pred:
        trajs = i["pred_trajs"]
        num_trajs = trajs.shape[0]
        for j in range(num_trajs):
            alpha = i["pred_scores"][j] # range : 0 to 1
            ax.scatter(trajs[j, :, 0], trajs[j, :, 1], color="green", s=0.1, alpha=alpha, zorder=2, label="Prediction")

    # add the gradient green as cmap


with open(file, 'rb') as f:
    data = pickle.load(f)
    print(data.keys())
    print(data["map_infos"].keys())
    print(data["map_infos"]["all_polylines"].shape)
    # print(data["map_infos"]["road_line"][0])
    # polyline is an array of N array of size 7
    # polyline : [x, y, z, dir_x, dir_y, dir_z, global_type]

    # two subplots, side by side to compare the models
    fig, ax = plt.subplots(1,2, figsize=(20, 10))

    checkpoint_path = "../checkpoint_epoch_1.pth"
    pred1 = get_prediction(checkpoint_path=checkpoint_path)
    checkpoint_path2 = "../checkpoint_epoch_15.pth"
    pred2 = get_prediction(checkpoint_path=checkpoint_path2)

    cmap = plt.cm.Greens
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    print("model applied")

    def animate_func(j):
        ax[0].clear()
        ax[1].clear() 
        cbar = plt.colorbar(sm, cax=ax[1].inset_axes([1.05, 0.1, 0.02, 0.8]))
        cbar.set_label('Prediction score')
        legends = plot_frame(ax[0], data, j)
        add_prediction_to_plot(ax[0], pred1)
        ax[0].set_title("1 epoch")
        legends = plot_frame(ax[1], data, j)
        add_prediction_to_plot(ax[1], pred2)
        ax[1].set_title("15 epochs")
        # ax[0].legend(handles=legends, loc='upper right', bbox_to_anchor=(0.9, 0.9, 0.1, 0.1), fontsize='xx-small')
        ax[1].legend(handles=legends, loc='upper right', bbox_to_anchor=(0.9, 0.9, 0.1, 0.1), fontsize='xx-small')

    # animate the trajectories with rectangles    
    # the animation should last 8 seconds 
    interval = 1000/(data["track_infos"]["trajs"].shape[1] / 8)
    # ax[0].set_aspect('equal')
    ax[0].set_box_aspect(1)
    # ax[1].set_aspect('equal')
    ax[1].set_box_aspect(1)    
    # hide the axis


    anim = FuncAnimation(fig, animate_func, frames=range(0, data["track_infos"]["trajs"].shape[1]), interval=interval)
    # plot_frame(ax[0], data, 0)   
    # plot_frame(ax[1], data, 0)
    # add_prediction_to_plot(ax[0], pred1)
    # add_prediction_to_plot(ax[1], pred2) 
    plt.tight_layout()
  
    # save the animation
    anim.save(filename + '_animation_1_15.gif', writer='pillow', fps=20, dpi=350)
    

    # plt.show()
