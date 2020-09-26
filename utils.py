import csv
import os
import matplotlib.pyplot as plt


def add_scores_to_file(scores_file_path, model_version, steps, scores):
    file_name = 'IS_' + model_version + '.csv'
    file_dir = os.path.join(scores_file_path, file_name)

    with open(file_dir, 'a', newline='', delimiter=':') as file:
        writer = csv.writer(file)
        for i in range(len(steps)):
            writer.writerow([steps[i], scores[i]])


def save_scores_plot(scores_file_path, model_version, steps, scores, epoch):
    plt.plot(steps, scores)
    plt.xlabel('Iteration')
    plt.ylabel('Inception score')
    plt.title('IS ' + model_version)
    #plt.show()
    plot_name = 'IS_{}_plot_{}.png'.format(model_version, epoch)
    #plot_name = 'IS_' + model_version + '_plot.png'
    plot_dir = os.path.join(scores_file_path, plot_name)
    plt.savefig(plot_dir)


class CheckpointData():
    def __init__(self, start_from_epoch=0, start_from_step=0, prev_scores_steps=[], prev_scores=[]):
        self.start_from_epoch = start_from_epoch
        self.start_from_step = start_from_step
        self.prev_scores_steps = prev_scores_steps
        self.prev_scores = prev_scores
