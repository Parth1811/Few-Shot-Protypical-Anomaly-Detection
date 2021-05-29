import argparse
import pygame, sys
from pygame.locals import *
import numpy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import os
from os.path import join

FPS = 30
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 576
WHITE = (255,255,255)
plotsize = (SCREEN_WIDTH, int(2*SCREEN_WIDTH/6))
imgsize = (SCREEN_WIDTH, int(4*SCREEN_WIDTH/6))

matplotlib.use("Agg")
fig = plt.figure(figsize=[6, 2])
ax = fig.add_subplot(111)
canvas = agg.FigureCanvasAgg(fig)


def display_image(screen, image_path, size=(SCREEN_WIDTH, SCREEN_HEIGHT), pos = (0,0)):
    image = pygame.image.load(image_path)
    image = pygame.transform.scale(image, size)
    screen.blit(image, pos)

def plot(data, total=None):
    ax.cla()
    ax.plot(data[0])
    ax.plot(data[1])
    ax.set_ylim([0, 1.1])
    if total is not None:
        ax.set_xlim([0, total])
    canvas.draw()
    renderer = canvas.get_renderer()

    raw_data = renderer.tostring_rgb()
    size =  canvas.get_width_height()

    return pygame.image.fromstring(raw_data, size, "RGB")


def run(img_files, values, file_path, save_video_flag):
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Video Player with Graph')

    running = True
    start_trip = False
    counter = 0
    start = False
    data = [[],[]]

    while running:

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                  running = False
                if event.key == pygame.K_UP:
                    start = True
                if event.key == pygame.K_SPACE:
                    start = not start
            if event.type == pygame.QUIT:
                running = False

        if start:
            display_image(screen, join(file_path, img_files[counter]), imgsize)
            image = pygame.transform.scale(plot(data, len(values[0])), plotsize)
            screen.blit(image, (0, int(4*screen.get_height()/6)))
            data = values[:,:counter]

            if counter < len(values[0]):
                if save_video_flag:
                    pygame.image.save(screen, "output/frame_%05d.jpeg" % counter)
                counter += 1

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    quit()





if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="GUI for viewing output of evalution")
    parser.add_argument('--video_frames_path', type=str, default=None, help='Path of the folder containg the video frames', required=True)
    parser.add_argument('--processed_npy_file', type=str, default=None, help='path of the output npy file for the video frames', required=True)
    parser.add_argument('--save_video', type=bool, default=False, help='Save the processed video')
    args = parser.parse_args()


    values = numpy.load(args.processed_npy_file)
    values[1] = 1-values[1]
    file_path = args.video_frames_path
    img_files = os.listdir(file_path)
    img_files.sort()
    if args.save_video:
        try:
            os.mkdir("output")
        except:
            print("All the data will overwritten, to avoid this empty your output folder")

    run(img_files, values, file_path, args.save_video)
