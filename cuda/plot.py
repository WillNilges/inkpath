#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import argparse
import PIL # to get image resolution

parser = argparse.ArgumentParser()
parser.add_argument('--path', dest='path', required=True, type=str, help='Path to your data')
parser.add_argument('--threshold-path', dest='threshold_path', required=False, type=str, help='Path to your threshold-only data')
parser.add_argument('--truncate', dest='truncate', type=int, default=0, help='Number of rows of data to remove')
parser.add_argument('--show', dest='show', type=bool, default=False, help='Weather or not to display the data in a window')
parser.add_argument('--separate', dest='separate', type=bool, default=False, help='Plot each operation in a separate graph')
parser.add_argument('--output', dest='output', type=str, help='Path to save data to')
args = parser.parse_args()

headers = ['device','filename','upscale_amt','time_cpu_otsu','time_cpu_adaptive','time_gpu_otsu','time_gpu_adaptive','speedup_otsu','speedup_adaptive']
data = pd.read_csv(args.path, header=0, names=headers, index_col=False)
try:
    threshold_data = pd.read_csv(args.threshold_path, header=0, names=headers, index_col=False)
except ValueError:
    print("no secondary data");
    threshold_data = None
#matplotlib.use('tkagg')

def truncate(data, headers, rows):
    for header in headers:
        data[header] = data[header][: len(data[header]) - rows]
    return data

def plot_pixels(ax, wid, hgt):
    # Plot number of pixels in image
    ax2 = ax.twiny()
    ax2.set_xticks( ax.get_xticks() )
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(["{0:.{1}e}".format(2**int(x) * (wid * hgt), 2) for x in ax.get_xticks()], rotation=30)
    ax2.set_xlabel('Number of Pixels')


def plot_threshold(data, threshold_data, show = True, save = False, output = '', thresh_type = 'otsu'):
    img = PIL.Image.open(f"../../samples/{data['filename'][0]}")
    wid, hgt = img.size
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.75) # use a lower number to make more vertical space
    # Plot compute time
    ax.plot(data['upscale_amt'], data[f'time_cpu_{thresh_type}'], 'bv', label=f'cpu {thresh_type} time')
    ax.plot(data['upscale_amt'], data[f'time_gpu_{thresh_type}'], 'g^', label=f'gpu {thresh_type} time')
    if (not threshold_data.empty):
        ax.plot(threshold_data['upscale_amt'], threshold_data[f'time_cpu_{thresh_type}'], 'cv', label=f'cpu {thresh_type} time (thresholding only)')
        ax.plot(threshold_data['upscale_amt'], threshold_data[f'time_gpu_{thresh_type}'], 'y^', label=f'gpu {thresh_type} time (thresholding only)')

    # Plot number of pixels in image
    plot_pixels(ax, wid, hgt)

    ax.set_title(f'Time to process {data["filename"][0]} ({thresh_type} Method) ({data["device"][0]})')
    ax.set_xlabel('Upscaling amount') # TODO: Print resolution of the image here? Going to need some changes in the debug file.
    ax.set_ylabel('Compute time (ms)')
    ax.legend()

    if save:
        plt.savefig(f'./{output}/{thresh_type}_time_comparison.png')
        print(f'Graph has been saved to ./{output}/time_comparison.png')

    # Plot speedup
    fig, ax = plt.subplots()
    plt.subplots_adjust(top=0.75) # use a lower number to make more vertical space
    #plt.figure()
    ax.plot(data['upscale_amt'], data[f'speedup_{thresh_type}'], 'kx', label='speedup')
    if (not threshold_data.empty):
        ax.plot(threshold_data['upscale_amt'], threshold_data[f'speedup_{thresh_type}'], 'rx', label=f'speedup (thresholding only)')
    ax.set_title(f'Time to process {data["filename"][0]} vs. Speedup ({thresh_type} Method) ({data["device"][0]})')

    plot_pixels(ax, wid, hgt)

    ax.set_xlabel('Upscaling amount')
    ax.set_ylabel('Speedup')
    ax.legend()
    if save:
        plt.savefig(f'./{output}/{thresh_type}_speedup.png')
        print(f'Graph has been saved to ./{output}/speedup.png')

    if show:
        print(f'Showing graphs...')
        plt.show()

if (args.truncate != 0):
    data = truncate(data, headers, args.truncate)
    threshold_data = truncate(threshold_data, headers, args.truncate)
print(data)
#print(threshold_data)

display_data = args.show
save_data = False
if args.output != None:
    save_data = True

plot_threshold(data, threshold_data, show=display_data, save=save_data, output=args.output, thresh_type='otsu')
plot_threshold(data, threshold_data, show=display_data, save=save_data, output=args.output, thresh_type='adaptive')
