#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', dest='path', required=True, type=str, help='Path to your data')
parser.add_argument('--truncate', dest='truncate', type=int, default=0, help='Number of rows of data to remove')
parser.add_argument('--show', dest='show', type=bool, default=False, help='Weather or not to display the data in a window')
parser.add_argument('--separate', dest='separate', type=bool, default=False, help='Plot each operation in a separate graph')
parser.add_argument('--output', dest='output', type=str, help='Path to save data to')
args = parser.parse_args()

headers = ['filename','upscale_amt','time_cpu_otsu','time_cpu_adaptive','time_gpu_otsu','time_gpu_adaptive','speedup_otsu','speedup_adaptive']
data = pd.read_csv(args.path, header=1, names=headers, index_col=False)
#matplotlib.use('tkagg')

def truncate(data, headers, rows):
    for header in headers:
        data[header] = data[header][: len(data[header]) - rows]
    return data

def plot_threshold(data, show = True, save = False, output = '', thresh_type = 'otsu'):
    fig, ax = plt.subplots()
    # Plot compute time
    ax.plot(data['upscale_amt'], data[f'time_cpu_{thresh_type}'], 'bv', label='cpu otsu threshold time')
    ax.plot(data['upscale_amt'], data[f'time_gpu_{thresh_type}'], 'g^', label='gpu otsu threshold time')
    ax.set_title(f'Time to process {data["filename"][0]} ({thresh_type} Method)')
    ax.set_xlabel('Upscaling amount') # TODO: Print resolution of the image here? Going to need some changes in the debug file.
    ax.set_ylabel('Compute time (ms)')
    ax.legend()

    if save:
        plt.savefig(f'./{output}/{thresh_type}_time_comparison.png')
        print(f'Graph has been saved to ./{output}/time_comparison.png')

    # Plot speedup
    fig, ax = plt.subplots()
    #plt.figure()
    ax.plot(data['upscale_amt'], data[f'speedup_{thresh_type}'], 'kx', label='speedup')
    ax.set_title(f'Time to process {data["filename"][0]} vs. Speedup ({thresh_type} Method)')
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
print(data)

display_data = args.show
save_data = False
if args.output != None:
    save_data = True

plot_threshold(data, show=display_data, save=save_data, output=args.output, thresh_type='otsu')
plot_threshold(data, show=display_data, save=save_data, output=args.output, thresh_type='adaptive')
