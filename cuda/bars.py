#!/usr/bin/env python3
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import argparse
import PIL # to get image resolution

parser = argparse.ArgumentParser()
parser.add_argument('--path', dest='path', required=True, type=str, help='Path to your data')
parser.add_argument('--threshold-path', dest='threshold_path', required=False, type=str, help='Path to your threshold-only data')
parser.add_argument('--show', dest='show', type=bool, default=False, help='Weather or not to display the data in a window')
#parser.add_argument('--separate', dest='separate', type=bool, default=False, help='Plot each operation in a separate graph')
parser.add_argument('--output', dest='output', default=None, type=str, help='Path to save data to')
args = parser.parse_args()

headers = ['device','filename','upscale_amt','time_cpu_otsu','time_cpu_adaptive','time_gpu_otsu','time_gpu_adaptive','speedup_otsu','speedup_adaptive']

try:
    full_files = glob.glob(args.path + "/full*.csv")
    short_files = glob.glob(args.path + "/short*.csv")

    full_data_frame = pd.DataFrame()
    full_content = []

    for filename in full_files:
        df = pd.read_csv(filename, header=0, names=headers, index_col=False)
        full_content.append(df)

    full_data_frame = pd.concat(full_content)
    full_data_frame = full_data_frame[full_data_frame['upscale_amt'] == 0]
except:
    print("no full data.")

short_data_frame = pd.DataFrame()
short_content = []

for filename in short_files:
    df = pd.read_csv(filename, header=0, names=headers, index_col=False)
    short_content.append(df)

short_data_frame = pd.concat(short_content)

short_data_frame = short_data_frame[short_data_frame['upscale_amt'] == 0]

print(full_data_frame)
print(short_data_frame)

labels = ['chom']
labels.extend(short_data_frame['filename'])

width = 0.1

def bars(data, output):
    my_device = data['device'][0].values[0]
    my_device = my_device.replace(' with Max-Q Design', '')
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(top=0.80)
    ax.bar(np.arange(len(data['time_cpu_otsu'])), data['time_cpu_otsu'], label='time_cpu_otsu', color='#008ec6', width=width)
    ax.bar(np.arange(len(data['time_cpu_otsu'])) + width, data['time_gpu_otsu'], label='time_gpu_otsu', color='#0dc600', width=width)
    ax.bar(np.arange(len(data['time_cpu_adaptive'])) + width * 2, data['time_cpu_adaptive'], label='time_cpu_adaptive', color='#002bc6', width=width)
    ax.bar(np.arange(len(data['time_cpu_adaptive'])) + width * 3, data['time_gpu_adaptive'], label='time_gpu_adaptive', color='green', width=width)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel('File Name')
    ax.set_ylabel('Compute time (ms)')
    ax.set_title(f'Threshold Computation Time ({my_device})')
    resolutions = []

    for file in data['filename']:
        img = PIL.Image.open(f"../samples/{file}")
        wid, hgt = img.size
        resolutions.append(wid * hgt)

    # Plot number of pixels in image
    ax2 = ax.twiny()
    ax2.set_xticks( ax.get_xticks() )
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(["{0:.{1}e}".format(resolutions[int(x)], 2) for x in ax.get_xticks()-1], rotation=20)
    ax2.set_xlabel('Number of Pixels')

    ax.legend()

    if args.output:
        plt.savefig(f'./{args.output}/{output}-bars.png')
        print(f'Graph has been saved to ./{args.output}/{output}-bars.png')

def speedup_bars(data, output):
    my_device = data['device'][0].values[0]
    my_device = my_device.replace(' with Max-Q Design', '')
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(top=0.80)
    ax.bar(np.arange(len(data['time_cpu_otsu'])), data['speedup_otsu'], label='speedup_otsu', color='orange', width=width)
    ax.bar(np.arange(len(data['time_cpu_adaptive'])) + width, data['speedup_adaptive'], label='speedup_adaptive', color='red', width=width)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel('File Name')
    ax.set_ylabel('Speedup')
    ax.set_title(f'Threshold Computation Speedup ({my_device})')
    resolutions = []

    for file in data['filename']:
        img = PIL.Image.open(f"../samples/{file}")
        wid, hgt = img.size
        resolutions.append(wid * hgt)

    # Plot number of pixels in image
    ax2 = ax.twiny()
    ax2.set_xticks( ax.get_xticks() )
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(["{0:.{1}e}".format(resolutions[int(x)], 2) for x in ax.get_xticks()-1], rotation=20)
    ax2.set_xlabel('Number of Pixels')

    ax.legend()

    if args.output:
        plt.savefig(f'./{args.output}/{output}-bars.png')
        print(f'Graph has been saved to ./{args.output}/{output}-bars.png')

try:
    bars(full_data_frame, 'full')
except:
    print("No full data.")
bars(short_data_frame, 'short')

speedup_bars(short_data_frame, 'speedup-short')
