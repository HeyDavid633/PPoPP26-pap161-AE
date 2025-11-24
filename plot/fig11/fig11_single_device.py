import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import re
import matplotlib as mpl


def parse_file(file_path, is_mcfuser=False):
    data = {}
    current_model = None

    with open(file_path, 'r') as f:
        for line in f:

            model_match = re.search(r'\[model\] Unified bench test for (\w+)', line)
            if model_match:
                current_model = model_match.group(1)
                continue

            if is_mcfuser:
                submodel_match = re.search(r'\[(\w+)\] Unified bench test for', line)
                if submodel_match:
                    current_model = submodel_match.group(1)
                    continue


            pattern = r'.*?bs:(\d+).*?seq:(\d+).*?\|\s+([\w\s]+)\s*:\s*([\d.]+)'
            match = re.search(pattern, line)
            if match:
                bs = int(match.group(1))
                seq = int(match.group(2))
                method = match.group(3).strip()
                time = float(match.group(4))

                if is_mcfuser:
                    method = "MCFuser"

                if bs not in data:
                    data[bs] = {}
                if seq not in data[bs]:
                    data[bs][seq] = {}
                if current_model not in data[bs][seq]:
                    data[bs][seq][current_model] = {}

                data[bs][seq][current_model][method] = time
    return data

def parse_file2(file_path, is_mcfuser=False):
    data = {}
    current_model = None

    with open(file_path, 'r') as f:
        for line in f:
            model_match = re.search(r'e2e (\w+)', line)
            if model_match:
                current_model = model_match.group(1)

            if is_mcfuser:
                submodel_match = re.search(r'$$\w+$$', line)
                if submodel_match:
                    continue

            pattern = r'.*?bs:(\d+).*?seq:(\d+).*?\|\s+([\w\s]+)\s*:\s*([\d.]+)'
            match = re.search(pattern, line)
            if match and current_model:
                bs = int(match.group(1))
                seq = int(match.group(2))
                method = match.group(3).strip()
                time = float(match.group(4))

                if is_mcfuser:
                    method = "MCFuser"

                data.setdefault(bs, {}).setdefault(seq, {}).setdefault(current_model, {})[method] = time

    return data


import torch
gpu_name = torch.cuda.get_device_name()
gpu_info = ""
if "A100" in gpu_name:
    gpu_info = "A100"
elif "4090" in gpu_name:
    gpu_info = "4090"


import argparse
parser = argparse.ArgumentParser(description='Plot End2End performance')
parser.add_argument('--all_data_device', default="../../data/End2End_Performance/all_" +gpu_info+ ".txt", help="Path to All data file")
parser.add_argument('--bolt_device', default="../../data/End2End_Performance/" + gpu_info + "bolt.txt", help="Path to Bolt data file")


args = parser.parse_args()
data1 = parse_file2(args.all_data_device)
data2 = parse_file(args.bolt_device)



def combinedData(data1,data2):
    combined_data = {}
    target_pairs = [
        (1, 128),
        (8, 512),
        (16, 2048),
    ]

    methods_mapping = {

        'MCFuser': ('data1', 'MCFuser'),
        'Torch Native': ('data1', 'Torch Native'),
        'Torch Compile': ('data1', 'Torch Compile'),
        'ByteTransformer': ('data1', 'ByteTransformer'),
        'STOF': ('data1', 'STOF'),
        'Bolt': ('data2', 'Bolt')
    }

    for bs in [1, 8, 16]:
        combined_data[bs] = {}

        target_seqs = [seq for b, seq in target_pairs if b == bs]

        for seq in target_seqs:
            combined_data[bs][seq] = {}

            for model in [ 'bert_base', 'bert_large', 'gpt', 't5' , 'llama_base' , 'vit_base']:
                combined_data[bs][seq][model] = {}


                for method, (source, key) in methods_mapping.items():
                    if source == 'data1':
                        src_data = data1
                    elif source == 'data2':
                        src_data = data2

                    val = src_data.get(bs, {}).get(seq, {}).get(model, {}).get(key, 0)
                    combined_data[bs][seq][model][method] = val
    return combined_data

combined_data1=combinedData(data1,data2)



def normalizedData(combined_data):
    normalized_data = {}
    for bs in combined_data:
        normalized_data[bs] = {}
        for seq in combined_data[bs]:
            normalized_data[bs][seq] = {}
            for model in combined_data[bs][seq]:
                model_data = combined_data[bs][seq][model]
                tn_time = model_data.get('Torch Native', 0)

                normalized = {}
                for method, value in model_data.items():

                    if tn_time > 0 and value > 0:
                        normalized[method] = tn_time / value
                    else:
                        normalized[method] = 0

                if tn_time > 0:
                    normalized['Torch Native'] = 1.0

                normalized_data[bs][seq][model] = normalized
    return normalized_data

normalized_data1=normalizedData(combined_data1)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 25

MODELS = ['bert_base', 'bert_large', 'gpt', 'llama_base', 't5', 'vit_base']
models = ['BERT-Base', 'BERT-Large', 'GPT-2', 'LLaMA', 'T5', 'ViT']
BS_SEQ_PAIRS = [(1, 128), (8, 512), (16, 2048)]
METHODS = ['Torch Native', 'MCFuser', 'ByteTransformer', 'Bolt', 'Torch Compile', 'STOF']
COLORS = ['#999696', '#f4b1c9', '#f2edb7', '#dff1d7', '#afc8ea', '#d06c5a']
LABEL_NAMES = ['PyTorch Native', 'MCFuser', 'ByteTransformer', 'Bolt', 'PyTorch Compile', 'STOF']


def extract_data(normalized_data):
    return {
        (bs, seq): {model: [normalized_data[bs][seq][model][method] for method in METHODS]
                    for model in MODELS}
        for bs, seq in BS_SEQ_PAIRS
    }


data1 = extract_data(normalized_data1)



def plot_subplot(ax, data, panel_label):
    n_models = len(MODELS)
    n_methods = len(METHODS)
    bar_width = 0.8
    model_spacing = 2.0
    group_width = n_models * (n_methods * bar_width) + (n_models - 1) * model_spacing
    group_spacing = 3.0

    model_centers = []
    bs_seq_centers = []

    for pair_idx, (bs, seq) in enumerate(BS_SEQ_PAIRS):
        x_group_start = pair_idx * (group_width + group_spacing)
        bs_seq_center = x_group_start + group_width / 2
        bs_seq_centers.append(bs_seq_center)

        for model_idx, model in enumerate(MODELS):
            x_model_start = x_group_start + model_idx * (n_methods * bar_width + model_spacing)
            model_center = x_model_start + (n_methods * bar_width) / 2
            model_centers.append(model_center)

            values = data[(bs, seq)][model]

            for method_idx, value in enumerate(values):
                x = x_model_start + method_idx * bar_width

                rect = ax.bar(x, value, width=bar_width,
                              color=COLORS[method_idx],
                              edgecolor='black',
                              linewidth=0.8, zorder=3)

                if value > 4:
                    ax.text(x - 0.6, 3.9,
                            f"{value:.1f}",
                            ha='right', va='top',
                            fontsize=22.57,
                            color='black')


    for i in range(1, len(BS_SEQ_PAIRS)):
        offset_xian = -0.4 if i == 1 else -0.2
        x = i * (group_width + group_spacing + offset_xian) - group_spacing / 2
        ax.axvline(x, color='gray', linestyle='--', linewidth=3)


    ax.set_ylabel('Speedup', fontsize=34)
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.grid(True, linestyle=(0, (10, 5)), linewidth=1)
    ax.set_xticks([])
    ax.margins(x=0.01)


    label_y_positions = {
        'model': -0.05,
        'bs_seq': -2.05,
        'panel': -2.75
    }


    for i, (bs, seq) in enumerate(BS_SEQ_PAIRS):
        ax.text(bs_seq_centers[i], label_y_positions['bs_seq'],
                f'({bs}, {seq})',
                ha='center', va='top', fontsize=32)


    for i, model_name in enumerate(models):
        if model_name in ['BERT-Base', 'BERT-Large']:
            offset = -4.0  
        elif model_name == 'GPT-2':
            offset = -1.5  
        elif model_name == 'LLaMA':
            offset = -1.0 
        else:
            offset = 0.0  

    
        ax.text(model_centers[i] + offset, label_y_positions['model'],
                model_name,
                ha='center', va='top', fontsize=24, rotation=45)

  
        for group_idx in range(1, len(BS_SEQ_PAIRS)):
            ax.text(model_centers[i + group_idx * n_models] + offset, label_y_positions['model'],
                    model_name,
                    ha='center', va='top', fontsize=24, rotation=45)


    total_width = len(BS_SEQ_PAIRS) * (group_width + group_spacing) - group_spacing
    ax.text(total_width / 2, label_y_positions['panel'],
            panel_label,
            ha='center', va='top', fontsize=34)



fig, (ax1) = plt.subplots(1, 1, figsize=(40, 3.6), dpi=300, gridspec_kw={'wspace': 0.08})



plot_subplot(ax1, data1, '(a) NVIDIA RTX 4090 GPU')



legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black')
                   for c in COLORS]
fig.legend(legend_elements, LABEL_NAMES,
           loc='upper center',
           ncol=6,
           frameon=False,
           fontsize=34.4,
           bbox_to_anchor=(0.5, 1.16))


plt.subplots_adjust(wspace=0.12)
plt.savefig('5-eva-EE-' + gpu_info + '.pdf',
            bbox_inches='tight',
            dpi=300,
            facecolor='white')

plt.close()