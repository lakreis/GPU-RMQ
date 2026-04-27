from collections import OrderedDict, defaultdict
import random

save_as="svg"
SIZE_MULT = 1.3
plot_dir="../plots/"

line_styles = OrderedDict(
       [('solid',               (0, ())),
        ('loosely dotted',      (0, (1, 10))),
        ('dotted',              (0, (1, 5))),
        ('densely dotted',      (0, (1, 1))),

        ('loosely dashed',      (0, (5, 10))),
        ('dashed',              (0, (5, 5))),
        ('densely dashed',      (0, (5, 1))),

        ('loosely dashdotted',  (0, (4, 10, 1, 10))),
        ('dashdotted',          (0, (4, 5, 1, 5))),
        ('densely dashdotted',  (0, (4, 1, 1, 1))),

        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        
        ('loosely dashdotdotdotted', (0, (3, 10, 1, 10, 1, 10, 1, 10))),
        ('dashdotdotdotted',         (0, (3, 5, 1, 5, 1, 5, 1, 5))),
        ('densely dashdotdotdotted', (0, (3, 1, 1, 1, 1, 1, 1, 1)))]
        )

lrLabels = {
    -1: "Large $(l,r)$ Range",
    -2: "Medium $(l,r)$ Range",
    -3: "Small $(l,r)$ Range",
    -6: "Mixed $(l,r)$ Range"
}

def get_random_linestyle():
    poss_linestyles = ['densely dotted', 'densely dashed', 'densely dashdotted', 'densely dashdotdotdotted', 'densely dashdotdotted']
    return line_styles[poss_linestyles[random.randrange(0, len(poss_linestyles))]]

linestyles = defaultdict(get_random_linestyle)

linestyles["HRMQ@128c"] = line_styles['densely dashed']
linestyles["RTXRMQ"] = line_styles['solid']
linestyles["SER"] = line_styles['densely dashed']
linestyles["LCA"] = line_styles['densely dashdotdotdotted']
linestyles["Exhaustive"] = line_styles['densely dashdotted']

linestyles["XXX"] = line_styles['densely dashdotdotted']
linestyles["XXX_without_shuffles"] = line_styles['densely dashdotdotdotted']
linestyles["Interleaved"] = line_styles['solid']
linestyles["Interleaved2"] = line_styles['densely dashdotted']
linestyles["Interleaved_in_CUDA"] = line_styles['densely dotted']
linestyles["Hierarchical_vector_load"] = line_styles['densely dotted']
linestyles["Interleaved_in_OptiX"] = line_styles['densely dashed']
linestyles["GPU_RMQ"] = line_styles['densely dashdotdotted']
linestyles["XXX_multiload"] = line_styles['densely dashed']

linestyles["RTXAda4090"] = line_styles['densely dashed']
linestyles["RTX5090"] = line_styles['solid']
linestyles["RTX6000"] = line_styles['densely dotted']
linestyles["RTX3090"] = line_styles['solid']

def get_random_color():
    poss_colors = ["#6B6B6B", "#F5BA99", "black", "#3A8AD2", "#DE5A18", "#C1002A", "#163E64", "green"]
    return poss_colors[random.randrange(0, len(poss_colors))]

colors = defaultdict(get_random_color)

colors["HRMQ@128c"] = "#FF81FF" #light pink
colors["SER"] = "#6B6B6B" # grey
colors["RTXRMQ"] = "#3A8AD2" # lighter blue
colors["LCA"] = "#DE5A18" # orange
colors["Exhaustive"] = "#6B6B6B" # grey
colors["XXX"] = "#00BFA9" # blue
colors["Interleaved"] = "#3A8AD2" # lighter blue
colors["Interleaved2"] = "#6A0DAD"  # Purple
colors["Interleaved_in_CUDA"] = "#E2BC00" # Yellow (Gold)
colors["Hierarchical_vector_load"] = "#E2BC00" # Yellow (Gold)
colors["Interleaved_in_OptiX"] = "#DE5A18" # orange
colors["XXX_without_shuffles"] = "#163E64" # blue
colors["GPU_RMQ"] = "#00BFA9" # blue 
colors["XXX_multiload"] = "#DE5A18" # orange

colors["RTXAda4090"] = "#163E64" # blue
colors["RTX5090"] = "#DE5A18" # orange
colors["RTX6000"] = "#C1002A" # red
colors["RTX3090"] = "#DE5A18" # orange

all_colors = [
    "#C1002A",  # Red
    "#163E64",  # Navy Blue
    "#FF81FF",  # Pink / Magenta
    "#6B6B6B",  # Gray
    "#7EBB26",  # Lime Green
    "#DE5A18",  # Orange
    "#3A8AD2",  # Blue
    "#008080",  # Teal
    "#6A0DAD",  # Purple
    "#00CFC1",  # Turquoise
    "#FFD700",  # Yellow (Gold)
    "#5C4033",  # Dark Brown
    "#6EC3F4",  # Sky Blue
    "#800000",  # Maroon
    "#808000",  # Olive
    "#FF6F61",   # Coral
    "#FF4500",  # Orange Red
    "#2E8B57",  # Sea Green
    "#9370DB",  # Medium Purple
    "#F4A460",  # Sandy Brown
    "#1E90FF",  # Dodger Blue
    "#FFC0CB",  # Light Pink
    "#A52A2A",   # Brown
    "#163E64",  # Navy Blue
    "#6B6B6B",  # Gray
    "#3A8AD2",  # Blue
    "#C1002A",  # Red
    "#DE5A18",  # Orange
    "#FF81FF",  # Pink / Magenta
    "#7EBB26",  # Lime Green
    "#008080",  # Teal
    "#6A0DAD",  # Purple
    "#00CFC1",  # Turquoise
    "#FFD700",  # Yellow (Gold)
    "#5C4033",  # Dark Brown
    "#6EC3F4",  # Sky Blue
    "#800000",  # Maroon
    "#808000",  # Olive
    "#FF6F61",   # Coral
    "#FF4500",  # Orange Red
    "#2E8B57",  # Sea Green
    "#9370DB",  # Medium Purple
    "#F4A460",  # Sandy Brown
    "#1E90FF",  # Dodger Blue
    "#FFC0CB",  # Light Pink
    "#A52A2A"   # Brown
]

all_linestyles = [line_styles['solid'], line_styles['densely dashed'], line_styles['densely dotted'], line_styles['densely dashdotted'], line_styles['densely dotted'], line_styles['solid'], line_styles['densely dashdotdotdotted'], line_styles['densely dashdotdotted'], line_styles['densely dashdotted'], line_styles['densely dotted'], line_styles['densely dashed'], line_styles['solid'], line_styles['densely dashdotdotdotted'], line_styles['densely dashdotdotted'], line_styles['densely dashdotted'], line_styles['densely dotted'], line_styles['densely dashed'], line_styles['solid'], line_styles['densely dashdotdotdotted'], line_styles['densely dashdotdotted'], line_styles['densely dashdotted'], line_styles['densely dotted'], line_styles['densely dashed'], line_styles['solid'], line_styles['densely dashdotdotdotted'], line_styles['densely dashdotdotted'], line_styles['densely dashdotted'], line_styles['densely dotted'], line_styles['densely dashed'], line_styles['solid'], line_styles['densely dashdotdotdotted'], line_styles['densely dashdotdotted'], line_styles['densely dashdotted'], line_styles['densely dotted'], line_styles['densely dashed'], line_styles['solid']]

orders = defaultdict(lambda: 2)

defined_orders={
    "HRMQ@64c": 2,
    "HRMQ@128c": 2,
    "RTXRMQ": 5,
    "SER": 4,
    "LCA": 3,
    "Exhaustive": 1,
    "XXX": 5,
    "Interleaved": 12,
    "Interleaved2": 4,
    "Interleaved_in_CUDA": 4.7,
    "Hierarchical_vector_load": 3,
    "Interleaved_in_OptiX": 4.5,
    "XXX_without_shuffles": 3.6,
    "GPU_RMQ": 5,
    "XXX_multiload": 4.9
}

for val in defined_orders.keys():
    orders[val] = defined_orders[val]

alphas = defaultdict(lambda: 1)

data_path = "../data/"

split_GPU_RMQ_value = 23
files = {
    "HRMQ@64c": None,
    "HRMQ@128c": lambda _: f"{data_path}perfexp-3990X-ALG1.csv",
    "RTXRMQ": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG5.csv",
    "SER": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG10.csv",
    "LCA": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG7.csv",
    "Exhaustive": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG2.csv",
    "XXX": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG16.csv",
    "GPU_RMQ": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG16.csv",
    "Interleaved": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG17.csv",
    "Interleaved2": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG18.csv",
    "Interleaved_in_CUDA": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG19.csv",
    "Hierarchical_vector_load": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG20.csv",
    "Interleaved_in_OptiX": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG21.csv",
    "XXX_without_shuffles": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG23.csv",
    "XXX_multiload": lambda device_name="RTXAda4090": f"{data_path}perfexp-{device_name}-ALG24.csv"
}

start_times = defaultdict(lambda: None )
end_times = defaultdict(lambda: None)

# start_times["GPU_RMQ"] = "2025-11-30T23:00:00Z" 

nb_log_precision_upper_bound = defaultdict(lambda: None)

nb_log_precision_upper_bound["RTXRMQ"] = 24
nb_log_precision_upper_bound["SER"] = 24

bs_log_precision_upper_bound = defaultdict(lambda: None)

bs_log_precision_upper_bound["RTXRMQ"] = 17
bs_log_precision_upper_bound["SER"] = 17

final_label_names = {
    "HRMQ@128c": "HRMQ (CPU)", 
    "RTXRMQ": "RTXRMQ",
    "SER": "RTXRMQ (with SER)",
    "LCA": "LCA",
    "Exhaustive": "Full GPU Scan",
    "XXX": "GPU-RMQ (CL)",
    "Interleaved": "Interleaved (more arrays)",
    "Interleaved2": "GPU-RMQ (CL) w/o warp intrinsics in OptiX w/ RT",
    "Interleaved_in_CUDA": "GPU-RMQ (CL) w/o warp intrinsics",
    "Hierarchical_vector_load": "GPU-RMQ (VL)",
    "GPU_RMQ": "GPU-RMQ",
    "Interleaved_in_OptiX": "GPU-RMQ (CL) w/o warp intrinsics in OptiX w/o RT",
    "XXX_without_shuffles": "GPU-RMQ (CL) w/o shuffles",
    "XXX_multiload": "GPU-RMQ (CL) multi load"
}

final_hyperpara_name = {
    "GPU_BSIZE": "GPU blocksize",
    "scan_threshold": "t",
    "XXX_CG_SIZE_LOG": "g",
    "XXX_CG_AMOUNT_LOG": "CG amount",
    "XXX_chunk_size": "c",
    "build_threshold": "build threshold"
}

final_device_names = {
    "RTXAda4090": "NVIDIA RTX 4090",
    "RTX5090": "NVIDIA RTX 5090",
    "RTX6000": "NVIDIA RTX 6000",
    "RTX3090": "NVIDIA RTX 3090"
}

# sizes and pads of the plots
range_font_size = 18
range_pad = 10
x_label_font_size = 16
x_label_pad = 8
x_ticks_font_size = 16
x_ticks_pad = 8
y_label_font_size = 16
y_label_pad = 10
y_ticks_font_size = 14
#y_ticks_pad = 8

outer_legend_font_size = 12
inner_legend_font_size = 10

#optimal for 2^20 <= n <= 2^24
Hierarchical_vector_load_optimal_scan_threshold = 32
Hierarchical_vector_load_optimal_amount_log = 1
