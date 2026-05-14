import gradio as gr
import numpy as np
import pandas as pd
import struct
import os
import kagglehub
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

matplotlib.rcParams['figure.facecolor'] = '#0d1f3c'
matplotlib.rcParams['axes.facecolor'] = '#0d1f3c'
matplotlib.rcParams['axes.edgecolor'] = '#1e3a5f'
matplotlib.rcParams['text.color'] = '#a8c4e0'
matplotlib.rcParams['axes.titlecolor'] = '#a8c4e0'
matplotlib.rcParams['xtick.color'] = '#a8c4e0'
matplotlib.rcParams['ytick.color'] = '#a8c4e0'

# Ensure TensorFlow uses eager execution for easier debugging if needed
tf.config.run_functions_eagerly(True)

# --- 1. Load MNIST Data ---
# Download the dataset and construct the file path
dataset_dir = kagglehub.dataset_download("hojjatk/mnist-dataset")
file_path = os.path.join(dataset_dir, "train-images-idx3-ubyte", "train-images-idx3-ubyte")

df = None
image_dimension = 28

try:
    with open(file_path, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}. Expected 2051 for image file.")
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
    images = image_data.reshape(num_images, num_rows, num_cols)
    flattened_images = images.reshape(num_images, num_rows * num_cols)
    df = pd.DataFrame(flattened_images)
    image_dimension = num_rows
    print(f"Successfully loaded {num_images} MNIST images.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except ValueError as e:
    print(f"Data parsing error: {e}")
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")

# --- 2. Define Convolution Kernels ---
kernels3 = {
    'Identity': np.array([[0, 0, 0],
                          [0, 1, 0],
                          [0, 0, 0]], dtype=np.float32),
    'Edge Detection (Laplacian)': np.array([[-1, -1, -1],
                                            [-1,  8, -1],
                                            [-1, -1, -1]], dtype=np.float32),
    'Sharpen': np.array([[ 0, -1,  0],
                         [-1,  5, -1],
                         [ 0, -1,  0]], dtype=np.float32),
    'Blur (Box)': np.array([[1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9]], dtype=np.float32),
    'Emboss': np.array([[-2, -1,  0],
                        [-1,  1,  1],
                        [ 0,  1,  2]], dtype=np.float32)
}

kernels5 = {
    'Identity': np.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0]], dtype=np.float32),
    'Edge Detection (Laplacian)': np.array([[-1, -1, -1, -1, -1],
                                            [-1, -1, -1, -1, -1],
                                            [-1, -1, 8, -1, -1],
                                            [-1, -1, -1, -1, -1],
                                            [-1, -1, -1, -1, -1]], dtype=np.float32),
    'Sharpen': np.array([[ 0, 0, -1, 0,  0],
                         [ 0, 0, -1, 0,  0],
                         [-1, -1, 5, -1, -1],
                         [ 0, 0, -1, 0,  0],
                         [ 0, 0, -1, 0,  0]], dtype=np.float32),
    'Blur (Box)': np.array([[1/9, 1/9, 1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9, 1/9, 1/9],
                            [1/9, 1/9, 1/9, 1/9, 1/9]], dtype=np.float32),
    'Emboss': np.array([[-2, -1, -1, 0, 0],
                        [-1, -1, 0, 0, 0],
                        [-1, 0, 1, 0,  1],
                        [ 0, 0, 0, 1, 1],
                        [ 0, 0, 1, 1, 2]], dtype=np.float32)
}
kernel_types = list(kernels3.keys())

# --- 3. Convolution Function ---
def perform_single_convolution(image_data, kernel_type, kernel_size, dilations, strides, padding_size):
    if image_data is None:
        return None, "No image selected or image failed to load."

    image_tensor = tf.constant(image_data[tf.newaxis, :, :, tf.newaxis], dtype=tf.float32)

    kernel = kernels3[kernel_type] if kernel_size == 3 else kernels5[kernel_type]
    kernel_tensor = tf.constant(kernel[:, :, tf.newaxis, tf.newaxis], dtype=tf.float32)

    if padding_size > 0:
        paddings = tf.constant([[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
        image_tensor = tf.pad(image_tensor, paddings, "CONSTANT")
        padding_mode = 'VALID'
    else:
        padding_mode = 'VALID'

    try:
        output_tensor = tf.nn.conv2d(
            input=image_tensor,
            filters=kernel_tensor,
            strides=[1, strides, strides, 1],
            padding=padding_mode,
            dilations=[1, dilations, dilations, 1]
        )
        return output_tensor[0, :, :, 0].numpy(), None
    except Exception as err:
        return None, f"Error during convolution: {err}"


def make_image_figure(img_data, title, cmap='gray'):
    """Create a single-image matplotlib figure with dark blue styling."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    fig.patch.set_facecolor('#0d1f3c')
    ax.set_facecolor('#0d1f3c')
    if img_data is not None:
        ax.imshow(img_data, cmap=cmap, interpolation='nearest')
    ax.set_title(title, color='#a8c4e0', fontsize=11, pad=8)
    ax.axis('off')
    # Subtle border
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    plt.tight_layout(pad=0.5)
    return fig


# --- 4. Interactive Convolution Function for Gradio ---
def interactive_convolution(
    image_input_label,
    kernel_type_1, kernel_size_1, dilations_1, strides_1, padding_size_1,
    kernel_type_2, kernel_size_2, dilations_2, strides_2, padding_size_2
):
    global df, image_dimension

    if df is None:
        placeholder = make_image_figure(None, "No data")
        err_msg = "Error: MNIST data not loaded."
        return placeholder, placeholder, placeholder, err_msg, err_msg, err_msg, "```\nLoading...\n```", "```\nLoading...\n```"

    selected_index = int(image_input_label.split('(idx: ')[1][:-1])
    image_input = df.loc[selected_index].values.reshape(image_dimension, image_dimension).astype(np.float32) / 255.0

    # --- Original image figure ---
    original_fig = make_image_figure(image_input, 'Original Image')

    # --- Setup 1 ---
    output_image_data_1 = None
    output_shape_1_str = "N/A"
    rf_1_str = "N/A"
    warning_1 = ""

    kernel_1 = kernels3[kernel_type_1] if kernel_size_1 == 3 else kernels5[kernel_type_1]
    kernel_matrix_1_str = f"```\n{np.array_str(kernel_1, precision=3, suppress_small=True)}\n```"

    output_image_data_1, err_1 = perform_single_convolution(
        image_input, kernel_type_1, kernel_size_1, dilations_1, strides_1, padding_size_1
    )
    if err_1:
        warning_1 = f"Setup 1 Error: {err_1}"
    elif output_image_data_1 is not None:
        output_shape_1_str = f"{output_image_data_1.shape}"
        rf_1 = kernel_size_1 + (kernel_size_1 - 1) * (dilations_1 - 1)
        rf_1_str = f"{rf_1}×{rf_1}"

    fig_1 = make_image_figure(
        output_image_data_1 if output_image_data_1 is not None else image_input,
        f'Kernel type: {kernel_type_1}'
    )

    # --- Setup 2 ---
    output_image_data_2 = None
    output_shape_2_str = "N/A"
    rf_2_str = "N/A"
    warning_2 = ""

    kernel_2 = kernels3[kernel_type_2] if kernel_size_2 == 3 else kernels5[kernel_type_2]
    kernel_matrix_2_str = f"```\n{np.array_str(kernel_2, precision=3, suppress_small=True)}\n```"

    output_image_data_2, err_2 = perform_single_convolution(
        image_input, kernel_type_2, kernel_size_2, dilations_2, strides_2, padding_size_2
    )
    if err_2:
        warning_2 = f"Setup 2 Error: {err_2}"
    elif output_image_data_2 is not None:
        output_shape_2_str = f"{output_image_data_2.shape}"
        rf_2 = kernel_size_2 + (kernel_size_2 - 1) * (dilations_2 - 1)
        rf_2_str = f"{rf_2}×{rf_2}"

    fig_2 = make_image_figure(
        output_image_data_2 if output_image_data_2 is not None else image_input,
        f'Kernel type: {kernel_type_2}'
    )

    # --- Info texts ---
    output_text_1 = f"Output Size: {output_shape_1_str}\nReceptive Field: {rf_1_str}"
    output_text_2 = f"Output Size: {output_shape_2_str}\nReceptive Field: {rf_2_str}"
    if warning_1:
        output_text_1 = warning_1 + "\n\n" + output_text_1
    if warning_2:
        output_text_2 = warning_2 + "\n\n" + output_text_2

    # --- Image metadata ---
    image_meta = (
        f"**Size:** {image_dimension}×{image_dimension}\n\n"
        f"**Index:** {selected_index}\n\n"
        f"**Source:** MNIST\n"
        f"Source: {file_path}" if df is not None else f"No data loaded"
    )

    return original_fig, fig_1, fig_2, output_text_1, output_text_2, image_meta, kernel_matrix_1_str, kernel_matrix_2_str


# --- Gradio Interface Setup ---

random_indices_for_selection = np.random.choice(df.index, size=5, replace=False) if df is not None else []
random_image_labels = [f"Random Image {i+1} (idx: {index})" for i, index in enumerate(random_indices_for_selection)]

if not random_image_labels:
    random_image_labels = ["No images loaded"]

# Dark blue CSS theme
custom_css = """
/* ── Global background & text ── */
.gradio-container, body, * {
    background-color: #071428 !important;
    color: #ffffff !important;
    font-family: 'Courier New', monospace !important;
}

/* ── Block / panel backgrounds ── */
.gr-block, .gr-box, .gr-panel,
.block, .panel, .form,
div.svelte-1gfkfd6,
.wrap.svelte-byatnx {
    background-color: #0d1f3c !important;
    border: 1px solid #2a5080 !important;
    border-radius: 6px !important;
    color: #ffffff !important;
}

/* ── All text elements explicitly white ── */
p, span, div, h1, h2, h3, h4, h5, h6,
label, legend, li, td, th, caption,
.label-wrap, .label-wrap span,
label span, span.svelte-1gfkfd6,
.prose p, .prose h1, .prose h2, .prose h3,
.gr-markdown, .gr-markdown * {
    color: #ffffff !important;
}

/* ── Labels ── */
label span, .label-wrap span, span.svelte-1gfkfd6 {
    color: #ffffff !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Markdown headings ── */
.gr-markdown h1 { color: #4fc3f7 !important; font-size: 1.5rem !important; }
.gr-markdown h3 { color: #7dd3fc !important; font-size: 1rem !important; letter-spacing: 0.1em !important; }
.gr-markdown p  { color: #ffffff !important; font-size: 0.85rem !important; }

/* ── Dropdowns ── */
.gr-dropdown select, select, select option {
    background-color: #0a1a33 !important;
    color: #ffffff !important;
    border: 1px solid #2a5080 !important;
    border-radius: 4px !important;
}

/* ── Radio buttons ── */
.gr-radio input[type=radio] { accent-color: #4fc3f7 !important; }
.gr-radio label { color: #ffffff !important; font-size: 0.85rem !important; }
input[type=radio] + span, input[type=radio] ~ span { color: #ffffff !important; }

/* ── Sliders ── */
input[type=range] { accent-color: #4fc3f7 !important; }
.gr-slider .wrap { background-color: transparent !important; }
.gr-slider span, .gr-slider label { color: #ffffff !important; }

/* ── Textboxes ── */
textarea, .gr-textbox textarea, input[type=text] {
    background-color: #091627 !important;
    color: #ffffff !important;
    border: 1px solid #2a5080 !important;
    border-radius: 4px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.85rem !important;
}

/* ── Plot panels ── */
.gr-plot, .plot-container {
    background-color: #0d1f3c !important;
    border: 1px solid #2a5080 !important;
    border-radius: 6px !important;
}

/* ── Divider / separator ── */
hr { border-color: #2a5080 !important; }

/* ── Number inputs next to sliders ── */
input[type=number] {
    background-color: #0a1a33 !important;
    color: #ffffff !important;
    border: 1px solid #2a5080 !important;
}

/* ── Section headers styling ── */
.section-header {
    color: #4fc3f7 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #2a5080;
    padding-bottom: 4px;
    margin-bottom: 8px;
}

/* ── Metadata box ── */
.meta-box {
    background-color: #091627 !important;
    border: 1px solid #2a5080 !important;
    border-radius: 6px !important;
    padding: 12px !important;
    color: #ffffff !important;
}

/* ── Catch-all for any remaining dark text ── */
[class*="svelte"] { color: #ffffff !important; }
[class*="prose"] { color: #ffffff !important; }
.markdown-body, .markdown-body * { color: #ffffff !important; }
"""

with gr.Blocks(title="Interactive Convolution", css=custom_css) as demo:

    gr.Markdown(
        """
        Interactive One-Stage Convolutional Network
        Select an image, configure two convolution setups independently, and visually compare outputs side-by-side.
        """
    )

    # ── ROW 1: Image Selector | Original Image | Image Metadata ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=2, min_width=220):
            gr.Markdown("### Image Selector")
            image_selector = gr.Radio(
                random_image_labels,
                label="Choose Input Image",
                value=random_image_labels[0] if random_image_labels else ""
            )

        with gr.Column(scale=2, min_width=260):
            original_plot = gr.Plot(show_label =False) #(label="Original Image", show_label=True)

        with gr.Column(scale=2, min_width=160):
            gr.Markdown("### Image Info")
            image_meta = gr.Markdown(
                "**Size:** —\n\n"
                "**Index:** —\n\n"
                "**Source:** MNIST",
                elem_classes=["meta-box"]
            )

    gr.Markdown("---")

    # ── ROW 2: Convolution Setup 1 | Convolution Setup 2 ──
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.Markdown("### Convolution Setup 1")
            kernel_type_1   = gr.Dropdown(kernel_types, label="Kernel Type", value='Identity')
            kernel_size_1   = gr.Radio([3, 5], label="Kernel Size (N×N)", value=3, interactive=True)
            kernel_matrix_display_1 = gr.Markdown(label="Kernel Matrix", value="```\nLoading...\n```") # New element for kernel matrix
            padding_size_1  = gr.Slider(0, 5, value=0, step=1, label="Padding")
            strides_1       = gr.Slider(1, 5, value=1, step=1, label="Stride")
            dilations_1     = gr.Slider(1, 5, value=1, step=1, label="Dilation")

        with gr.Column(scale=1):
            gr.Markdown("### Convolution Setup 2")
            kernel_type_2   = gr.Dropdown(kernel_types, label="Kernel Type", value='Edge Detection (Laplacian)')
            kernel_size_2   = gr.Radio([3, 5], label="Kernel Size (N×N)", value=3, interactive=True)
            kernel_matrix_display_2 = gr.Markdown(label="Kernel Matrix", value="```\nLoading...\n```") # New element for kernel matrix
            padding_size_2  = gr.Slider(0, 5, value=0, step=1, label="Padding")
            strides_2       = gr.Slider(1, 5, value=1, step=1, label="Stride")
            dilations_2     = gr.Slider(1, 5, value=1, step=1, label="Dilation")

    gr.Markdown("---")

    # ── ROW 3: Output Info 1 | Output Plot 1 | Output Plot 2 | Output Info 2 ──
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=160):
            gr.Markdown("### ⑤ Setup 1 Details")
            output_info_1 = gr.Textbox(
                label="Convolution Setup 1 Details",
                lines=4,
                interactive=False
            )

        with gr.Column(scale=2, min_width=240):
            gr.Markdown("### ④ Output 1")
            output_plot_1 = gr.Plot(label="Convolved Output 1", show_label=False)

        with gr.Column(scale=2, min_width=240):
            gr.Markdown("### ④ Output 2")
            output_plot_2 = gr.Plot(label="Convolved Output 2", show_label=False)

        with gr.Column(scale=1, min_width=160):
            gr.Markdown("### ⑥ Setup 2 Details")
            output_info_2 = gr.Textbox(
                label="Convolution Setup 2 Details",
                lines=4,
                interactive=False
            )

    # ── Event bindings ──
    all_inputs = [
        image_selector,
        kernel_type_1, kernel_size_1, dilations_1, strides_1, padding_size_1,
        kernel_type_2, kernel_size_2, dilations_2, strides_2, padding_size_2
    ]
    # Update all_outputs to include new kernel matrix displays
    all_outputs = [original_plot, output_plot_1, output_plot_2, output_info_1, output_info_2, image_meta, kernel_matrix_display_1, kernel_matrix_display_2]

    gr.on(
        [
            image_selector.change,
            kernel_type_1.change, kernel_size_1.change, dilations_1.change, strides_1.change, padding_size_1.change,
            kernel_type_2.change, kernel_size_2.change, dilations_2.change, strides_2.change, padding_size_2.change
        ],
        interactive_convolution,
        inputs=all_inputs,
        outputs=all_outputs
    )

    demo.load(
        interactive_convolution,
        inputs=all_inputs,
        outputs=all_outputs
    )

demo.launch(debug=True, share=True)
