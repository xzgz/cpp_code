import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def draw_vector(ax, y_pos, elements, colors, width=0.8, height=0.5, x_start=0):
    x = x_start
    for i, (num, color) in enumerate(zip(elements, colors)):
        rect = Rectangle((x, y_pos), width, height, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + width/2, y_pos + height/2, str(num), ha='center', va='center', fontsize=9)
        x += width + 0.1

def draw_arrows(ax, src_y, dst_y, src_elements, dst_elements, x_start=0, arrow_color='gray'):
    src_x = x_start + 0.4
    dst_x = x_start + 0.4
    dx = 0.8 + 0.1
    for i in range(len(src_elements)):
        for j in range(len(dst_elements)):
            ax.annotate('', xy=(dst_x + j*dx, dst_y), xytext=(src_x + i*dx, src_y),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=0.5))

fig, axs = plt.subplots(7, 1, figsize=(16, 12), gridspec_kw={'hspace': 0.5})

instructions = [
    {
        'label': 'Instruction 1',
        'v0': [1]*32,
        'v1': [2]*32,
        'v0_color': ['yellow']*32,
        'v1_color': ['yellow']*32,
        'arrow_color': 'gray'
    },
    {
        'label': 'Instruction 2',
        'v0': [1]*32,
        'v1': list(range(1,33)),
        'v0_color': ['yellow']*32,
        'v1_color': ['green']*32,
        'arrow_color': 'gray'
    },
    {
        'label': 'Instruction 3',
        'v0': [1]*32,
        'v1': [4]*32,
        'v0_color': ['yellow']*32,
        'v1_color': ['orange']*32,
        'arrow_color': 'gray'
    },
    {
        'label': 'Instruction 4',
        'v0': [1]*32,
        'v1': list(range(1,33)),
        'v0_color': ['yellow']*32,
        'v1_color': ['green']*32,
        'arrow_color': 'gray'
    },
    {
        'label': 'Instruction 5',
        'v0': [1]*32,
        'v1': list(range(1,33)),
        'v0_color': ['yellow']*32,
        'v1_color': ['green']*32,
        'arrow_color': 'gray'
    },
    {
        'label': 'Instruction 6',
        'v0': list(range(1,33)),
        'v1': list(range(1,33)),
        'v0_color': ['green']*32,
        'v1_color': ['yellow']*32,
        'arrow_color': 'black'
    },
    {
        'label': 'Instruction 7',
        'v0': list(range(1,33)),
        'v1': list(range(1,33)),
        'v0_color': ['green']*32,
        'v1_color': ['yellow']*32,
        'arrow_color': 'black'
    }
]

for i, ax in enumerate(axs):
    inst = instructions[i]
    
    # Draw instruction label
    ax.text(-0.5, 1.2, inst['label'], ha='left', va='center', fontsize=10, bbox={'facecolor': 'lightgray', 'edgecolor': 'black'})
    
    # Draw v0 vector
    draw_vector(ax, 1.0, inst['v0'], inst['v0_color'], x_start=0)
    ax.text(-0.5, 1.0, 'v0', ha='right', va='center', fontsize=10)
    
    # Draw v1 vector
    draw_vector(ax, 0.0, inst['v1'], inst['v1_color'], x_start=0)
    ax.text(-0.5, 0.0, 'v1', ha='right', va='center', fontsize=10)
    
    # Draw arrows
    draw_arrows(ax, 1.0, 0.0, inst['v0'], inst['v1'], arrow_color=inst['arrow_color'])
    
    # Formatting
    ax.set_xlim(-1, 32*(0.8+0.1)+0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

plt.tight_layout()
plt.savefig('vector_operations.png', dpi=300, bbox_inches='tight')
