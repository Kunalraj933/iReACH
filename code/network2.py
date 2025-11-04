import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Define positions for two hospitals and ShCR hub
hospital1_pos = (2, 2)
hospital2_pos = (4, 2)
shcr_pos = (3, 5)  # ShCR positioned above and centered

# Draw hospital nodes (inner circles)
hospital1 = plt.Circle(hospital1_pos, 0.5, linewidth=2, zorder=3)
hospital2 = plt.Circle(hospital2_pos, 0.5, linewidth=2, zorder=3)
ax.add_patch(hospital1)
ax.add_patch(hospital2)

# Draw ShCR hub (square to distinguish it from hospitals)
shcr_square = mpatches.Rectangle((shcr_pos[0]-0.6, shcr_pos[1]-0.5), 1.2, 1, 
                                   linewidth=3, zorder=3)
ax.add_patch(shcr_square)

# Add hospital labels
ax.text(hospital1_pos[0], hospital1_pos[1], 'A', color='white', 
        fontsize=16, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(hospital2_pos[0], hospital2_pos[1], 'B', color='white', 
        fontsize=16, fontweight='bold', ha='center', va='center', zorder=4)

# Add ShCR label
ax.text(shcr_pos[0], shcr_pos[1], 'ShCR', color='white', 
        fontsize=14, fontweight='bold', ha='center', va='center', zorder=4)

# Draw arrows from hospitals to ShCR (hospitals write to ShCR)
# Hospital A to ShCR
arrow_a_to_shcr = FancyArrowPatch(
    (hospital1_pos[0] + 0.2, hospital1_pos[1] + 0.5),  # Start from hospital A
    (shcr_pos[0] - 0.4, shcr_pos[1] - 0.5),  # End at ShCR
    arrowstyle='->,head_width=4,head_length=4',
    color='black',
    linewidth=1,
    zorder=2
)
ax.add_patch(arrow_a_to_shcr)

# Hospital B to ShCR
arrow_b_to_shcr = FancyArrowPatch(
    (hospital2_pos[0] - 0.2, hospital2_pos[1] + 0.5),  # Start from hospital B
    (shcr_pos[0] + 0.4, shcr_pos[1] - 0.5),  # End at ShCR
    arrowstyle='->,head_width=4,head_length=4',
    color='black',
    linewidth=1,
    zorder=2
)
ax.add_patch(arrow_b_to_shcr)



# Set axis properties
ax.set_xlim(0, 6)
ax.set_ylim(0, 7.5)
ax.set_aspect('equal')
ax.axis('off')


plt.tight_layout()
plt.savefig('hospital_shcr_network.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("Image saved as 'hospital_shcr_network.png'")