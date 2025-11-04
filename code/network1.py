import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Define positions for two hospitals
hospital1_pos = (2, 3)
hospital2_pos = (4, 3)

# Draw hospital nodes (inner circles)
hospital1 = plt.Circle(hospital1_pos, 0.5, zorder=3)
hospital2 = plt.Circle(hospital2_pos, 0.5, zorder=3)
ax.add_patch(hospital1)
ax.add_patch(hospital2)

# Add hospital labels
ax.text(hospital1_pos[0], hospital1_pos[1], 'A', color='white', 
        fontsize=16, fontweight='bold', ha='center', va='center', zorder=4)
ax.text(hospital2_pos[0], hospital2_pos[1], 'B', color='white', 
        fontsize=16, fontweight='bold', ha='center', va='center', zorder=4)

# Add hospital names below
ax.text(hospital1_pos[0], hospital1_pos[1] - 1.8, 'Hospital A', 
        fontsize=12, ha='center', fontweight='bold')
ax.text(hospital2_pos[0], hospital2_pos[1] - 1.8, 'Hospital B', 
        fontsize=12, ha='center', fontweight='bold')

# Draw dual-facing arrows (bidirectional information exchange)
# Arrow from A to B (upper arrow)
arrow1 = FancyArrowPatch(
    (hospital1_pos[0] + 0.7, hospital1_pos[1] + 0.2),  # Start point
    (hospital2_pos[0] - 0.7, hospital2_pos[1] + 0.2),  # End point
    arrowstyle='->,head_width=4,head_length=4',
    color='black',
    linewidth=1,
    zorder=2
)
ax.add_patch(arrow1)

# Arrow from B to A (lower arrow)
arrow2 = FancyArrowPatch(
    (hospital2_pos[0] - 0.7, hospital2_pos[1] - 0.2),  # Start point
    (hospital1_pos[0] + 0.7, hospital1_pos[1] - 0.2),  # End point
    arrowstyle='->,head_width=4,head_length=4',
    color='black',
    linewidth=1,
    zorder=2
)
ax.add_patch(arrow2)


# Set axis properties
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.set_aspect('equal')
ax.axis('off')


plt.tight_layout()
plt.savefig('hospital_bidirectional_exchange.png', dpi=300, bbox_inches='tight')
plt.show()

print("Image saved as 'hospital_bidirectional_exchange.png'")