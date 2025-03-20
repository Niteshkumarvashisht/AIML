import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Ellipse
from matplotlib.collections import PatchCollection

print("Matplotlib Intermediate Tutorial")
print("==============================")
print("Creating various intermediate plots... Check the saved images.")

# 1. Advanced Line Styles and Colors
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), 'r--', label='sin(x)', linewidth=2)
plt.plot(x, np.cos(x), 'b-.', label='cos(x)', linewidth=2)
plt.plot(x, -np.sin(x), 'g:', label='-sin(x)', linewidth=2)
plt.title('Advanced Line Styles')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('advanced_lines.png')
plt.close()

# 2. Custom Markers
plt.figure(figsize=(10, 6))
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', '+', 'x', 'D']
for i, marker in enumerate(markers):
    plt.plot([i], [0], marker=marker, markersize=15, label=f'Marker {marker}')
plt.title('Custom Markers')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.savefig('markers.png', bbox_inches='tight')
plt.close()

# 3. Contour Plots
plt.figure(figsize=(10, 6))
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

plt.contour(X, Y, Z, levels=15, cmap='viridis')
plt.colorbar(label='Z Value')
plt.title('Contour Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('contour.png')
plt.close()

# 4. 3D Surface Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
plt.colorbar(surf)
ax.set_title('3D Surface Plot')
plt.savefig('surface3d.png')
plt.close()

# 5. Custom Shapes and Patches
fig, ax = plt.subplots(figsize=(10, 6))
# Create some shapes
shapes = [
    Circle((0.2, 0.5), 0.1),
    Rectangle((0.4, 0.4), 0.2, 0.2),
    Ellipse((0.8, 0.5), 0.2, 0.1)
]
colors = np.linspace(0, 1, len(shapes))
p = PatchCollection(shapes, cmap='viridis', alpha=0.6)
p.set_array(colors)
ax.add_collection(p)
plt.colorbar(p)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.title('Custom Shapes')
plt.savefig('shapes.png')
plt.close()

# 6. Advanced Subplots
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3)

# Plot 1: Line plot
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(np.random.rand(10))
ax1.set_title('Line Plot')

# Plot 2: Scatter plot
ax2 = fig.add_subplot(gs[0, 1:])
ax2.scatter(np.random.rand(10), np.random.rand(10))
ax2.set_title('Scatter Plot')

# Plot 3: Bar plot
ax3 = fig.add_subplot(gs[1, :2])
ax3.bar(['A', 'B', 'C'], [1, 2, 3])
ax3.set_title('Bar Plot')

# Plot 4: Pie chart
ax4 = fig.add_subplot(gs[1, 2])
ax4.pie([1, 2, 3], labels=['X', 'Y', 'Z'])
ax4.set_title('Pie Chart')

plt.tight_layout()
plt.savefig('advanced_subplots.png')
plt.close()

# 7. Custom Colormaps
plt.figure(figsize=(10, 6))
data = np.random.rand(10, 10)
plt.imshow(data, cmap='viridis')
plt.colorbar(label='Value')
plt.title('Custom Colormap')
plt.savefig('colormap.png')
plt.close()

# 8. Text and Annotations
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.annotate('Maximum', xy=(4.7, 1), xytext=(5.5, 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.title('Text and Annotations')
plt.text(2, -0.5, 'sin(x) function', fontsize=12)
plt.savefig('annotations.png')
plt.close()

print("\nGenerated plots:")
print("1. advanced_lines.png - Different line styles and colors")
print("2. markers.png - Various marker types")
print("3. contour.png - Contour plot with colorbar")
print("4. surface3d.png - 3D surface plot")
print("5. shapes.png - Custom shapes and patches")
print("6. advanced_subplots.png - Complex subplot arrangement")
print("7. colormap.png - Custom colormap visualization")
print("8. annotations.png - Text and arrow annotations")
