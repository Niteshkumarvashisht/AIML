import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap

print("Matplotlib Advanced Tutorial")
print("==========================")
print("Creating advanced plots... Check the saved images and animations.")

# 1. Custom Projection
from matplotlib.projections import register_projection
from mpl_toolkits.mplot3d import Axes3D

class CustomProjection(Axes3D):
    name = 'custom_3d'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view_init(elev=20, azim=45)

register_projection(CustomProjection)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='custom_3d')
x = y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.savefig('custom_projection.png')
plt.close()

# 2. Custom Transform
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
base = plt.gca().transData
rot = transforms.Affine2D().rotate_deg(45)
trans_data = base + rot
plt.plot(x, y, transform=trans_data)
plt.title('Custom Transform')
plt.savefig('custom_transform.png')
plt.close()

# 3. Custom Path
fig, ax = plt.subplots(figsize=(10, 6))
vertices = [(0., 0.), (1., 1.), (2., 0.), (1., -1.), (0., 0.)]
codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
path = Path(vertices, codes)
patch = PathPatch(path, facecolor='orange', alpha=0.5)
ax.add_patch(patch)
ax.set_xlim(-1, 3)
ax.set_ylim(-2, 2)
plt.title('Custom Path')
plt.savefig('custom_path.png')
plt.close()

# 4. Custom Animation
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10.0))
    return line,

ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
ani.save('animation.gif', writer='pillow')
plt.close()

# 5. Custom Colormap
colors = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
n_bins = 100
cmap_name = 'custom_cmap'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

fig, ax = plt.subplots(figsize=(10, 6))
data = np.random.rand(10, 10)
plt.imshow(data, cmap=custom_cmap)
plt.colorbar(label='Custom Colors')
plt.title('Custom Colormap')
plt.savefig('custom_colormap.png')
plt.close()

# 6. Advanced 3D Visualization
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Generate data for a mobius strip
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(-1, 1, 50)
U, V = np.meshgrid(u, v)

x = (1 + 0.5*V*np.cos(U/2.0))*np.cos(U)
y = (1 + 0.5*V*np.cos(U/2.0))*np.sin(U)
z = 0.5*V*np.sin(U/2.0)

ax.plot_surface(x, y, z, cmap='viridis')
plt.title('Mobius Strip')
plt.savefig('mobius.png')
plt.close()

# 7. Custom Event Handling
fig, ax = plt.subplots(figsize=(10, 6))
points, = ax.plot([], [], 'ro')
text = ax.text(0.5, 0.5, '', transform=ax.transAxes)

x_data, y_data = [], []

def onclick(event):
    if event.inaxes == ax:
        x_data.append(event.xdata)
        y_data.append(event.ydata)
        points.set_data(x_data, y_data)
        text.set_text(f'Points: {len(x_data)}')
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
plt.title('Click to Add Points')
plt.savefig('interactive.png')
plt.close()

# 8. Custom Styling
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), 'r-', label='sin(x)')
plt.plot(x, np.cos(x), 'b-', label='cos(x)')
plt.title('Custom Style')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig('custom_style.png')
plt.close()

print("\nGenerated files:")
print("1. custom_projection.png - Custom 3D projection")
print("2. custom_transform.png - Transformed plot")
print("3. custom_path.png - Custom path and patches")
print("4. animation.gif - Animated sine wave")
print("5. custom_colormap.png - Custom colormap")
print("6. mobius.png - 3D Mobius strip")
print("7. interactive.png - Interactive plot template")
print("8. custom_style.png - Custom styled plot")
