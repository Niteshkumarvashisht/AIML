import matplotlib.pyplot as plt
import numpy as np

# Matplotlib Basics - Data Visualization Library
print("Creating various plots... Check the saved images.")

# 1. Line Plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)')
plt.title('Simple Line Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.savefig('line_plot.png')
plt.close()

# 2. Scatter Plot
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=colors, alpha=0.5)
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('scatter_plot.png')
plt.close()

# 3. Bar Plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.figure(figsize=(8, 6))
plt.bar(categories, values)
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.savefig('bar_plot.png')
plt.close()

# 4. Multiple Subplots
plt.figure(figsize=(12, 4))

# First subplot
plt.subplot(131)
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Plot 1')

# Second subplot
plt.subplot(132)
plt.scatter([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Plot 2')

# Third subplot
plt.subplot(133)
plt.bar([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Plot 3')

plt.tight_layout()
plt.savefig('subplots.png')
plt.close()

# 5. Pie Chart
sizes = [30, 20, 25, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.axis('equal')
plt.savefig('pie_chart.png')
plt.close()

print("All plots have been saved as PNG files in the current directory.")
print("Generated plots: line_plot.png, scatter_plot.png, bar_plot.png, subplots.png, pie_chart.png")
