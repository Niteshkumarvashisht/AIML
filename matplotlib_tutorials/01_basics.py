import matplotlib.pyplot as plt
import numpy as np

print("Matplotlib Basics Tutorial")
print("========================")
print("Creating various basic plots... Check the saved images.")

# 1. Line Plot
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.title('Basic Line Plot')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.savefig('basic_line.png')
plt.close()

# 2. Scatter Plot
plt.figure(figsize=(8, 6))
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label='Color Value')
plt.savefig('scatter.png')
plt.close()

# 3. Bar Plot
plt.figure(figsize=(8, 6))
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 43]
plt.bar(categories, values, color='skyblue')
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
for i, v in enumerate(values):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.savefig('bar.png')
plt.close()

# 4. Histogram
plt.figure(figsize=(8, 6))
data = np.random.randn(1000)
plt.hist(data, bins=30, color='green', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('histogram.png')
plt.close()

# 5. Pie Chart
plt.figure(figsize=(8, 8))
sizes = [30, 20, 25, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
colors = ['red', 'green', 'blue', 'yellow', 'orange']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.axis('equal')
plt.savefig('pie.png')
plt.close()

# 6. Multiple Plots
plt.figure(figsize=(12, 4))
# First subplot
plt.subplot(131)
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Line')

# Second subplot
plt.subplot(132)
plt.scatter([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Scatter')

# Third subplot
plt.subplot(133)
plt.bar([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Bar')

plt.tight_layout()
plt.savefig('subplots.png')
plt.close()

# 7. Error Bars
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 10)
y = np.exp(-x/2)
error = 0.1 + 0.2 * np.random.rand(len(x))
plt.errorbar(x, y, yerr=error, fmt='o-', capsize=5)
plt.title('Error Bar Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('errorbar.png')
plt.close()

# 8. Fill Between
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.sin(x) + 0.5
plt.fill_between(x, y1, y2, alpha=0.3)
plt.plot(x, y1, 'b-', label='Lower bound')
plt.plot(x, y2, 'r-', label='Upper bound')
plt.title('Fill Between')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('fill_between.png')
plt.close()

print("\nGenerated plots:")
print("1. basic_line.png - Basic line plot")
print("2. scatter.png - Scatter plot with color mapping")
print("3. bar.png - Bar plot with value labels")
print("4. histogram.png - Histogram of random data")
print("5. pie.png - Pie chart with percentages")
print("6. subplots.png - Multiple plots in one figure")
print("7. errorbar.png - Error bar plot")
print("8. fill_between.png - Fill between two curves")
