# Functions in Python

# 1. Basic Function
def greet(name):
    """Simple function that returns a greeting"""
    return f"Hello, {name}!"

# 2. Function with Multiple Parameters
def calculate_rectangle_area(length, width):
    """Calculate the area of a rectangle"""
    return length * width

# 3. Function with Default Parameters
def power(base, exponent=2):
    """Calculate power with default exponent of 2"""
    return base ** exponent

# 4. Function with Multiple Returns
def get_min_max(numbers):
    """Return both minimum and maximum from a list"""
    return min(numbers), max(numbers)

# 5. Function with *args (Variable Arguments)
def sum_all(*numbers):
    """Sum any number of arguments"""
    return sum(numbers)

# Testing the functions
print("Basic Function:")
print(greet("Alice"))

print("\nRectangle Area:")
print(f"Area of 5x3 rectangle: {calculate_rectangle_area(5, 3)}")

print("\nPower Function:")
print(f"2² = {power(2)}")  # Using default exponent
print(f"2³ = {power(2, 3)}")  # Specifying exponent

print("\nMin and Max:")
numbers = [1, 5, 2, 8, 3]
min_num, max_num = get_min_max(numbers)
print(f"Min: {min_num}, Max: {max_num}")

print("\nSum All:")
print(f"Sum of 1,2,3,4: {sum_all(1,2,3,4)}")
