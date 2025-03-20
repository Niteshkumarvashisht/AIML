# Python Basics - Variables and Data Types

# 1. Basic Variables
name = "Alice"        # String
age = 25             # Integer
height = 1.75        # Float
is_student = True    # Boolean

# 2. Print and String Formatting
print("Basic String Operations:")
print(f"Name: {name}")
print(f"Age: {age}")
print("Height: {:.2f}".format(height))

# 3. Basic Math Operations
print("\nMath Operations:")
x = 10
y = 3
print(f"Addition: {x + y}")
print(f"Subtraction: {x - y}")
print(f"Multiplication: {x * y}")
print(f"Division: {x / y}")
print(f"Integer Division: {x // y}")
print(f"Modulus: {x % y}")
print(f"Power: {x ** 2}")

# 4. String Operations
print("\nString Operations:")
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name  # Concatenation
print(f"Full name: {full_name}")
print(f"Uppercase: {full_name.upper()}")
print(f"Length of name: {len(full_name)}")

# 5. Type Conversion
print("\nType Conversion:")
number_str = "123"
number_int = int(number_str)  # String to Integer
print(f"String to Integer: {number_int + 7}")  # Will print 130

# Try running this file and experiment with the values!
