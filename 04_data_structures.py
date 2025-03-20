# Data Structures in Python

# 1. Lists
print("Lists:")
fruits = ["apple", "banana", "orange"]
print(f"Original list: {fruits}")
fruits.append("grape")              # Add item
print(f"After append: {fruits}")
fruits.pop()                        # Remove last item
print(f"After pop: {fruits}")
print(f"First fruit: {fruits[0]}")  # Indexing

# 2. Dictionaries
print("\nDictionaries:")
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}
print(f"Person: {person}")
print(f"Name: {person['name']}")
person["email"] = "john@example.com"  # Add new key-value
print(f"After adding email: {person}")

# 3. Sets (unique items)
print("\nSets:")
numbers = {1, 2, 2, 3, 3, 4}  # Duplicates are removed
print(f"Set of numbers: {numbers}")
numbers.add(5)
print(f"After adding 5: {numbers}")

# 4. Tuples (immutable)
print("\nTuples:")
coordinates = (10, 20)
print(f"Coordinates: {coordinates}")
x, y = coordinates  # Tuple unpacking
print(f"X: {x}, Y: {y}")

# 5. List Comprehension
print("\nList Comprehension:")
squares = [x**2 for x in range(5)]
print(f"Squares: {squares}")

# 6. Dictionary Comprehension
print("\nDictionary Comprehension:")
square_dict = {x: x**2 for x in range(5)}
print(f"Square dictionary: {square_dict}")

# 7. Nested Data Structures
print("\nNested Structures:")
users = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30}
]
print(f"Users: {users}")
print(f"First user's name: {users[0]['name']}")
