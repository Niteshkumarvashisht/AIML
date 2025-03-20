# Control Flow in Python

# 1. If-Elif-Else Statements
print("If-Elif-Else Example:")
score = 85

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
else:
    print("Grade: D")

# 2. For Loops
print("\nFor Loop Examples:")
# Loop through a range
print("Counting:")
for i in range(1, 5):
    print(i)

# Loop through a list
print("\nLooping through list:")
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(f"I like {fruit}")

# 3. While Loops
print("\nWhile Loop Example:")
count = 0
while count < 3:
    print(f"Count is: {count}")
    count += 1

# 4. Break and Continue
print("\nBreak and Continue Example:")
for i in range(5):
    if i == 2:
        continue  # Skip 2
    if i == 4:
        break    # Stop at 4
    print(f"Number: {i}")

# 5. Try-Except (Error Handling)
print("\nError Handling Example:")
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Cannot divide by zero!")
finally:
    print("This always runs")
