# Object-Oriented Programming in Python

# 1. Basic Class Definition
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def info(self):
        return f"{self.name} is {self.age} years old"

# 2. Inheritance
class Animal:
    def __init__(self, species):
        self.species = species
    
    def make_sound(self):
        return "Some sound"

class Cat(Animal):
    def __init__(self, name):
        super().__init__("cat")
        self.name = name
    
    def make_sound(self):
        return "Meow!"

# 3. Class with Properties
class BankAccount:
    def __init__(self):
        self._balance = 0
    
    @property
    def balance(self):
        return self._balance
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            return f"Deposited ${amount}"
        return "Invalid amount"

# Testing the classes
print("Dog Class Example:")
my_dog = Dog("Rex", 3)
print(my_dog.bark())
print(my_dog.info())

print("\nCat Class Example:")
my_cat = Cat("Whiskers")
print(f"{my_cat.name} is a {my_cat.species}")
print(f"{my_cat.name} says: {my_cat.make_sound()}")

print("\nBank Account Example:")
account = BankAccount()
print(f"Initial balance: ${account.balance}")
print(account.deposit(100))
print(f"New balance: ${account.balance}")
