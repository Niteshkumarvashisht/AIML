import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype

print("Pandas Advanced Tutorial")
print("=====================")

# 1. Custom Aggregation Functions
print("\n1. Custom Aggregation Functions:")
# Create sample data
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': ['a', 'a', 'b', 'b', 'c']
})

# Define custom aggregation function
def custom_agg(x):
    return pd.Series({
        'mean': x.mean(),
        'std': x.std(),
        'custom': (x.max() - x.min()) / x.mean()
    })

print("Custom aggregation:")
print(df.groupby('C').agg(custom_agg))

# 2. Window Functions
print("\n2. Window Functions:")
# Create time series data
dates = pd.date_range('20230101', periods=10)
df2 = pd.DataFrame({
    'date': dates,
    'value': np.random.randn(10)
})

print("\nRolling mean:")
print(df2.set_index('date')['value'].rolling(window=3).mean())

print("\nExpanding mean:")
print(df2.set_index('date')['value'].expanding().mean())

# 3. Advanced MultiIndex Operations
print("\n3. Advanced MultiIndex Operations:")
# Create multi-index DataFrame
arrays = [
    ['A', 'A', 'B', 'B'],
    [1, 2, 1, 2]
]
df3 = pd.DataFrame(np.random.randn(4, 2),
                  index=pd.MultiIndex.from_arrays(arrays, names=['first', 'second']),
                  columns=['col1', 'col2'])
print("\nMulti-index DataFrame:")
print(df3)

print("\nStack operation:")
print(df3.stack())

print("\nUnstack operation:")
print(df3.unstack())

# 4. Advanced Merge and Concatenation
print("\n4. Advanced Merge and Concatenation:")
# Create sample DataFrames
df4 = pd.DataFrame({
    'key1': ['A', 'B', 'C'],
    'key2': [1, 2, 3],
    'value1': [100, 200, 300]
})

df5 = pd.DataFrame({
    'key1': ['A', 'B', 'D'],
    'key2': [1, 2, 4],
    'value2': [400, 500, 600]
})

print("\nComplex merge:")
result = pd.merge(df4, df5, 
                 on=['key1', 'key2'], 
                 how='outer',
                 indicator=True)
print(result)

# 5. Custom Data Types
print("\n5. Custom Data Types:")
# Create categorical type with custom ordering
cat_type = CategoricalDtype(
    categories=['low', 'medium', 'high'],
    ordered=True
)

s = pd.Series(['medium', 'high', 'low', 'medium'],
             dtype=cat_type)
print("\nOrdered categorical:")
print(s)
print("\nComparison:")
print(s > 'low')

# 6. Advanced Time Series
print("\n6. Advanced Time Series:")
# Create business day calendar
bday = pd.offsets.BusinessDay()
dates = pd.date_range('20230101', periods=10, freq=bday)
ts = pd.Series(np.random.randn(10), index=dates)

print("\nBusiness day time series:")
print(ts)

print("\nLagged correlation:")
print(ts.corr(ts.shift(1)))

# 7. Memory Optimization
print("\n7. Memory Optimization:")
# Create large DataFrame
df_large = pd.DataFrame({
    'A': np.random.choice(['foo', 'bar', 'baz'], 1000),
    'B': np.random.randint(0, 100, 1000),
    'C': np.random.randn(1000)
})

print("\nOriginal memory usage:")
print(df_large.memory_usage(deep=True))

# Optimize memory
df_large['A'] = df_large['A'].astype('category')
df_large['B'] = df_large['B'].astype('int8')

print("\nOptimized memory usage:")
print(df_large.memory_usage(deep=True))

# 8. Custom Indexers
print("\n8. Custom Indexers:")
# Create custom indexer
class CustomIndexer:
    def __init__(self, window_size):
        self.window_size = window_size
    
    def __call__(self, data):
        return np.convolve(data, 
                          np.ones(self.window_size)/self.window_size, 
                          mode='valid')

# Apply custom indexer
s = pd.Series(np.random.randn(100))
custom_roll = s.rolling(window=5, center=True).apply(CustomIndexer(5))
print("\nCustom rolling calculation:")
print(custom_roll.head())

# 9. Advanced String Operations
print("\n9. Advanced String Operations:")
# Create DataFrame with complex strings
df_str = pd.DataFrame({
    'text': ['Hello, World!', 'Python_123', 'Data Science'],
    'pattern': [r'\w+', r'\d+', r'[A-Za-z]+']
})

print("\nExtract with regex:")
print(df_str.apply(lambda x: pd.Series(x['text']).str.extract(x['pattern'])))

# 10. Performance Optimization
print("\n10. Performance Optimization:")
# Compare apply vs vectorized operations
df_perf = pd.DataFrame(np.random.randn(10000, 2), columns=['A', 'B'])

def slow_function(x):
    return np.sqrt((x ** 2).sum())

import time

# Time apply method
start = time.time()
result1 = df_perf.apply(slow_function)
print(f"\nApply method time: {time.time() - start:.4f} seconds")

# Time vectorized operation
start = time.time()
result2 = np.sqrt((df_perf ** 2).sum())
print(f"Vectorized operation time: {time.time() - start:.4f} seconds")
