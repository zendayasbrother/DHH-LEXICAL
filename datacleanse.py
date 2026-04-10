import numpy as np 
import pandas as pd

df = pd.read_csv('General Lexicon DHH Annotations.tsv') 

print("Column names")
print(df.columns.tolist())

# display first 15 rows 
print("\n")
print(df.head(15))

# display dimensions of dataframe 
print("\n Dimensions of dataframe")
print(df.shape)  

# display column names 
print("\n Column names")
print(list(df)) 

# display count of missing values#
print("\n Missing values count: ")
print(df.isnull().sum().reset_index(name = 'Missing Values Counted')) 

for col in df.columns:
  print(col, df[col].nunique(), len(df))
  

df2 = pd.read_csv('General Lexicon Linguistic Characteristics.tsv')

print("\n Column names")
print(df2.columns.tolist())

# display first 15 rows 
print("\n")
print(df.head(15))

# display dimensions of dataframe 
print("\n Dimensions of dataframe") 
print(df2.shape)

# display column names
print("\n Column names")
print(list(df2))

# display count of missing values#
print("\n Missing values count: ")
print(df2.isnull().sum().reset_index(name = 'Missing Values Counted'))

for col in df2.columns:
  print(col, df2[col].nunique(), len(df2))
  


