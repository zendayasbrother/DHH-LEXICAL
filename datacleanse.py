import numpy as np 
import pandas as pd

class DataCleaner:
    def __init__(self, ds, ds2):
        if isinstance(ds, ds2, pd.DataFrame):
            self.df = ds.merge(ds2, how='right') # full join like merge
            print("DataFrame merged successfully.")
        else:
            df1 = pd.DataFrame(ds)
            df2 = pd.DataFrame(ds2) 
            self.df = df1.merge(df2, how='right')
            print("DataFrame merged successfully.")
    
    def clean_data(self):
      
      # display dimensions of dataframe 
      print("\n Dimensions of dataframe")
      print(self.df.shape) 
      
      print("\nColumn names") 
      print(self.df.columns.tolist())
      
      # display first 15 rows 
      print("\n")
      print(self.df.head(15)) 
      
      print(self.df.info()) 
      
      # display count of missing values#
      print("\n Missing values count: ")
      print(self.df.isnull().sum().reset_index(name = 'Missing Values Counted')) # process and clean individually before merging / concatenating

  
