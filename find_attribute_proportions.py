import pandas as pd
import sys


filename=sys.argv[1]
df=pd.read_csv(filename)
df=df["label"]
attribute_count={}
attribute_proportion={}
length_df=len(df.index)

for row in df.values:
    # print(row)
    if row not in attribute_count:
        attribute_count[row]=1
    else:
        attribute_count[row]+=1

for x,v in attribute_count.items():
    if x not in attribute_proportion:
        attribute_proportion[x]= v/length_df

print("This is attribute count",attribute_count)
print("This is attribute proportion",attribute_proportion)

    
