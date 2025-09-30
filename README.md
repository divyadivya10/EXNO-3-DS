## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="328" height="416" alt="image" src="https://github.com/user-attachments/assets/c0881d1f-cae0-418d-bec9-d4ee6f75d2d5" />
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="153" height="216" alt="image" src="https://github.com/user-attachments/assets/71e82b7b-5e2a-4a7e-8e7d-a3f9b86e1233" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

<img width="364" height="423" alt="image" src="https://github.com/user-attachments/assets/2c16b3ba-8674-470b-95a4-cff7ef885ec9" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="364" height="429" alt="image" src="https://github.com/user-attachments/assets/fee6bea9-f093-40c2-9d95-eadbf480e995" />
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="479" height="439" alt="image" src="https://github.com/user-attachments/assets/b84a6dc6-a4af-4b20-9ee5-f36630cd01c3" />
```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="736" height="425" alt="image" src="https://github.com/user-attachments/assets/9ccc75ee-113f-4839-8659-96aab4958b79" />

```
pip install --upgrade category_encoders
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="750" height="428" alt="image" src="https://github.com/user-attachments/assets/e9fb72a2-ef85-47f8-bd82-7063645564c1" />
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="605" height="420" alt="image" src="https://github.com/user-attachments/assets/eff5143d-f9cb-4e39-b0b6-3c1f86392d6a" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="822" height="525" alt="image" src="https://github.com/user-attachments/assets/68ceefe9-4669-4c8b-bb77-0bbe8e330aeb" />
```
df.skew()
```
<img width="365" height="110" alt="image" src="https://github.com/user-attachments/assets/1a2f2e2c-d051-420f-953b-c1d2cf3d798a" />
```
np.log(df["Highly Positive Skew"])
```
<img width="547" height="265" alt="Screenshot 2025-09-30 105316" src="https://github.com/user-attachments/assets/88699b50-7fc8-437f-984a-3736057722cb" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="572" height="265" alt="image" src="https://github.com/user-attachments/assets/cd7dedb6-99e2-4d10-96ea-5cbee40475b7" />
```
np.sqrt(df["Highly Positive Skew"])
```
<img width="535" height="266" alt="image" src="https://github.com/user-attachments/assets/d04bef5e-6144-4abd-a99b-016c1e0b13f9" />
```
np.square(df["Highly Positive Skew"])
```
<img width="563" height="273" alt="image" src="https://github.com/user-attachments/assets/a74b3f09-4b54-4b6f-b362-e09c8e47fcff" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="834" height="535" alt="image" src="https://github.com/user-attachments/assets/b931427a-5ab7-4743-8578-52515194016c" />
```
df.skew()
```
<img width="407" height="150" alt="image" src="https://github.com/user-attachments/assets/b12b8726-80a8-4702-b4fc-f80c7a644b52" />
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="449" height="152" alt="image" src="https://github.com/user-attachments/assets/2430ffd2-55cd-4348-b1d1-7532f9d0151f" />
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="840" height="550" alt="image" src="https://github.com/user-attachments/assets/ff83b48e-21e4-49e3-866d-f03834c13043" />
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="791" height="540" alt="image" src="https://github.com/user-attachments/assets/a724e6d6-0d4f-40c8-b83c-f68b376b20d4" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="790" height="550" alt="image" src="https://github.com/user-attachments/assets/5a744439-7387-49a2-b75a-7139f47f8028" />
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="743" height="536" alt="image" src="https://github.com/user-attachments/assets/a0c7c206-af68-4dc1-a5a2-6c67023f9875" />

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
<img width="736" height="541" alt="image" src="https://github.com/user-attachments/assets/0929bc9d-7476-4347-8542-b38764e2d950" />
```
dt=pd.read_csv("titanic_dataset.csv")
dt
```
<img width="1307" height="505" alt="image" src="https://github.com/user-attachments/assets/a9d94ad2-e05c-4a1f-8ca5-856ae2dbe933" />
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="751" height="553" alt="image" src="https://github.com/user-attachments/assets/d534f9a9-06d6-4829-b707-681f776f4cb7" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="762" height="535" alt="image" src="https://github.com/user-attachments/assets/80e6b7af-88d3-4931-a1f1-a25e344c4fa6" />




















# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.
       # INCLUDE YOUR RESULT HERE

       
