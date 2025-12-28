import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "t": [1,2,3,4,5],
    "value": [100,105,110,120,130]
})

X = df[["t"]]
y = df["value"]

model = LinearRegression()
model.fit(X, y)

print("Next prediction:", model.predict([[6]])[0])
