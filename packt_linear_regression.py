import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(
    {
        "Area": [2600, 3000, 3200, 3600, 4000],
        "Price": [550000, 565000, 610000, 680000, 725000]
    }
)

plt.xlabel('Price(US$)')
plt.ylabel('Price(US$)')
plt.scatter(df.Area, df.Price, color='green', marker='+')
plt.show()

reg = LinearRegression()
reg.fit(df[['Area']], df.Price)
print(reg.predict([[3300]]))
print(reg.coef_)
print(reg.intercept_)