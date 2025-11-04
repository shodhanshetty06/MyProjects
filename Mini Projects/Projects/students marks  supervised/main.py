
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report


# Load Dataset

df = pd.read_excel("Dataset\student_marks_with_internals.xlsx")
print(" Dataset Loaded Successfully!")
print(df.head())


# Handle Missing Values (Regression Imputation)


# Fill missing Internal_1_Marks using regression based on Internal_2 and Semester
df_int1 = df.dropna(subset=["Internal_1_Marks", "Internal_2_Marks", "Semester_Marks"])
reg1 = LinearRegression()
reg1.fit(df_int1[["Internal_2_Marks", "Semester_Marks"]], df_int1["Internal_1_Marks"])

missing_int1 = df[df["Internal_1_Marks"].isna()]
if not missing_int1.empty:
    df.loc[df["Internal_1_Marks"].isna(), "Internal_1_Marks"] = reg1.predict(
        missing_int1[["Internal_2_Marks", "Semester_Marks"]].fillna(0)
    )

# Fill missing Internal_2_Marks using regression based on Internal_1 and Semester
df_int2 = df.dropna(subset=["Internal_2_Marks", "Internal_1_Marks", "Semester_Marks"])
reg2 = LinearRegression()
reg2.fit(df_int2[["Internal_1_Marks", "Semester_Marks"]], df_int2["Internal_2_Marks"])

missing_int2 = df[df["Internal_2_Marks"].isna()]
if not missing_int2.empty:
    df.loc[df["Internal_2_Marks"].isna(), "Internal_2_Marks"] = reg2.predict(
        missing_int2[["Internal_1_Marks", "Semester_Marks"]].fillna(0)
    )

# Fill missing Semester marks using regression based on internals
df_sem = df.dropna(subset=["Semester_Marks", "Internal_1_Marks", "Internal_2_Marks"])
reg3 = LinearRegression()
reg3.fit(df_sem[["Internal_1_Marks", "Internal_2_Marks"]], df_sem["Semester_Marks"])

missing_sem = df[df["Semester_Marks"].isna()]
if not missing_sem.empty:
    df.loc[df["Semester_Marks"].isna(), "Semester_Marks"] = reg3.predict(
        missing_sem[["Internal_1_Marks", "Internal_2_Marks"]].fillna(0)
    )

print("\n Missing values handled successfully!")


#  Calculate Total Percentage

df["Total_Percentage"] = ((df["Internal_1_Marks"] + df["Internal_2_Marks"]) / 2) * 0.4 + (df["Semester_Marks"] * 0.6)

# Classification (Pass/Fail)

df["Result"] = df["Total_Percentage"].apply(lambda x: "Pass" if x >= 45 else "Fail")
df["Result_Label"] = (df["Result"] == "Pass").astype(int)


#  Train Classification Model

X = df[["Internal_1_Marks", "Internal_2_Marks", "Semester_Marks"]]
y = df["Result_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n Classification Accuracy: {acc*100:.2f}%")

print("\n Classification Report:\n", classification_report(y_test, y_pred))

# Regression Evaluation

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, df["Total_Percentage"], test_size=0.2, random_state=42)
reg_model = LinearRegression()
reg_model.fit(Xr_train, yr_train)
y_reg_pred = reg_model.predict(Xr_test)

mae = mean_absolute_error(yr_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(yr_test, y_reg_pred))
print(f"\n Regression MAE: {mae:.2f}")
print(f" Regression RMSE: {rmse:.2f}")

# Final Output Preview

print("\n Final Data Preview:")
print(df.head(10))


