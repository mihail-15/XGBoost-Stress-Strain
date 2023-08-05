# import the necessary packages
import pandas as pd
import contextlib
import numpy as np
from xgboost import XGBRegressor # import XGBRegressor 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import patheffects
import seaborn as sns




# load the data from the 'xlsx' files (for example "C_0.01_stress-strain_data.xlsx")
data = pd.read_excel("C_0.01_stress-strain_data.xlsx")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

data1 = pd.read_excel("SC_0.01_stress-strain_data.xlsx")
X1 = data1.iloc[:, :-1].values
y1 = data1.iloc[:, -1].values

data2 = pd.read_excel("C_0.001_stress-strain_data.xlsx")
X2 = data2.iloc[:, :-1].values
y2 = data2.iloc[:, -1].values

data3 = pd.read_excel("SC_0.001_stress-strain_data.xlsx")
X3 = data3.iloc[:, :-1].values
y3 = data3.iloc[:, -1].values

# split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train, test_size=0.25, random_state=42)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.25, random_state=42)

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
X3_train, X3_val, y3_train, y3_val = train_test_split(X3_train, y3_train, test_size=0.25, random_state=42)

# define a function to perform hyperparameter tuning using GridSearchCV
def xgb_gridsearch(X_train, y_train, X_val, y_val):
 xgb = XGBRegressor(random_state=42) # use XGBRegressor 
 param_grid = {
 'eta': [0.1, 0.2, 0.4, 0.6], # use eta instead of max_features
 'n_estimators': [10, 20, 50, 100]
 }
 gs = GridSearchCV(xgb, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
 gs.fit(X_train, y_train)
 xgb_best = gs.best_estimator_
 y_val_pred = xgb_best.predict(X_val)
 r2 = r2_score(y_val, y_val_pred)
 rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
 mse = mean_squared_error(y_val, y_val_pred)
 mae = mean_absolute_error(y_val, y_val_pred)
 print("Best parameters: ", gs.best_params_)
 print("R2 score: {:.5f}".format(r2))
 print("RMSE: {:.5f}".format(rmse))
 print("MSE: {:.5f}".format(mse))
 print("MAE: {:.5f}".format(mae))
 return xgb_best

# perform hyperparameter tuning on the training and validation sets for each dataset
xgb_best = xgb_gridsearch(X_train, y_train, X_val, y_val)
xgb_best1 = xgb_gridsearch(X1_train, y1_train, X1_val, y1_val)
xgb_best2 = xgb_gridsearch(X2_train, y2_train, X2_val, y2_val)
xgb_best3 = xgb_gridsearch(X3_train, y3_train, X3_val, y3_val)

# train the XGB model on the entire training set using the best hyperparameters for each dataset
xgb = XGBRegressor(learning_rate=xgb_best.learning_rate,n_estimators=xgb_best.n_estimators ,random_state=42) # use learning_rate instead of eta
xgb.fit(X_train,y_train)

xgb1 = XGBRegressor(learning_rate=xgb_best1.learning_rate,n_estimators=xgb_best1.n_estimators ,random_state=42) # use learning_rate instead of eta
xgb1.fit(X1_train,y1_train)

xgb2 = XGBRegressor(learning_rate=xgb_best2.learning_rate,n_estimators=xgb_best2.n_estimators ,random_state=42) # use learning_rate instead of eta
xgb2.fit(X2_train,y2_train)

xgb3 = XGBRegressor(learning_rate=xgb_best3.learning_rate,n_estimators=xgb_best3.n_estimators ,random_state=42) # use learning_rate instead of eta
xgb3.fit(X3_train,y3_train)

# predict the stress-strain for the test set for each dataset
y_pred = xgb.predict(X_test)
y_pred1 = xgb1.predict(X1_test)
y_pred2 = xgb2.predict(X2_test)
y_pred3 = xgb3.predict(X3_test)

# predict the stress-strain for the validation set for each dataset
y_val_pred = xgb.predict(X_val)
y_val_pred1 = xgb1.predict(X1_val)
y_val_pred2 = xgb2.predict(X2_val)
y_val_pred3 = xgb3.predict(X3_val)

# calculate and store the performance metrics for both sets for each dataset
with open('stress-strain_performance_metrics.txt','w') as f: # use 'w' mode to overwrite the existing file
 f.write('Dataset: C_0.01_stress-strain_data.xlsx\n')
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.5f}\n'.format(r2_score(y_test,y_pred)))
 f.write('RMSE: {:.5f}\n'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
 f.write('MSE: {:.5f}\n'.format(mean_squared_error(y_test,y_pred)))
 f.write('MAE: {:.5f}\n'.format(mean_absolute_error(y_test,y_pred)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.5f}\n'.format(r2_score(y_val,y_val_pred)))
 f.write('RMSE: {:.5f}\n'.format(np.sqrt(mean_squared_error(y_val,y_val_pred))))
 f.write('MSE: {:.5f}\n'.format(mean_squared_error(y_val,y_val_pred)))
 f.write('MAE: {:.5f}\n'.format(mean_absolute_error(y_val,y_val_pred)))
 f.write('Dataset: SC_0.01_stress-strain_data.xlsx\n')
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.5f}\n'.format(r2_score(y1_test,y_pred1)))
 f.write('RMSE: {:.5f}\n'.format(np.sqrt(mean_squared_error(y1_test,y_pred1))))
 f.write('MSE: {:.5f}\n'.format(mean_squared_error(y1_test,y_pred1)))
 f.write('MAE: {:.5f}\n'.format(mean_absolute_error(y1_test,y_pred1)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.5f}\n'.format(r2_score(y1_val,y_val_pred1)))
 f.write('RMSE: {:.5f}\n'.format(np.sqrt(mean_squared_error(y1_val,y_val_pred1))))
 f.write('MSE: {:.5f}\n'.format(mean_squared_error(y1_val,y_val_pred1)))
 f.write('MAE: {:.5f}\n'.format(mean_absolute_error(y1_val,y_val_pred1)))
 f.write('Dataset: C_0.001_stress-strain_data.xlsx\n')
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.5f}\n'.format(r2_score(y2_test,y_pred2)))
 f.write('RMSE: {:.5f}\n'.format(np.sqrt(mean_squared_error(y2_test,y_pred2))))
 f.write('MSE: {:.5f}\n'.format(mean_squared_error(y2_test,y_pred2)))
 f.write('MAE: {:.5f}\n'.format(mean_absolute_error(y2_test,y_pred2)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.5f}\n'.format(r2_score(y2_val,y_val_pred2)))
 f.write('RMSE: {:.5f}\n'.format(np.sqrt(mean_squared_error(y2_val,y_val_pred2))))
 f.write('MSE: {:.5f}\n'.format(mean_squared_error(y2_val,y_val_pred2)))
 f.write('MAE: {:.5f}\n'.format(mean_absolute_error(y2_val,y_val_pred2)))
 f.write('Dataset: SC_0.001_stress-strain_data.xlsx\n')
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.5f}\n'.format(r2_score(y3_test,y_pred3)))
 f.write('RMSE: {:.5f}\n'.format(np.sqrt(mean_squared_error(y3_test,y_pred3))))
 f.write('MSE: {:.5f}\n'.format(mean_squared_error(y3_test,y_pred3)))
 f.write('MAE: {:.5f}\n'.format(mean_absolute_error(y3_test,y_pred3)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.5f}\n'.format(r2_score(y3_val,y_val_pred3)))
 f.write('RMSE: {:.5f}\n'.format(np.sqrt(mean_squared_error(y3_val,y_val_pred3))))
 f.write('MSE: {:.5f}\n'.format(mean_squared_error(y3_val,y_val_pred3)))
 f.write('MAE: {:.5f}\n'.format(mean_absolute_error(y3_val,y_val_pred3)))
 f.close()

# Shadow effect objects with different transparency and smaller linewidth
pe1 = [patheffects.SimpleLineShadow(offset=(0.5,-0.5), alpha=0.4), patheffects.Normal()]

# create two subplots for the two multiplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot of the actual vs predicted stress-strain as a function of  time for the first multiplot
ax1.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test C', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax1.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test C', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax1.scatter(X_val[:, 0], y_val,color='green',label='Actual val C', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax1.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val C', linewidth=0.5,alpha=0.9,zorder=1,marker='+',path_effects=pe1)
ax1.scatter(X1_test[:, 0], y1_test,color='blue',label='Actual test SC', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax1.scatter(X1_test[:, 0], y_pred1,color='red',label='Predicted test SC', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax1.scatter(X1_val[:, 0], y1_val,color='yellow',label='Actual val SC', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax1.scatter(X1_val[:, 0], y_val_pred1,color='black',label='Predicted val SC', linewidth=0.5,alpha=0.9,zorder=1,marker='+',path_effects=pe1)
ax1.set_xlabel('Strain, %', fontsize='15', fontweight='bold')
ax1.set_ylabel('Stress, MPa', fontsize='15', fontweight='bold')
ax1.legend(loc='lower right')

# Add title
ax1.set_title("Strain rate 0.01", fontsize='18', fontweight='bold')

# Add a legend with shadow and different font size 
ax1.legend(shadow=True, prop={'size':'12'}, loc='upper left')
# Set the x axis limit to 60
ax1.set_xlim(0, 60) # use set_xlim instead of xlim
# Set the y axis limit to 60
ax1.set_ylim(0, 60) # use set_ylim instead of ylim
# Change the axes numbering size and font
ax1.tick_params(axis='both', which='major', labelsize=12, labelcolor='black')
ax1.grid() # add grid to the second subplot


# Plot of the actual vs predicted stress-strain as a function of  time for the second multiplot
ax2.scatter(X2_test[:, 0], y2_test,color='cyan',label='Actual test C', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax2.scatter(X2_test[:, 0], y_pred2,color='orange',label='Predicted test C', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax2.scatter(X2_val[:, 0], y2_val,color='green',label='Actual val C', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax2.scatter(X2_val[:, 0], y_val_pred2,color='magenta',label='Predicted val C', linewidth=0.5,alpha=0.9,zorder=1,marker='+',path_effects=pe1)
ax2.scatter(X3_test[:, 0], y3_test,color='blue',label='Actual test SC', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax2.scatter(X3_test[:, 0], y_pred3,color='red',label='Predicted test SC', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax2.scatter(X3_val[:, 0], y3_val,color='yellow',label='Actual val SC', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
ax2.scatter(X3_val[:, 0], y_val_pred3,color='black',label='Predicted val SC', linewidth=0.5,alpha=0.9,zorder=1,marker='+',path_effects=pe1)
ax2.set_xlabel('Strain, %', fontsize='15', fontweight='bold')
ax2.set_ylabel('Stress, MPa', fontsize='15', fontweight='bold')


# Add title
ax2.set_title("Strain rate 0.001",fontsize='18', fontweight='bold')

# Add a legend with shadow and different font size 
ax2.legend(shadow=True, prop={'size':'12'}, loc='upper left')
# Set the x axis limit to 60
ax2.set_xlim(0, 60) # use set_xlim instead of xlim
# Set the y axis limit to 60
ax2.set_ylim(0, 60) # use set_ylim instead of ylim
# Change the axes numbering size and font
ax2.tick_params(axis='both', which='major', labelsize=12, labelcolor='black')
ax2.grid() # add grid to the second subplot


# Save the plot with dpi=500 in 'png'
fig.savefig('pred_stress-strain_c_multi.png', dpi=500)

# create a DataFrame from the variables for each dataset
df1 = pd.DataFrame({"Actual test": y_test, "Predicted test": y_pred, "Actual val": y_val, "Predicted val": y_val_pred})
df2 = pd.DataFrame({"Actual test": y1_test, "Predicted test": y_pred1, "Actual val": y1_val, "Predicted val": y_val_pred1})
df3 = pd.DataFrame({"Actual test": y2_test, "Predicted test": y_pred2, "Actual val": y2_val, "Predicted val": y_val_pred2})
df4 = pd.DataFrame({"Actual test": y3_test, "Predicted test": y_pred3, "Actual val": y3_val, "Predicted val": y_val_pred3})

# save the DataFrames to an Excel file with different sheets
with pd.ExcelWriter("pred_stress-strain_multi.xlsx") as writer:
 df1.to_excel(writer, sheet_name="C_0.01", index=False)
 df2.to_excel(writer, sheet_name="SC_0.01", index=False)
 df3.to_excel(writer, sheet_name="C_0.001", index=False)
 df4.to_excel(writer, sheet_name="SC_0.001", index=False)


# Descriptive statistics


# create a DataFrame from the descriptive statistics for each dataset
df1_stats = df1.describe()
df2_stats = df2.describe()
df3_stats = df3.describe()
df4_stats = df4.describe()

# rename the labels of the rows
df1_stats = df1_stats.rename(index={'50%': 'median'})
df2_stats = df2_stats.rename(index={'50%': 'median'})
df3_stats = df3_stats.rename(index={'50%': 'median'})
df4_stats = df4_stats.rename(index={'50%': 'median'})


# select only the standard deviation, mean, median, minimum, and maximum from each DataFrame
df1_stats = df1_stats.loc[['std', 'mean', 'median', 'min', 'max']]
df2_stats = df2_stats.loc[['std', 'mean', 'median', 'min', 'max']]
df3_stats = df3_stats.loc[['std', 'mean', 'median', 'min', 'max']]
df4_stats = df4_stats.loc[['std', 'mean', 'median', 'min', 'max']]

# print the selected statistics for each DataFrame
print("Statistics for C_0.01_stress-strain_data.xlsx")
print(df1_stats)
print("Statistics for SC_0.01_stress-strain_data.xlsx")
print(df2_stats)
print("Statistics for C_0.001_stress-strain_data.xlsx")
print(df3_stats)
print("Statistics for SC_0.001_stress-strain_data.xlsx")
print(df4_stats)

# save the DataFrames to an Excel file with different sheets
with pd.ExcelWriter("statistics_stress-strain.xlsx") as writer:
 df1_stats.to_excel(writer, sheet_name="C_0.01", index=True) # use index=True to write the row names
 df2_stats.to_excel(writer, sheet_name="SC_0.01", index=True) # use index=True to write the row names
 df3_stats.to_excel(writer, sheet_name="C_0.001", index=True) # use index=True to write the row names
 df4_stats.to_excel(writer, sheet_name="SC_0.001", index=True) # use index=True to write the row names



