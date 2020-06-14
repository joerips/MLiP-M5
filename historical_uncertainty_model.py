import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#import ipywidgets

#from ipywidgets import widgets, interactive, interact
import ipywidgets as widgets
from IPython.display import display

import os
for dirname, _, filenames in os.walk('input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train_sales = pd.read_csv(r'C:\Users\Kapiteun\Desktop\Master\MLiP\m5-forecasting-accuracy/sales_train_validation.csv')
calendar_df = pd.read_csv(r'C:\Users\Kapiteun\Desktop\Master\MLiP\m5-forecasting-accuracy/calendar.csv')
submission_file = pd.read_csv(r'C:\Users\Kapiteun\Desktop\Master\MLiP\m5-forecasting-accuracy/sample_submission.csv')
sell_prices = pd.read_csv(r'C:\Users\Kapiteun\Desktop\Master\MLiP\m5-forecasting-accuracy/sell_prices.csv')

total = ['Total']
train_sales['Total'] = 'Total'
train_sales['state_cat'] = train_sales.state_id + "_" + train_sales.cat_id
train_sales['state_dept'] = train_sales.state_id + "_" + train_sales.dept_id
train_sales['store_cat'] = train_sales.store_id + "_" + train_sales.cat_id
train_sales['store_dept'] = train_sales.store_id + "_" + train_sales.dept_id
train_sales['state_item'] = train_sales.state_id + "_" + train_sales.item_id
train_sales['item_store'] = train_sales.item_id + "_" + train_sales.store_id

val_eval = ['validation', 'evaluation']

# creating lists for different aggregation levels
total = ['Total']
states = ['CA', 'TX', 'WI']
num_stores = [('CA',4), ('TX',3), ('WI',3)]
stores = [x[0] + "_" + str(y + 1) for x in num_stores for y in range(x[1])]
cats = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
num_depts = [('FOODS',3), ('HOBBIES',2), ('HOUSEHOLD',2)]
depts = [x[0] + "_" + str(y + 1) for x in num_depts for y in range(x[1])]
state_cats = [state + "_" + cat for state in states for cat in cats]
state_depts = [state + "_" + dept for state in states for dept in depts]
store_cats = [store + "_" + cat for store in stores for cat in cats]
store_depts = [store + "_" + dept for store in stores for dept in depts]
prods = list(train_sales.item_id.unique())
prod_state = [prod + "_" + state for prod in prods for state in states]
prod_store = [prod + "_" + store for prod in prods for store in stores]

print("Departments: ", depts)
print("Categories by state: ", state_cats)

quants = ['0.005', '0.025', '0.165', '0.250', '0.500', '0.750', '0.835', '0.975', '0.995']
days = range(1, 1913 + 1)
time_series_columns = [f'd_{i}' for i in days]

def create_sales(name_list, group):
    '''
    This function returns a dataframe (sales) on the aggregation level given by name list and group
    '''
    rows_ve = [(name + "_X_" + str(q) + "_" + ve, str(q)) for name in name_list for q in quants for ve in val_eval]
    sales = train_sales.groupby(group)[time_series_columns].sum() #would not be necessary for lowest level
    return sales
    
total = ['Total']
train_sales['Total'] = 'Total'
train_sales['state_cat'] = train_sales.state_id + "_" + train_sales.cat_id
train_sales['state_dept'] = train_sales.state_id + "_" + train_sales.dept_id
train_sales['store_cat'] = train_sales.store_id + "_" + train_sales.cat_id
train_sales['store_dept'] = train_sales.store_id + "_" + train_sales.dept_id
train_sales['state_item'] = train_sales.state_id + "_" + train_sales.item_id
train_sales['item_store'] = train_sales.item_id + "_" + train_sales.store_id

#example usage of CreateSales
sales_by_state_cats = create_sales(state_cats, 'state_cat')
sales_by_state_cats

def create_quantile_dict(name_list = stores, group = 'store_id' ,X = False):
    '''
    This function writes creates sales data on given aggregation level, and then writes predictions to the global dictionary my_dict
    '''
    sales = create_sales(name_list, group)
    sales = sales.iloc[:, 1668:] #using the last few months data only
    sales_quants = pd.DataFrame(index = sales.index)
    for q in quants:
        sales_quants[q] = np.quantile(sales, float(q), axis = 1)
    full_mean = pd.DataFrame(np.mean(sales, axis = 1))
    daily_means = pd.DataFrame(index = sales.index)
    for i in range(7):
        daily_means[str(i)] = np.mean(sales.iloc[:, i::7], axis = 1)
    daily_factors = daily_means / np.array(full_mean)

    daily_factors = pd.concat([daily_factors, daily_factors, daily_factors, daily_factors], axis = 1)
    daily_factors_np = np.array(daily_factors)

    factor_df = pd.DataFrame(daily_factors_np, columns = submission_file.columns[1:])
    factor_df.index = daily_factors.index

    for i,x in enumerate(tqdm(sales_quants.index)):
        for q in quants:
            v = sales_quants.loc[x, q] * np.array(factor_df.loc[x, :])
            if X:
                my_dict[x + "_X_" + q + "_validation"] = v
                my_dict[x + "_X_" + q + "_evaluation"] = v
            else:
                my_dict[x + "_" + q + "_validation"] = v
                my_dict[x + "_" + q + "_evaluation"] = v
                
my_dict = {}
#adding prediction to my_dict on all 12 aggregation levels
create_quantile_dict(total, 'Total', X=True) #1
create_quantile_dict(states, 'state_id', X=True) #2
create_quantile_dict(stores, 'store_id', X=True) #3
create_quantile_dict(cats, 'cat_id', X=True) #4
create_quantile_dict(depts, 'dept_id', X=True) #5
create_quantile_dict(state_cats, 'state_cat') #6
create_quantile_dict(state_depts, 'state_dept') #7
create_quantile_dict(store_cats, 'store_cat') #8
create_quantile_dict(store_depts, 'store_dept') #9
create_quantile_dict(prods, 'item_id', X=True) #10
create_quantile_dict(prod_state, 'state_item') #11
create_quantile_dict(prod_store, 'item_store') #12

pred_df = pd.DataFrame(my_dict)
pred_df = pred_df.transpose()
pred_df_reset = pred_df.reset_index()
final_pred = pd.merge(pd.DataFrame(submission_file.id), pred_df_reset, left_on = 'id', right_on = 'index')
del final_pred['index']
final_pred = final_pred.rename(columns={0: 'F1', 1: 'F2', 2: 'F3', 3: 'F4', 4: 'F5', 5: 'F6', 6: 'F7', 7: 'F8', 8: 'F9',
                                        9: 'F10', 10: 'F11', 11: 'F12', 12: 'F13', 13: 'F14', 14: 'F15', 15: 'F16',
                                        16: 'F17', 17: 'F18', 18: 'F19', 19: 'F20', 20: 'F21', 21: 'F22', 
                                        22: 'F23', 23: 'F24', 24: 'F25', 25: 'F26', 26: 'F27', 27: 'F28'})
final_pred = final_pred.fillna(0)

final_pred.to_csv(r'C:\Users\Kapiteun\Desktop\Master\MLiP\m5-forecasting-accuracy/submission.csv', index=False)

final_pred.head()

## Making Boxplot of Total Prediction

df = pd.read_csv(r'C:\Users\Kapiteun\Desktop\Master\MLiP\m5-forecasting-accuracy/submission.csv')

#define quantiles
quants = ['0.005', '0.025', '0.165', '0.250', '0.500', '0.750', '0.835', '0.975', '0.995']
#get wanted values of predictions
total_intervals = ["Total_X_" + str(q) + "_validation" for q in quants]

boxdata = []
for id in total_intervals:
      newdf = df[df["id"] == id]
      boxdata.append(newdf.values)

#get certain quantile data to plot
lowest_quart = boxdata[0][0][1:].tolist() 
lower_quart = boxdata[1][0][1:].tolist() 
low_quart = boxdata[2][0][1:].tolist() 
first_quart = boxdata[3][0][1:].tolist()     
median = boxdata[4][0][1:].tolist()
third_quart = boxdata[5][0][1:].tolist()
high_quart = boxdata[6][0][1:].tolist()
higher_quart = boxdata[7][0][1:].tolist()  
highest_quart = boxdata[8][0][1:].tolist()

#get actual data to plot
df_ev = pd.read_csv(r'C:\Users\Kapiteun\Desktop\Master\MLiP\m5-forecasting-accuracy/sales_train_evaluation.csv')
last_days_values = df_ev.iloc[:,1918:1947]
last_days_ids = df_ev.iloc[:,0:5]
last_days = last_days_ids.join(last_days_values)

sum_sales = last_days.sum()
scatter_sales = sum_sales[6:34]

#make percentile-actual data plot
x = np.arange(1,29,1)

plt.xlim(0.85,28.15)
plt.ylim(20000, 85000)
plt.xticks(np.arange(1,29,3))
plt.yticks(np.arange(20000,80000,10000))
plt.fill_between(x, first_quart, third_quart, alpha=0.9, color='g', label='25.0 - 75.0 percentile')
plt.fill_between(x, low_quart, high_quart, alpha=0.40, color='g', label = '16.5 - 83.5 percentile')
plt.fill_between(x, lower_quart, higher_quart, alpha=0.30, color='g', label='2.5 - 97.5 percentile')
plt.fill_between(x, lowest_quart, highest_quart, alpha=0.15, color='g', label='0.05 - 99.5 percentile')

plt.plot(x,median, 'k', label='median')
plt.scatter(x,scatter_sales, s=20, c='r', marker="o", label='actual data points')
plt.legend(loc="upper left", ncol=2)
plt.title('The median, percentile predictions and actual data of the total sales', fontsize=9.8)
plt.xlabel('prediction day number')
plt.ylabel('amount of sales')
plt.grid(color='k', linestyle='-.', linewidth=0.1)
plt.show()

