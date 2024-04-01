import streamlit as st
import pandas as pd

import math
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


df = pd.read_csv('final_smartphones.csv')
df.drop(columns='Unnamed: 0',inplace=True)

X = df.drop(columns='price')
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
#
#
# # import the model
# pipe = pickle.load(open('pipe.pkl','rb'))
# # df = pickle.load(open('df.pkl','rb'))
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
step1 = ColumnTransformer(transformers=[('cols',
                                    OneHotEncoder(sparse_output=False,drop='first',handle_unknown='ignore')
                                         ,[0,2,3,4,5,6,11,19])],
                         remainder='passthrough')
step2 = LinearRegression()

pipe = Pipeline([
    ('step1',step1),
    ('step2',step2)
])
pipe.fit(X_train,y_train)









st.title('Smart Phone Price Predictor')
# brand_names
st.write('\n')
brand = st.selectbox('Select the Smartphone Company\n',df['brand_names'].unique())

# Rating
st.write('\n')
rating = float(st.slider("Select a Rating of SmartPhone", 0, 100, 50))

# has_5G
st.write('\n')
temp = st.checkbox('Require support of 5G? ')
if temp == True:
    has_5G = 'Yes'
else:
    has_5G = 'No'

# Chipset
st.text('\n')
selected_option = st.radio("Select an ChipSet", ("NFC", "IR-Blaster", "None"))
# Display message based on selected option
if selected_option == "NFC":
    chipset = 'NFC'
elif selected_option == "IR Blaster":
    chipset = 'IR-Blaster'
else:
    chipset = 'None'

# Processor_Company
st.write('\n')
df['Processor_Company']=df['Processor_Company'].str.lower()
df['Processor_Company']=df['Processor_Company'].str.replace('(28','Mediatek')
processor_company = st.selectbox('Select the Processor Company\n',df['Processor_Company'].unique())

# Processor_Name
processor_name = st.selectbox('Select the Processor Name\n',df['Processor_Name'].unique())

# Processor_Core
processor_core = st.selectbox('Select the Processor Cores\n',df['Processor_Core'].unique())

# Processor speed
st.write('\n')
speed = float(st.slider("Select a Processor Speed in GHz", 1.0, 4.5,2.7))

# RAM_in_GB
st.write('\n')
ram = int(st.selectbox('Select the RAM in GB\n',[1,2,3,4,6,10,12,16,18]))

# ROM_in_GB
st.write('\n')
rom = int(st.selectbox('Select the ROM in GB\n',sorted(df['ROM_in_GB'].unique())))

#Battery_Capacity_mAh
st.write('\n')
battery = float(st.slider("Select a Battery Capacity in mAh", 1000, 25000,3500))

# Supports_Fast_Charging
st.write('\n')
temp = st.checkbox('Require support of Fast Charging ? ')
if temp == True:
    fast = 'Yes'
else:
    fast = 'No'

# Display_in_inches
st.write('\n')
display = float(st.slider("Select a Size of Display in inches", 2.0, 8.5,6.56))

# Display_Refresh_Rate
st.write('\n')
refresh_rate =int( st.selectbox('Select the Refresh Rate of Display in Hz \n',sorted(df['Display_Refresh_Rate'].unique())))

# Number_of_Rear_Camera
st.write('\n')
rear = int(st.selectbox('Select the Number of Rear Camera \n',sorted(df['Number_of_Rear_Camera'].unique())))

# Number_of_Front_Camera
st.write('\n')
front = int(st.selectbox('Select the Number of Front Camera \n',sorted(df['Number_of_Front_Camera'].unique())))

# MP_of_Rear
st.write('\n')
rear_mp = float(st.slider("Select a MegaPixels of Rear Camera", 2,200,48))

# MP_of_Front
st.write('\n')
front_mp = float(st.slider("Select a MegaPixels of Front Camera", 2,60,12))

# Memory_Card_Supported_GB

st.write('\n')
card = int(st.selectbox('Select the Size of Memory Card to be Supported \n',sorted(df['Memory_Card_Supported_GB'].unique())))
# OS_Name
os = st.selectbox('Select the Appropiate OS \n',df['OS_Category'].unique())

# OS_Version
version = st.selectbox('Select the Appropiate OS Version\n',sorted(df['OS_Version'].unique()))

st.text('\n')
query = [[brand, rating, has_5G, chipset, processor_company, processor_name, processor_core, speed, ram, rom
    , battery, fast, display, refresh_rate, rear, front, rear_mp, front_mp, card, os, version]]
if st.button("Predict Price"):
    price = pipe.predict(query)
    price = math.floor(price)
    st.subheader('Predicted Price of the Smartphone may be:   ' + str(price)+'₹')