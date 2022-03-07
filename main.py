import streamlit as st
from firebase import  firebase
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.fft import fftshift
import os
import glob
import json
from PIL import Image
import csv
import json

# s3_client = boto3.client('s3')
# response=s3_client.get_object(Bucket='trebirth1',Key='FB.json')
# result = response['Body'].read()
# print("Result is",result)
#retrieve json file from firebase
firebase = firebase.FirebaseApplication('https://esp32-d544d-default-rtdb.firebaseio.com/',None)
result = firebase.get("test","sensor")
scan = firebase.get("test","scan_number")

Np_result=np.array(result)
print(Np_result)
df = pd.DataFrame(Np_result)
df.to_csv("scan.csv")


jtopy=json.dumps(result)       #json.dumps take a dictionary as input and returns a string as output.
dict_json=json.loads(jtopy)    # json.loads take a string as input and returns a dictionary as output.
#print(dict_json)

# Digital Filter starts from here
def Data_Preprocess(x):
 sig = [np.array(x)]
 # print("Sig is ",sig)
 return sig

def Apply_Filter(sig):
    sos = signal.butter(1, [0.1, 20], 'band', fs=100, output='sos')
    filtered = signal.sosfilt(sos, sig)
    # print ("Filtered data is ",filtered)
    return filtered.squeeze()


def Plot_Graph(filtered):
   t = np.linspace(0, 30,3000, False)
   t = t[:filtered.size]
   fig, ax = plt.subplots()
   x = t.squeeze()
   line, = ax.plot(x, filtered.squeeze())

   #plt.plot(t.squeeze(),filtered.squeeze())
  # plt.suptitle('Filtered Scan Data')
   #plt.axis([0, 15, 0, 400])
   #plt.savefig("output.jpg")

Data = Data_Preprocess(dict_json)
# print("Data is ",Data)
Filtered_data = Apply_Filter(Data)
# print(Filtered_data)
Plot_Graph(Filtered_data)
# print(Filtered_data)
# plt.show()
plt.savefig("output.jpg")



#Streamlit GUI starts from here
a=st.sidebar.radio('Navigation',['Farm Information','Farmer Data'])
# df = pd.read_csv("Trebirth.csv")

if a == "Farm Information":
 st.header("Welcome to Trebirth Tech Development")
 # form = st.form(key='my_form',clear_on_submit=True)
 # F_name= form.text_input(label='Enter Farmer Name')
 # F_health= form.text_input(label='Enter Farm Health')
 # Number= form.number_input(label='Enter No. of trees scanned')
 # Remark = form.text_area(label='Remark')
 # submit_button = form.form_submit_button(label='Submit')


 # st.sidebar.markdown(
 #    f"""
 #     * Farmer name :        {F_name}
 #     * Farm health :        {F_health}
 #     * No of trees scanned: {Number}
 #     * Remark      :        {Remark}
 # """
 #  )

 st.line_chart(Filtered_data,use_container_width=True)

 st.write(df)
 st.write("Scan number is",scan)


 # st.write(result)
 # st.write(scan)


 # st.dataframe(df)


 # df.to_csv("Trebirth.csv",index=False)
 # st.dataframe(df)
 # if submit_button:
 #     st.write(F_name,F_health,Number,Remark)
 # new_data = {"Farmer_Name": F_name,"Farm_Health": F_health,"Trees_Scanned": int(Number),"Remark": Remark}
 #st.write(new_data)

 # df = df._append(new_data,ignore_index=True,sort=False)
 # df.to_csv("Trebirth.csv",index=False)
 # st.dataframe(df)



@st.cache
def convert_df(df):
     return df.to_csv().encode('utf-8')
csv = convert_df(df)
st.download_button(
     "Press to Download",
     csv,
     "file.csv",
     "text/csv",
     key='download-csv'
 )





 # col1, col2= st.columns(2)
 #
 # with col1:
 #     st.header("Filtered Data")
 #     st.line_chart(Filtered_data)

 # with col2:
 #     st.header(" Accelerometer")
 #     st.line_chart(Filtered_data)






