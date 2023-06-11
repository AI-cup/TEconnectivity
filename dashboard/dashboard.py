import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go

df = pd.read_csv("data/data_final.csv")
p_index = df.iloc[:0, 1:]


smooth=0
lumpy=0
bulky=0



# for data in sales_data[]:
#     cv = (np.std(data) / np.mean(data)) * 100
#     if(cv<20):
#         smooth+=1
#     elif(cv>20 and cv<50):
#         lumpy+=1
#     elif(cv>=50):
#         bulky+=1


st.header(" DEMAND FORECASTING MODEL")
col1, col2, col3 = st.columns(3)
col1.metric("Smooth", "000", "6%")
col2.metric("Lumpy", "000", "8%")
col3.metric("Bulky", "999", "4%")
# streamlit run test.py in the terminal


st.subheader("Spider plot distribution of given Data")
# Define the data



# ''' In this example, we use the go.Scatterpolar() function from the plotly.graph_objs module to define the data for the radar chart. The r parameter specifies the radial values, while the theta parameter specifies the angles. We also set the fill parameter to 'toself' to fill the area enclosed by the chart.

# We then define the layout for the chart using the go.Layout() function. In this case, we set the radialaxis parameter to have a visible range from 0 to 5.

# Next, we use the go.Figure() function to create the figure, passing in the data and layout objects as arguments.'''

data = [
    go.Scatterpolar(
        r=[5, 4, 3, 2, 1],
        theta=['Smooth', 'Lumpy', 'Bulky',],
        fill='toself'
    )
]

# Define the layout
layout = go.Layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 5]
        )
    )
)

# Create the figure
fig = go.Figure(data=data, layout=layout)

# Plot the figure using Streamlit
st.plotly_chart(fig)



options = ['Option 1', 'Option 2', 'Option 3']
selected_option = st.selectbox('Select a part number', options)

st.write('You selected:', selected_option)


# Code to link select box with line chart


st.subheader("Comparision by line chart for Part no")
chart_data = pd.DataFrame(
    np.random.randn(20, 2),
    columns=['a', 'b'])

st.line_chart(chart_data)