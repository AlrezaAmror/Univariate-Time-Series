# Dashboard Framework
import streamlit as st

# Data Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Time Series Analysis
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
from pmdarima.arima import auto_arima

st.title("Analysis dan Peramalan Univariate Time Series")
st.write("Pastikan Seluruh Data Memiliki Format yang Telah Ditetapkan dan Tidak Ada Nilai yang Kosong")

# Widget untuk mengunggah file berekstensi csv/xlsx
uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

# Membaca dataset
if uploaded_file is not None:
        
    # Read file jika data memiliki ekstensi .csv
    if uploaded_file.type == "text/csv":
        dataBPS = pd.read_csv(uploaded_file)

    # Read file jika data memiliki ekstensi .xlsx
    else:
        dataExcel = pd.ExcelFile(uploaded_file)
        sheets = dataExcel.sheet_names
        dataBPS = pd.read_excel(dataExcel)
        # dataBPS = pd.read_excel(dataExcel, sheet_name=sheets[0])

        # ubah nama kolom sesuai format yang ditetapkan
        dataBPS = dataBPS.rename(columns = {
            dataBPS.columns[0] : 'Periode',
            dataBPS.columns[1] : 'Value'
        })

        # ubah kolom berisi tanggal menjadi format date time dan set ke index
        dataBPS['Periode'] = pd.to_datetime(dataBPS['Periode'])
        dataBPS = dataBPS.set_index('Periode')

        # Menampilkan DataFrame pada website
        st.subheader("Data Time Series yang diunggah: ")
        st.dataframe(dataBPS.tail(12))

        # Analysis Awal Data
        fig = go.Figure() # figure
        fig.add_trace(
                go.Scatter(
                    x = dataBPS.index,
                    y = dataBPS['Value'],
                    name = 'Dataset',
                    mode = 'lines+markers',
                    line=dict(color='blue', width=2)
                )
            )
        fig.update_layout(
            title = 'Dataset Time Series',
            xaxis_title = 'Periode',
            yaxis_title = 'Value',
            hovermode = 'x unified',
            template = 'plotly_dark',
            xaxis=dict(tickformat='%d-%m-%Y'),
            showlegend = True
            )
        st.plotly_chart(fig, use_container_width=True)

        # Additive Seasonal Decompose
        result = seasonal_decompose(dataBPS.Value, model='additive')
        trend = pd.DataFrame(result.trend)
        seasonal = pd.DataFrame(result.seasonal)
        residual = pd.DataFrame(result.resid)

        # tren fit dengan linier regression
        X = [i for i in range(0, len(dataBPS.Value))]
        X = np.reshape(X, (len(X), 1))
        y = dataBPS['Value'].values
        LR = LinearRegression()
        LR.fit(X, y)
        trend_LR = LR.predict(X)

        decomposeDf = pd.concat([trend, seasonal, residual], axis=1)
        decomposeDf['LR_trend'] = trend_LR

        # Visualisasi Seasonal Decompose Trend
        trends = go.Figure() # figure
        # line trend
        trends.add_trace(
                go.Scatter(
                    x = decomposeDf.index,
                    y = decomposeDf['trend'],
                    name = 'Trend Dataset',
                    line=dict(color='blue', width=2)
                )
            )
        # line LR
        trends.add_trace(
                go.Scatter(
                    x = decomposeDf.index,
                    y = decomposeDf['LR_trend'],
                    name = 'Linear Trend',
                    line=dict(color='red', width=2)
                )
            )

        trends.update_layout(
                title = 'Trend Dataset Time Series',
                xaxis_title = 'Periode',
                yaxis_title = 'Value',
                hovermode = 'x unified',
                template = 'plotly_dark',
                showlegend= True
            )
        st.plotly_chart(trends, use_container_width=True)

        # Visualisasi Seasonality
        season = go.Figure() # figure
        season.add_trace(
                go.Scatter(
                    x = decomposeDf.index,
                    y = decomposeDf['seasonal'],
                    name = 'Seasonal',
                    mode = 'lines+markers',
                    line=dict(color='gray', width=2)
                )
            )
        season.update_layout(
            title = 'Seasonal Time Series',
            xaxis_title = 'Periode',
            yaxis_title = 'Value',
            hovermode = 'x unified',
            template = 'plotly_dark',
            # xaxis=dict(tickformat='%d-%m-%Y'),
            showlegend = True
            )
        st.plotly_chart(season, use_container_width=True)

        # Transform Value agar stationer
        # Log transform
        dataBPS['logVal'] = np.log(dataBPS['Value'])

        # Diff transform
        def diff_transform(series):
            return pd.Series(series.diff().values)
        # invert differenced value
        def invert_diff_transform(initial_actual_item, diff_series):
            return np.r_[initial_actual_item, diff_series].cumsum()
        
        # Uji stationer

        # Auto Arima
        season = st.text_input("Masukkan Seasonality Data :") # input seasonality
        if season == '':
            season=12
        else:
            pass
        
        if 'clicked' not in st.session_state:
            st.session_state.clicked = False
        def click_button():
            st.session_state.clicked = True

        st.button('Klik untuk melakukan peramalan (akan membutuhkan sedikit waktu)', on_click=click_button)
        if st.session_state.clicked:
            model = auto_arima(dataBPS['logVal'],
                            start_p=1, start_q=1, d=1, max_p=5, max_q=5, max_d=5, m=int(float(season)),
                            start_P=0, D=1, start_Q=0, max_P=5, Max_D=5, Max_Q=5,
                            seasonal=True, information_criteria='AIC')
            order = model.order
            seasonal_order = model.seasonal_order
            sarima = SARIMAX(dataBPS['logVal'], order=(order[0], order[1], order[2]),
                    seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_order[3]))
            sarima_model = sarima.fit()        

            # forecast
            pred = np.exp(sarima_model.predict(start = len(dataBPS), end = len(dataBPS)+11))
            forecast = pd.DataFrame(pred)
            forecast = forecast.rename(columns={forecast.columns[0]:'Value'})

            # Visualisasi forecast
            fig = go.Figure() # figure
            # line data observasi
            fig.add_trace(
                go.Scatter(
                    x = dataBPS.index,
                    y = dataBPS['Value'],
                    name = 'Data Observasi',
                    mode = 'lines+markers',
                    line=dict(color='blue', width=2)
                )
            )
            # line forecast
            fig.add_trace(
                go.Scatter(
                    x = forecast.index,
                    y = forecast['Value'],
                    mode = 'lines+markers',
                    name = 'Forecast',
                    line=dict(color='red', width=2)
                )
            )

            fig.update_layout(
                title = 'Peramalan Data Time Series 12 Bulan',
                xaxis_title = 'Periode',
                yaxis_title = 'Value',
                hovermode = 'x unified',
                template = 'plotly_dark',
                showlegend= True
            )
            st.plotly_chart(fig, use_container_width=True)

            # # visualisasi prediksi
            # fig = px.line(result, x=result.index, y=result.Value, markers=True)
            # fig.update_xaxes(title='Periode')
            # fig.update_yaxes(title='Total')
            # fig.update_layout(xaxis=dict(tickformat='%d-%m-%Y'))
            # st.plotly_chart(fig, use_container_width=True)