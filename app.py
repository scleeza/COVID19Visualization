import pandas as pd
import numpy as np
from pandas.plotting import register_matplotlib_converters
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
import datetime as dt
import sklearn
import altair as alt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import streamlit as st
import warnings
from pickle import load
import tensorflow as tf

warnings.filterwarnings("ignore")

register_matplotlib_converters()

# URL for fetching data
CONFIRM_DATA_URL = ('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
                    '/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv ')
DEATH_DATA_URL = ('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
                  '/csse_covid_19_time_series/time_series_covid19_deaths_global.csv ')

DEFAULT_NUMBER_OF_ROWS = 5
DEFAULT_NUMBER_OF_COLUMNS = 5


def main():
    st.sidebar.info('Streamlit Version:{}'.format(st.__version__))
    st.title('Covid-19 Worldwide Cases Prediction')
    with st.spinner('Loading Data...'):

        # Load dataset from github repo and cleaned data before print out.
        df_confirm = load_data(CONFIRM_DATA_URL)
        df_death = load_data(DEATH_DATA_URL)
        st.sidebar.subheader("Country/Region")
        country_name = st.sidebar.selectbox('Chose Country', df_confirm.columns.tolist(),
                                            index=len(df_confirm.columns) - 1)
        st.sidebar.subheader("Observations")
        num_observations = st.sidebar.slider('See how many observations', min_value=10, max_value=len(df_confirm),
                                             value=len(df_confirm), step=10)
        latest_date = datetime.strptime(df_confirm.index[-1], '%m/%d/%y').date()
        country2id = {}
        for i, name in enumerate(df_confirm.columns.to_list()):
            country2id[name] = i
        # '''
        # code below are working on:
        # Load DL models and pre-processing, including:
        # 1.trim input data (latest 30 days' daily increase cases)
        # 2.standardization by tf_scaler
        # 3.get prediction form tf_models
        # 4.inverse back to origin data format
        # '''
        # Load models and scaler
        tf_models = load_tf_models()
        tf_scaler = [load(open("scaler.pkl","rb")),load(open("scaler_d.pkl","rb"))]
        st.sidebar.subheader("LSTM Result")
        # #model_select = st.sidebar.multiselect("Select models:",list(tf_models.keys()),default=list(tf_models.keys()))
        lstm_select = st.sidebar.checkbox('Show LSTM Prediction',value=True)
        #arima_select = st.sidebar.checkbox('Select ARIMA', value=True)
        # trim data and standardization into tensor format
        STEP =30
        input = []
        confirm_input = df_confirm.diff().dropna()[-STEP:]
        input.append( tf.expand_dims(tf_scaler[0].transform(confirm_input),axis=0))
        death_input = df_death.diff().dropna()[-STEP:]
        input.append(tf.expand_dims(tf_scaler[1].transform(death_input), axis=0))
        # get predictions
        pred_conf, pred_death = get_preds(models=tf_models, input=input, scaler=tf_scaler)
        start_date = df_confirm.index[-1]
        index = pd.date_range(start= latest_date+ dt.timedelta(days=1), periods=STEP)
        conf_pred = pd.DataFrame(pred_conf['LSTM'],
                                 index=index,
                                 columns=df_confirm.columns.tolist()).cumsum()+df_confirm.loc[start_date,:]
        death_pred = pd.DataFrame(pred_death['LSTM'],
                                  index=index,
                                  columns=df_confirm.columns.tolist()).cumsum()+df_death.loc[start_date,:]




        # All preprocessing steps done
        st.success(r'Data last uploaded at {}'.format(latest_date))

        if country_name == 'World':
            st.subheader("Global Stats")
            c1, c2 = st.beta_columns(2)
            c1.write('Total Confirmed Cases:')
            c1.info('{:,}'.format(df_confirm.iloc[-1, -1]))
            c1.dataframe(
                df_confirm.diff().dropna()[-7:]['World'].rename('Daily Increase').apply(lambda x: "{:,}".format(int(x))))
            c1.subheader('Fastest increasing Countries')
            c1.dataframe(select_top_country(df_confirm))
            c2.write('Total Death Cases:')
            c2.info('{:,}'.format(df_death.iloc[-1, -1]))
            c2.dataframe(
                df_death.diff().dropna()[-7:]['World'].rename('Daily Increase').apply(lambda x: "{:,}".format(int(x))))
            c2.subheader('Fastest increasing Countries')
            c2.dataframe(select_top_country(df_death))
            st.sidebar.subheader('World Trend Prediction')
            show_global = st.sidebar.checkbox('Show prediction', value=False)
            # ARIMA model
            model = arima(df_confirm,
                          country_name,
                          latest_date,
                          lstm_pred=conf_pred,
                          obs_num=num_observations,
                          )
            death_model = arima(df_death,
                                country_name,
                                latest_date,
                                lstm_pred=death_pred,
                                obs_num=num_observations,
                                )
            st.subheader('Confirmed Cases Trend:')
            st.altair_chart(draw_trend(df=df_confirm,
                                       show_global=show_global,
                                       model=model,
                                       lstm_select=lstm_select,
                                       num_obv= num_observations))
            st.subheader('Death Cases Trend:')
            st.altair_chart(draw_trend(df=df_death,
                                       show_global=show_global,
                                       model=death_model,
                                       lstm_select=lstm_select,
                                       num_obv=num_observations))


        else:
            st.subheader("{} Stats".format(country_name))
            u1, u2 = st.beta_columns(2)
            u1.write('Total Confirmed Cases:')
            u1.info("{:,}".format(df_confirm[country_name][-1]))
            u1.dataframe(
                df_confirm.diff().dropna()[-7:][country_name].rename('Daily Increase').apply(lambda x: "{:,}".format(int(x))))
            u2.write('Total Death Cases:')
            u2.info("{:,}".format(df_death[country_name][-1]))
            u2.dataframe(
                df_death.diff().dropna()[-7:][country_name].rename('Daily Increase').apply(lambda x: "{:,}".format(int(x))))
            # fit model on confirm and death dataset
            model = arima(df_confirm,
                          country_name,
                          latest_date,
                          lstm_pred=conf_pred,
                          obs_num=num_observations,
                          )
            death_model = arima(df_death,
                                country_name,
                                latest_date,
                                lstm_pred=death_pred,
                                obs_num=num_observations,
                                )

            st.subheader('Confirmed Cases Trend:')
            model.draw_single_trend(lstm_on=lstm_select)
            st.subheader('Death Cases Trend:')
            death_model.draw_single_trend(lstm_on=lstm_select)



@st.cache(allow_output_mutation=True)
def load_data(url):
    """Load confirm cases from github, and apply some preprocessing steps """
    df = pd.read_csv(url)
    df = df.groupby('Country/Region').sum().drop(['Lat', 'Long'], axis=1)
    df_trans = df.transpose()
    df_trans['World'] = df_trans.sum(axis=1)
    return df_trans


@st.cache(allow_output_mutation=True)
def select_top_country(df, top_num=10, days=7):
    col_name = 'Last_{}_Days'.format(days)
    df = pd.DataFrame((df.iloc[-1, :] - df.iloc[-1 * days, :]).sort_values(ascending=False).head(top_num),
                      columns=[col_name])
    df[col_name] = df[col_name].apply(lambda x: "{:,}".format(x))
    return df


def _top_cases_list(df, top_num=10):
    return list(df.iloc[-1, :-1].sort_values(ascending=False).index[:top_num])


def _top_cased_global(df, top_num=10):
    return list(df.iloc[-1, :].sort_values(ascending=False).index[:top_num])


@st.cache(allow_output_mutation=True)
def draw_trend(df, show_global,model,lstm_select, num_obv=50):
    '''Draw trend chart of top 10 countries'''
    if show_global:
        #top_cases = _top_cased_global(df)
        return model.draw_single_trend(return_chart=True, lstm_on=lstm_select)
    else:
        top_cases = _top_cased_global(df)
        df_plot = pd.DataFrame()
        localize = lambda x:"{:,}".format(int(x))
        for col in top_cases:
            temp = pd.DataFrame(df[col][-1 * num_obv:]).rename(columns={col: 'cases'})
            temp['Date'] = temp.index
            temp['Country'] = col
            temp['Cases'] = temp['cases'].apply(localize)
            df_plot = df_plot.append(temp, ignore_index=True)

        trend_chart = alt.Chart(df_plot.reset_index()).mark_line().encode(
            x="Date:T",
            y="cases:Q",
            color=alt.Color('Country', sort=top_cases),
            tooltip=['Country:N', "Date:T", "Cases"]
        ).properties(width=800, height=300).interactive()

        return trend_chart



class arima():
    """ARIMA class object"""
    def __init__(self, df, col_name, latest_date, obs_num,lstm_pred,order=(0,2,1), pred_num=30, train_num=200,):
        self.df = df
        self.col = col_name
        self.series = df[col_name]
        self.date = latest_date
        self.order = order
        self.obs_num = obs_num
        self.pred_num = pred_num
        self.train_num = train_num
        self.lstm_pred = lstm_pred
        best_cfg = load(open('arima_order.pkl', 'rb'))
        try:
            order = best_cfg[col_name]
        except:
            order = (0,2,1)
        self.model = ARIMA(self.series[-1 * train_num:], order=order)
        self.model_fit = self.model.fit(disp=0)

    def quick_fit_plot(self):
        """create streamlit plot object"""
        st.pyplot(self.model_fit.plot_predict(1, self.obs_num + self.pred_num))

    def plot_acf_pacf(self):
        """Auto correlation function plot  on streamlit object"""
        fig, axs = plt.subplots(2)
        plt.subplots_adjust(hspace=0.4)
        plot_acf(self.series[-1 * self.train_num:], ax=axs[0])
        plot_pacf(self.series[-1 * self.train_num:], ax=axs[1])
        st.pyplot(fig)

    def get_pred(self):
        trend = self.model_fit.forecast(self.pred_num)[0]
        conf_inv = self.model_fit.forecast(self.pred_num)[2]
        return trend,conf_inv

    def _create_df_plot(self, col_type ='Cases',arima_on=True,lstm_on=True):
        """create  plot data frame of prediction and confidence region  for altair plot"""
        localize = lambda x: "{:,}".format(round(x))
        # origin data
        temp = pd.DataFrame(self.series[-1 * self.obs_num:]).rename(columns={self.col: 'cases'})
        temp['Date'] = temp.index
        temp['Type'] = col_type
        temp['Cases'] = temp['cases'].apply(localize)
        # predictions
        if arima_on:
            line = pd.DataFrame(self.model_fit.forecast(self.pred_num)[0],
                                index=pd.date_range(start=self.date+dt.timedelta(days=1), periods=self.pred_num),
                                columns=['cases'])
            line['Date'] = line.index
            line['Type'] = 'ARIMA_pred'
            line['Cases'] = line['cases'].apply(localize)

            temp = temp.append(line, ignore_index=True)

        # Predictions2
        if lstm_on:
            lstm_line= pd.DataFrame(self.lstm_pred[self.col]).rename(columns={self.col: 'cases'})
            lstm_line['Date'] = lstm_line.index
            lstm_line['Type'] = 'LSTM_pred'
            lstm_line['Cases'] = lstm_line['cases'].apply(localize)

            temp = temp.append(lstm_line, ignore_index=True)

        # confidence region
        cl = pd.DataFrame(self.model_fit.forecast(self.pred_num)[2],
                          index=pd.date_range(start=self.date + dt.timedelta(days=1), periods=self.pred_num),
                          columns=['lower', 'upper'])
        cl['Date'] = cl.index

        return temp, cl


    def draw_single_trend(self, return_chart=False,country_name = 'Cases',arima_on=True,lstm_on=True):

        df_plot, df_cl = self._create_df_plot(country_name,lstm_on=lstm_on)
        trend_chart = alt.Chart(df_plot).mark_line().encode(
            x=alt.X("Date:T",scale=alt.Scale(zero=False)),
            y=alt.Y("cases:Q",scale=alt.Scale(zero=False)),
            color=alt.Color('Type',sort=[country_name,'ARIMA_pred',"LSTM_pred"]),
            strokeDash=alt.condition( alt.FieldOneOfPredicate(field='Type', oneOf=['ARIMA_pred','LSTM_pred']),
                #((alt.datum.Type == 'ARIMA_pred') or (alt.datum.Type == 'LSTM_pred')),
                                     alt.value([10, 5]),  # dashed line: 5 pixels  dash + 5 pixels space
                                     alt.value([0])
                                     ),
            tooltip=["Date:T", "Cases:O"]
        ).properties(width=800, height=300).interactive()

        band = alt.Chart(df_cl).mark_area(
                opacity=0.5, color='grey'
            ).encode(
                x=alt.X("Date:T",scale=alt.Scale(zero=False)),
                y=alt.Y('lower',title='cases'),
                y2=alt.Y2('upper',title='cases')
            ).properties(width=800, height=300).interactive()

        if return_chart:
            return band+trend_chart
        else:
            st.altair_chart(band+trend_chart)



@st.cache(allow_output_mutation=True)
def load_tf_models():
    # models = {'Linear': tf.keras.models.load_model("model_confirm_case/Linear_model"),
    #           'Dense': tf.keras.models.load_model("model_confirm_case/Dense_model"),
    #           'Conv': tf.keras.models.load_model("model_confirm_case/Conv_model"),
    #           'RNN': tf.keras.models.load_model("model_confirm_case/RNN_model"),
    #           'LSTM': tf.keras.models.load_model("model_confirm_case/LSTM_model")}
    models ={'LSTM': [tf.keras.models.load_model("lstm_model.h5"),
                      tf.keras.models.load_model("lstm_model_d.h5")]}
    return models



def get_preds(models,input,scaler):
    """get average prediction of 100 trials from deep learning model"""
    preds_conf={}
    preds_death={}
    for k,v in models.items():
        conf_result = np.stack([v[0](input[0],training =True).numpy() for sample in range(100)])
        death_result = np.stack([v[1](input[1], training=True).numpy() for sample in range(100)])
        preds_conf[k] = scaler[0].inverse_transform(conf_result.mean(axis=0).squeeze())
        preds_death[k] = scaler[1].inverse_transform(death_result.mean(axis=0).squeeze())

    return preds_conf,preds_death





if __name__ == '__main__':
    main()
