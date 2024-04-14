import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import catboost as cb
import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


@st.cache_data
def data_transform(df1: pd.DataFrame, df2: pd.DataFrame):
    df1['cum_sum'] = df1.sort_values('quarter').groupby(
        'client_id')['churn'].cumsum().shift(1).fillna(0)
    df1["not_churn"] = df1["churn"].apply(lambda x: 1 if x == 0 else 0)
    df1['cum_cnt'] = df1.sort_values('quarter').groupby(
        'client_id')['not_churn'].cumsum().shift(1).fillna(0)
    df1.drop(columns='not_churn', inplace=True)
    df1.fillna(0, inplace=True)
    result = pd.merge(df1, df2, left_on='npo_account_id',
                      right_on='npo_accnt_id')
    result['accnt_pnsn_schm'] = result['accnt_pnsn_schm'].astype('str')
    result.drop(columns=["client_id", "npo_account_id",
                "npo_accnt_id"], axis=1, inplace=True)
    last_q = '2022Q1'

    result = result[result['balance'] > 0]

    result['pml_check'] = (
        result['lst_pmnt_date_per_qrtr'] != 0).astype('bool')
    result['q_n'] = result['quarter'].apply(lambda x: x[-1]).astype('int')
    result['frst_pmnt_date'] = result['frst_pmnt_date'].astype('category')
    result['lst_pmnt_date_per_qrtr'] = result['lst_pmnt_date_per_qrtr'].astype(
        'category')
    result['region'] = result['region'].astype('category')
    result['accnt_pnsn_schm'] = result['accnt_pnsn_schm'].astype('category')
    result['rfm'] = np.log10(result['balance'] + 1) * \
        result['pmnts_nmbr_per_qrtr'] / (result['lst_pmnt_rcnc_d'] + 1)

    result_test = result[result['quarter'] > last_q]
    result_train = result[result['quarter'] <= last_q]
    result_test['quarter'] = result['quarter'].astype('category')
    result_train['quarter'] = result['quarter'].astype('category')

    result_train, result_val = np.split(
        result_train, [int(0.7*len(result_train))])

    X_test = result_test.drop('churn', axis=1)
    y_test = result_test['churn']

    return X_test, y_test


@st.cache_resource
def pie_graph(data: pd.DataFrame):
    df_frst_pmnt_date = data
    df_frst_pmnt_date['date'] = pd.to_datetime(
        df_frst_pmnt_date['frst_pmnt_date'])

    df_frst_pmnt_date['year'] = df_frst_pmnt_date['date'].dt.year
    df_frst_pmnt_date['month'] = df_frst_pmnt_date['date'].dt.month
    month_names = {1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель', 5: 'Май', 6: 'Июнь',
                   7: 'Июль', 8: 'Август', 9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'}
    df_frst_pmnt_date['month'] = df_frst_pmnt_date['month'].map(month_names)
    df_frst_pmnt_date['year'] = df_frst_pmnt_date['year'].astype(
        str).str.rstrip('.0')
    for column in ['year', 'month']:
        category_counts = df_frst_pmnt_date[column].value_counts()

        top_categories = category_counts.index[:10]
        top_counts = category_counts[:10]
        other_count = category_counts[10:].sum()

        labels = list(top_categories) + ['Прочие']
        values = list(top_counts) + [other_count]

        max_value_index = values.index(max(values))
        pull = [0.15 if i ==
                max_value_index else 0 for i in range(len(values))]

        fig = go.Figure(
            data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker=dict(line=dict(color='darkgrey', width=1.5)),
                pull=pull,
            )]
        )
        fig.update_layout(
            title_text=f'Соотношение  даты первого взноса с {
                "годом" if column == "year" else "месяцем"}',
            title_x=0.5,
            width=1000,
            height=700,
            legend=dict(x=0.8, y=0.5)
        )

    return fig


@st.cache_resource
def corr_graph(data: pd.DataFrame):

    translated_dict = {
        'slctn_nmbr': 'Номер выборки ',
        'npo_accnts_nmbr': 'ID счета клиента',
        'pmnts_type': 'Тип взносов',
        'year': 'Год агрегации признаков',
        'gender': 'Пол',
        'age': 'Возраст',
        'clnt_cprtn_time_d': 'время сотрудничества с клиентом в днях',
        'actv_prd_d': 'активный период в днях',
        'lst_pmnt_rcnc_d': 'последний платеж недавно в днях',
        'balance': 'баланс',
        'oprtn_sum_per_qrtr': 'сумма операций за квартал',
        'oprtn_sum_per_year': 'сумма операций за год',
        'frst_pmnt': 'первый платеж',
        'lst_pmnt': 'последний платеж',
        'pmnts_sum': 'сумма платежей',
        'pmnts_nmbr': 'количество платежей',
        'pmnts_sum_per_qrtr': 'сумма платежей за квартал',
        'pmnts_sum_per_year': 'сумма платежей за год',
        'pmnts_nmbr_per_qrtr': 'количество платежей за квартал',
        'pmnts_nmbr_per_year': 'количество платежей за год',
        'incm_sum': 'сумма дохода',
        'incm_per_qrtr': 'доход за квартал',
        'incm_per_year': 'доход за год',
        'mgd_accum_period': 'управляемый период накопления',
        'mgd_payment_period': 'управляемый период платежей',
        'lk': 'ссылка',
        'citizen': 'гражданин',
        'appl_mrkr': 'маркер приложения',
        'evry_qrtr_pmnt': 'платеж каждый квартал'
    }

    translated_lst = ['slctn_nmbr_x',
                      'Число счетов клиента',
                      'Тип взносов',
                      'Год',
                      'Пол',
                      'Возраст',
                      'Время «жизни» клиента в днях',
                      'Активный период в днях',
                      'Давность предыдущего взноса (в днях)',
                      'Баланс счета на конец квартала',
                      'Сумма операций по счету в квартале',
                      'Сумма операций по счету в году',
                      'Размер первого взноса',
                      'Размер последнего взноса',
                      'Сумма взносов на конец квартала',
                      'Число взносов на конец квартала',
                      'Сумма НПО взносов в квартале',
                      'Сумма НПО взносов в году',
                      'Число НПО взносов за квартал',
                      'Число НПО взносов за год',
                      'Сумма инвест-дохода на конец квартала',
                      'Сумма инвест-дохода за квартал ',
                      'Сумма инвест-дохода за год',
                      'Мин. гаран-ый доход на этапе накопления',
                      'Мин. гаран-ый доход на этапе выплат',
                      'Есть номер телефона? (Да/Нет)',
                      'Есть email? (Да/Нет)',
                      'Есть номер личного кабинета? (Да/Нет)',
                      'Клиент приемник НПО счета? (Да/Нет)',
                      'Клиент приемник ОПС счета? (Да/Нет)',
                      'Почтовый индекс',
                      'Клиент городской житель? (Да/Нет)',
                      'Указ-ый адрес - адрес факт-го прож-я? (Да/Нет)',
                      'Были ли жалобы у клиента? (Да/Нет)',
                      'Клиент платит каждый каждый квартал? (Да/Нет)',
                      'Клиент ушел? (Да/Нет)',
                      'slctn_nmbr_y']

    corr_matrix = data.corr(numeric_only=True)

    annotations = np.empty_like(corr_matrix.values, dtype=str)

    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=translated_lst,
        y=translated_lst,
        annotation_text=annotations,
        showscale=True,
        colorscale='Plasma'
    )

    fig.update_layout(
        title=dict(text="Матрица корреляций", x=0.5, y=1),
        width=1000,
        height=700,
        xaxis=dict(tickangle=45)
    )

    fig['data'][0]['xgap'] = 3
    fig['data'][0]['ygap'] = 3

    return fig


@st.cache_data
def data_merge(df1: pd.DataFrame, df2: pd.DataFrame):
    result = pd.merge(df1, df2, left_on='npo_account_id',
                      right_on='npo_accnt_id')
    result['accnt_pnsn_schm'] = result['accnt_pnsn_schm'].astype('str')
    result.drop(columns=["client_id", "npo_account_id",
                "npo_accnt_id"], axis=1, inplace=True)
    return result


@st.cache_data
def load_data(file_path: str, sep_char=','):
    data = pd.read_csv(file_path, sep=sep_char)
    return data


@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')


@st.cache_resource
def get_model(file_pah: str):
    model = joblib.load(file_pah)
    return model


def make_predictions(model, data):
    predictions = model.predict_proba(data)
    return predictions


st.title("WGhack")
st.title("Прогнозирования динамики лояльности клиентов")

uploaded_file1 = st.file_uploader(
    "Загрузите данные для прогноза", type=['csv'])
if uploaded_file1 is not None:
    data1 = load_data(uploaded_file1)

data2 = load_data('cntrbtrs.csv', ';')

if (uploaded_file1 is not None):
    button1 = st.button('Построить графики')
    if button1:
        data_plot = data_merge(data1, data2)

        tab1, tab2 = st.tabs(
            ["Матрица корреляции", "Распределние открытых счетов"])
        with tab1:
            st.plotly_chart(corr_graph(data_plot))
            st.markdown(
                f"<style>div.stPlotlyChart:nth-of-type(1){{margin-left: -200px; margin-top: 20px}}</style>", unsafe_allow_html=True)

        with tab2:
            st.plotly_chart(pie_graph(data_plot))

    button2 = st.button('Получить предсказания')

    if button2:
        X_test, y_test = data_transform(data1, data2)

        models_list = ['res_model_cat.pkl', 'res_model_cat2.pkl',
                       'res_model_lgbm.pkl', 'res_preds_model_xgb.pkl']

        cat1, cat2, lgbm, xgb = get_model(models_list[0]), get_model(
            models_list[1]), get_model(models_list[2]), get_model(models_list[3])

        preds_cat_proba = cat1.predict_proba(X_test)[:, 1]
        preds_cat_proba2 = cat2.predict_proba(X_test)[:, 1]
        preds_lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
        preds_xgb_proba = xgb.predict_proba(X_test.drop(
            columns=['quarter', 'frst_pmnt_date', 'lst_pmnt_date_per_qrtr', 'region', 'accnt_pnsn_schm']))[:, 1]

        preds = (((preds_cat_proba + preds_cat_proba2 +
                 preds_lgbm_proba + preds_xgb_proba) / 4) > .51).astype('int')
        st.table(classification_report(y_test, preds))
        df = pd.DataFrame(preds)
        st.dataframe(df)
        csv = convert_df(df)

        st.download_button(
            label="Скачать предсказания",
            data=csv,
            file_name='preds.csv',
        )
