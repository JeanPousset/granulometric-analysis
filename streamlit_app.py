"""
%> Web application for Granulometric (grain-size) data analysis developped during an IRMAR internship <% 

Author :
Jean Pousset (4th year Applied Maths INSA Rennes)  contact : pousset.jean1@gmail.com 
DON'T HESITATE TO CONTACT ME FOR HELP / MORE EXPLANATIONS

Internship report : https://raw.githubusercontent.com/JeanPousset/granulometric-analysis/main/IRMAR_report/Rapport_IRMAR_d%C3%A9composition_courbe_granulom%C3%A9triques_Jean_POUSSET.pdf

Guidance:
- Valérie Monbet (IRMAR) https://perso.univ-rennes1.fr/valerie.monbet/
- Fabrice Mahé (IRMAR) https://perso.univ-rennes1.fr/fabrice.mahe/

Data proveiders :
- François Pustoc'h (CreAAH)
- Simond Puaud (CreAAH) https://creaah.cnrs.fr/team/puaud-simon-1/ 

Code writer of the BLASSO results : Clément Elvira (IETR / SCEE) https://c-elvira.github.io/
"""

import numpy as np
import pandas as pd
import streamlit as st
import time
import json
import random
from sklearn.decomposition import NMF
import plotly.graph_objects as go
from functools import partial
import subprocess
import re
from plotly.subplots import make_subplots
import sys
import os
# from backends.numpy_functions import robust_nmf # --> for robust nmf algorithm (commented because we don't use this technique anymore)




# Streamlit and environemetn managment
sys.path.append("..")
st.set_page_config(page_title="jpousset : granulometric analysis", layout="wide")
st.title("Component identification on granulometric data")

# removing old export file
if  'clean_exports_flag' not in st.session_state:
    
    
    if os.name == 'nt': # special case of Windows OS
        subprocess.run("del /q exports\*", shell=True, check=True)
    else:               # other cases : Unix
        subprocess.run("rm -f exports/*", shell=True, check=True)
    st.session_state['clean_exports_flag'] = True


# Loading observation data :
if "granulometrics" not in st.session_state:
    data = pd.read_excel(
        "Data/data_granulometry_03_06_24.xlsx", sheet_name=0, header=0, index_col=2, engine='openpyxl'
    )
    # Deletion of additional information
    data = data.drop(columns=["Dept", "Commune", "Type"])
    data = data.diff(axis=1)  # Cumulative difference on line item
    # Dividing by the log division of mesurement steps
    data.loc[:, data.columns != 0.03] = data.loc[:, data.columns != 0.03].div(
        np.log10(
            np.array([float(col) for col in data.columns])[1:]
            / np.array([float(col) for col in data.columns])[:-1]
        ),
        axis=1,
    )
    data[[0.03]] = 0
    data = data.div(data.sum(axis=1), axis=0) * 100  # Norming curves
    st.session_state["granulometrics"] = data        # dataframe to use
    # dataframe to update the excel file (adding or removing observations)
    st.session_state["raw_data"] = pd.read_excel(
        "Data/data_granulometry_03_06_24.xlsx", sheet_name=0, header=0, engine='openpyxl')

# region initialisation of session variables
if 'result_exporation_available' not in st.session_state:
    st.session_state['result_exporation_available'] = False
if 'flag_other_dataset' not in st.session_state:
    st.session_state['flag_other_dataset'] = False
if 'cd_errors' not in st.session_state:
    st.session_state['cd_errors'] = pd.DataFrame(columns=["L1 relative error", "l2 error", "nb components"])
if 'nmf_errors' not in st.session_state:
    st.session_state['nmf_errors'] = pd.DataFrame(columns=["L1 relative error", "l2 error", "nb components"])
if 'dd_errors' not in st.session_state:
    st.session_state['dd_errors'] = pd.DataFrame(columns=["L1 relative error", "l2 error", "nb components"])
if  'Prop_nmf' not in st.session_state:
    st.session_state["Prop_nmf"] = pd.DataFrame()
if 'export_available_flag' not in st.session_state:
    st.session_state['export_available_flag'] = False
if 'sep_input' not in st.session_state:
    st.session_state['sep_input'] = 'tabulation'
if 'dec_input' not in st.session_state:
    st.session_state['dec_input'] = 'comma'
if "flag_comparaison_curves_importation" not in st.session_state:
    st.session_state["flag_comparaison_curves_importation"] = False
if "dd_flag" not in st.session_state:
    st.session_state["dd_flag"] = False
if "X-X_hat-X_ref" not in st.session_state:
    st.session_state["X-X_hat-X_ref"] = st.session_state["granulometrics"].copy()
    # to have the same columns as approximations
    # st.session_state['X-X_hat-X_ref']['L1_rel_norm'] = '-'
if "rc_flag" not in st.session_state:
    st.session_state["rc_flag"] = False
if "nmf_flag" not in st.session_state:
    st.session_state["nmf_flag"] = False
if "lambda_robust" not in st.session_state:
    st.session_state["lambda_robust"] = 1.0
if "beta_r" not in st.session_state:
    st.session_state["beta_r"] = 1.5
if "selected_label" not in st.session_state:
    st.session_state["selected_label"] = []

if "X-X_hat-X_ref" not in st.session_state:
    st.session_state["X-X_hat-X_ref"] = st.session_state["granulometrics"]
# endregion

# Loading reference curves
if "ref_curves" not in st.session_state:
    st.session_state["ref_curves"] = {}  # empty initialization
    st.session_state["ref_curves"]["ref_ArgilesFines"] = np.genfromtxt(
        "ref_curves_&_exemples/ref_ArgilesFines.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_ArgilesClassiques"] = np.genfromtxt(
        "ref_curves_&_exemples/ref_ArgilesClassiques.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_Alterites"] = np.genfromtxt(
        "ref_curves_&_exemples/ref_Alterites.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_SablesFins"] = np.genfromtxt(
        "ref_curves_&_exemples/ref_SablesFins.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_SablesGrossiers"] = np.genfromtxt(
        "ref_curves_&_exemples/ref_SablesGrossiers.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_Loess"] = np.genfromtxt(
        "ref_curves_&_exemples/ref_Loess.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_LimonsGrossiers"] = np.genfromtxt(
        "ref_curves_&_exemples/ref_LimonsGrossiers.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_Loess_without_residules"] = np.genfromtxt(
        "ref_curves_&_exemples/ref_LimonsGrossiersLoess.csv", delimiter=","
    )
    st.session_state["scaled_ref_curves"] = (
        {}
        # for ref curves that will be on the same scale as observations (for NN-LASSO)
    )
    # st.session_state['scaled_ref_curves']['abscisses'] = st.session_state["ref_curves"]["ref_ArgilesFines"][0,:] # --> abscisses not necessary
    st.session_state["scaled_ref_curves"]["Argiles Fines"] = (
        st.session_state["ref_curves"]["ref_ArgilesFines"][1, :] * 0.004
    )
    st.session_state["scaled_ref_curves"]["Argiles Classiques"] = (
        st.session_state["ref_curves"]["ref_ArgilesClassiques"][1, :] * 0.03
    )
    st.session_state["scaled_ref_curves"]["Limons fins"] = (
        st.session_state["ref_curves"]["ref_Alterites"][1, :] * 0.107063
    )
    # We prefer don't use raw Loess component curve
    # st.session_state["scaled_ref_curves"]["ref_Loess"] = (
    #     st.session_state["ref_curves"]["ref_Loess"][1, :] * 0
    # )
    st.session_state["scaled_ref_curves"]["Limons Grossiers"] = (
        st.session_state["ref_curves"]["ref_LimonsGrossiers"][1, :] * 0.11
    )
    st.session_state["scaled_ref_curves"]["Loess"] = (
        st.session_state["ref_curves"]["ref_Loess_without_residules"][1, :] * 0.06
    )
    st.session_state["scaled_ref_curves"]["Sables Fins"] = (
        st.session_state["ref_curves"]["ref_SablesFins"][1, :] * 0.1
    )
    st.session_state["scaled_ref_curves"]["Sables Grossiers"] = (
        st.session_state["ref_curves"]["ref_SablesGrossiers"][1, :] * 0.05
    )

# region Other variables / functions

# Integral approximations with trapeze method for every observation
def trapeze_areas(x):
    # adding log10 to abscisse to have equal error important over the whole abscisse axis
    return 0.5 * np.sum(
        (
            np.log10(st.session_state["granulometrics"].columns[1:].astype(float).to_numpy())
            - np.log10(st.session_state["granulometrics"].columns[:-1].astype(float).to_numpy())
        )
        * ((x[1:]) + (x[:-1]))
    )


def L1_relative(x_approx, obs_index):
    x_obs = st.session_state["granulometrics"].loc[obs_index].to_numpy()
    numerator = trapeze_areas(np.abs(x_approx - x_obs))
    denominator = trapeze_areas(x_obs)
    return numerator / denominator * 100


# Calculate quotient between ||x-x_approx||_L1 et ||x||L1


# list of labelized materials regarding the location of the peak
materials = {
    "Autre": 0.03,
    "Argile Fines": 1,
    "Argile Grossieres": 7,
    "Limons fins": 20,
    "Limons grossiers/Loess": 50,
    "Sables Fin": 100,
    "Sables Grossiers": 2400,
}
# endregion

# declaration of the web app tabs
tab_intro, tab_data, tab_continous_dict, tab_discrete_dict, tab_NMF, tab_result = st.tabs(
    [
        "**Introduction**",
        "**Data Managment**",
        "**Continuous dictionary**",
        "**Discrete dictionary**",
        "**Unsupervised**",
        "**Results**",
    ])

# 1st tab : explain the purpose of this app
with tab_intro:
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:
        st.markdown("<h2 style='text-align: center;'>Granulometric (grain-size) data</h2>", unsafe_allow_html=True)
        st.write(" ")
        st.page_link("https://raw.githubusercontent.com/JeanPousset/granulometric-analysis/main/IRMAR_report/Rapport_IRMAR_d%C3%A9composition_courbe_granulom%C3%A9triques_Jean_POUSSET.pdf", label="*Click here to read project report (more context and information)*", icon="📄")
        st.page_link("https://github.com/JeanPousset/granulometric-analysis", label="*Github reposistory (open source code)*", icon="🛠️")
        st.write(" ")
        st.write("""In a sample of raw earth we can find several types of components. These take the form of a 
                    more or less recognizable peak in a distribution curve. These components are classified 
                    according to the interval (abscissa) in which the peak is located :""")

        st.image("ref_curves_&_exemples/comt_inter.png")
        st.write("""The component with a peak between 20 and 50 microns (µm) 
                    is of the Coarse Alteration Silt or Loess type. Coarse silts are the result of the separation 
                    of various granular materials with larger grain sizes. Loess, on the other hand, is formed by 
                    dust deposits (silts) transported by the wind. This eolian origin makes the peak associated 
                    with loess more constricted than that of coarse silt. In fact, only grains of the same size 
                    will be transported to the sample location. Smaller (or larger) grains will be carried further 
                    (or closer) by the wind.""")
        st.write("""The difference between the two types lies in the origin of the silt (one by weathering, the 
                    other by eolian transport). The general designations are Limons grossiers d'altération and 
                    Limons grossier de Loess. In our study, we will simplify these terms to Limons grossiers and 
                    Loess.""")
        st.markdown("<h2 style='text-align: center;'>Purpose of this web-app</h2>", unsafe_allow_html=True)

        st.write("""The first objective is to break down the observed curves into linear combinations of unimodal 
                    (single-peak) curves, which will be assigned to a component according to the peak's location 
                    on the x-axis. This enables geologists to determine the components (and their proportions) 
                    in each sample.""")

        st.write("""A second objective is to discriminate between the two components with peaks between 20 and 50 
                    µm: Coarse Silts and Loess. The supersposition of the peak intervals for these two components 
                    complicates the analysis of the curves by geologists. We therefore need to establish rules and 
                    criteria for discriminating between Loess and coarse silt peaks.""")
        st.markdown("<h2 style='text-align: center;'>Web app presentation </h2>", unsafe_allow_html=True)


 

        st.markdown("""
        - ***Introduction*** :  explain the global purpose of the application
        - ***Data managment*** :  Allows you to add, remove or reset some granulometric sample but also reset the dataset.
        - ***Continuous dictionary, Discrete dictionary, Unsupervized*** :   These 3 tabs are dedicated to different methods 
                        / algorithms that you can use with several parameter values to decompose sample into uni-modal curves.
        - ***Results*** : Allows you to plot every sample with the approximations of the various methods. It also summarizes 
                            proportion of decompositions, approximation error metrics (various norm) and others statistics.
                            In the end you can also export your results in various file extensions.""")
        
        st.markdown("----")
        st.markdown("<h4 style='text-align: center;'>A bit of context : reference curves</h4>", unsafe_allow_html=True)

        st.write("""In a previous work, reference curves were constructed, representative of each component sought. We show 
                    them here to give you an idea of what we are looking for. If you want to see an example of granulometric sample 
                    you can easily plot some in the **Results** tab.""")


        # region ref curves plot
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=st.session_state["ref_curves"]["ref_ArgilesFines"][0, :],
                y=st.session_state["scaled_ref_curves"]["Argiles Fines"],
                mode="lines",
                name="Argiles Fines (<1 microns)",
            )
        )
        fig.update_xaxes(type="log", tickformat=".1f", dtick=1, showgrid=True)
        fig.update_layout(
            height=700,
            showlegend=True,
            legend={
        'font': {'size': 20},
        'orientation': "h",  
        'yanchor': "bottom", 
        'y': 4.02,           
        'xanchor': "center",  
        'x': 0.5             
    },
            xaxis={'title': {'text': "grain size (micrometers, log-scale)", 'font': {'size': 20}}}
        )
        fig.update_traces(
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>")

        fig.add_trace(
            go.Scatter(
                x=st.session_state["ref_curves"]["ref_ArgilesClassiques"][0, :],
                y=st.session_state["scaled_ref_curves"]["Argiles Classiques"],
                mode="lines",
                name="Argiles Grossieres (1-7 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=st.session_state["ref_curves"]["ref_Alterites"][0, :],
                y=st.session_state["scaled_ref_curves"]["Limons fins"],
                mode="lines",
                name="Limons fins (7-20 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=st.session_state["ref_curves"]["ref_SablesFins"][0, :],
                y=st.session_state["scaled_ref_curves"]["Sables Fins"],
                mode="lines",
                name="Sables Fins (50-100 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=st.session_state["ref_curves"]["ref_SablesGrossiers"][0, :],
                y=st.session_state["scaled_ref_curves"]["Sables Grossiers"],
                mode="lines",
                name="Sables Grossiers (>100 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=st.session_state["ref_curves"]["ref_LimonsGrossiers"][0, :],
                y=st.session_state["scaled_ref_curves"]["Limons Grossiers"],
                mode="lines",
                name="Limons Grossiers (20-50 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=st.session_state["ref_curves"]["ref_Loess_without_residules"][0, :],
                y=st.session_state["scaled_ref_curves"]["Loess"],
                mode="lines",
                name="Loess (20-50 microns)",
            )
        )
        st.plotly_chart(fig)
        # endregion


with tab_data:

    st.markdown("<h3 style='text-align: center;'>Add new observation</h3>", unsafe_allow_html=True)
    with st.form(key='input_obs_form'):
        col1, col2 = st.columns(2)
        with col1:
            st.radio("**Separator**",
                        options=['**⇥**', '**␣**', '**,**', '**;**'],
                        captions=['tabulation', 'space',
                                'comma', 'semicolon'],
                        index=0,
                        key=st.session_state['sep_input'])
        with col2:
            st.radio("**Decimal**",
                        options=['**,**', '**.**'],
                        captions=['comma', 'dot'],
                        index=0,
                        key=st.session_state['dec_input'])
        # Utiliser un textarea pour plus de commodité
        input_data = st.text_area(
            'Raw data (with metadata), separated by tabulations :', height=150)

        col1, col2 = st.columns([9, 1])
        with col2:
            submit_button = st.form_submit_button(label='Add')

        if submit_button:
            # dict to translate sep option into ASCII symbol
            sep_dict = {
                'tabulation': '\t',
                'space': ' ',
                'comma': ',',
                'semicolon': ';'
            }
            dec_dict = {
                'comma': ',',
                'dot': '.'
            }
            df_input = pd.DataFrame(
                columns=st.session_state['raw_data'].columns)
            lines = input_data.strip().split('\n')

            for line in lines:
                row = line.split(
                    sep_dict[st.session_state['sep_input']])
                row[4:] = [float(val.replace(dec_dict[st.session_state['dec_input']], '.'))
                            # convert data point into float
                            for val in row[4:]]
                # adding the line (copy maybe useless)
                df_input.loc[len(df_input)] = row.copy()

            st.session_state['raw_data'] = pd.concat(
                [st.session_state['raw_data'], df_input])
            st.dataframe(st.session_state['raw_data'])
            st.session_state['raw_data'].to_excel(
                "Data/data_granulometry_03_06_24.xlsx", sheet_name='Feuil1', index=False)
            st.success(
                'Data loaded, please reload the page to save changes')

    st.markdown("---")
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:
        st.markdown("<h3 style='text-align: center;'>Remove observation(s)</h3>", unsafe_allow_html=True)

        st.subheader("")
        st.markdown(
            "Choose which label to remove and then click on \"Confirm\". Please reload the page to save change !")
        col1, col2 = st.columns(2)

        with col1:
            st.multiselect(" ", options=st.session_state["granulometrics"].index, key="labels_to_remove", label_visibility='collapsed')

        with col2:
            if st.button("Confirm"):
                # Select observation execpt those to be removed
                st.session_state['raw_data'] = st.session_state['raw_data'][~st.session_state['raw_data']['Echt'].isin(
                    st.session_state['labels_to_remove'])]
                # Update the excel file
                st.session_state['raw_data'].to_excel(
                    "Data/data_granulometry_03_06_24.xlsx", sheet_name='Feuil1', index=False)
                st.success("Removing asked, now reload the page")
                st.dataframe(st.session_state['raw_data'])

        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Data exportation</h3>", unsafe_allow_html=True)
        st.toggle("Export all", value=True, key='flag_output_all')
        with st.form(key='output_obs_form'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text_input("Enter file name *(without extension)*",
                                key='output_file_name', value="granulometric_data")
            with col2:
                st.radio("File format", options=["**Excel (.xlsx)**", "CSV (.csv)"], key='output_ext',
                            help="In .xlsx, float decimal is comma **,** . In .csv, float decimal is dot **.**")
            with col3:
                st.radio("data format", options=["**Cumulative**", "**Distributive**"], key='output_data_format',
                            help="Shape of observation curves : cumulative (raw) or disrtibutive (transformation)")

            # case with only few obs
            if not st.session_state['flag_output_all']:
                st.multiselect("Select observation to export :",
                                options=st.session_state['granulometrics'].index, key='output_labels')

            col1, col2 = st.columns([6, 1])
            with col2:
                submit_button_export = st.form_submit_button(
                    label='Export')

            if submit_button_export:

                if st.session_state['output_data_format'] == "**Cumulative**":
                    # cumulative output
                    df_export = pd.read_excel(
                        "Data/data_granulometry_03_06_24.xlsx", sheet_name=0, header=0, index_col=2, engine='openpyxl')
                else:  # distributive (transformation)
                    df_export = st.session_state['granulometrics']

                # export all
                if st.session_state['flag_output_all']:
                    if st.session_state['output_ext'] == "**Excel (.xlsx)**":
                        df_export.to_excel(
                            "exports/"+st.session_state['output_file_name']+'.xlsx', sheet_name='Feuil1', index=False)
                    else:
                        df_export.to_csv(
                            "exports/"+st.session_state['output_file_name']+'.csv', float_format='%.4f', index=False)
                # case with only few label
                else:
                    if st.session_state['output_ext'] == "**Excel (.xlsx)**":
                        df_export.loc[st.session_state['output_labels']].to_excel(
                            "exports/"+st.session_state['output_file_name']+'.xlsx', sheet_name='Feuil1', index=False)
                    else:
                        df_export.loc[st.session_state['output_labels']].to_csv(
                            "exports/"+st.session_state['output_file_name']+'.csv', float_format='%.4f', index=False)

                # session variable to handle download button next
                st.session_state['export_available_flag'] = True
                st.session_state['name_export_file'] = st.session_state['output_file_name']
                if st.session_state['output_ext'] == "**Excel (.xlsx)**":
                    st.session_state['export_file_ext'] = '.xlsx'
                else:
                    st.session_state['export_file_ext'] = '.csv'

                st.success("Export file created")

        def get_export_file(file_name):
            with open(file_name, "rb") as file:
                return file.read()

        if st.session_state['export_available_flag'] == True:
            export_file_name = st.session_state['name_export_file'] + \
                st.session_state['export_file_ext']
            if st.session_state['export_file_ext'] == '.xlsx':
                st.download_button("Download export file (.xlsx)", data=get_export_file(
                    "exports/"+export_file_name), mime="application/octet-stream", file_name=export_file_name)
            else:
                st.download_button("Download export file (.csv)", data=get_export_file(
                    "exports/"+export_file_name), mime="text/csv", file_name=export_file_name)

        st.markdown("---")
        st.markdown("<h3 style='text-align: center; color: red'>Dataset reset</h3>", unsafe_allow_html=True)
        st.markdown("""This button allows you to reset granulometrics data to the original 
                in case of error when updating data. """)
        st.warning(
            "⚠️ Are you sure ? You will not be able to retrieve changes made to the database ⚠️")
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Confirm reset"):
                df_save = pd.read_excel(
                    "Data/data_save_for_reset.xlsx", sheet_name=0, header=0, engine='openpyxl')
                df_save.to_excel(
                    "Data/data_granulometry_03_06_24.xlsx", sheet_name='Feuil1', index=False)
                st.success(
                    "Database reset made, please reload the page to apply changes.")


with tab_continous_dict:
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:
        st.markdown("<h1 style='text-align: center;'>Continuous dictionary</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h3 style='text-align: center;'>Import results</h3>", unsafe_allow_html=True)
        st.markdown("""For copyright and implementation reasons, we cannot run algorithms for Blasso on this web application. 
                    However, we can retrieve pre-calculated results for our basic data. Please select the $\\lambda$ of the results to import :""")
        
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("**Select dictionnary**",
                options = [
                    "None", 
                    "Gaussian 1 param (variance : 0.05)",
                    "Gaussian 2 param",
                    "Skew gaussian 1 param ()",
                    "Skew gaussian 2 param (shape fixed)",
                ],
                key = 'Blasso_dict', index = 4)
        with col2:
            st.selectbox("**Choose λ**", options = ['None','1e-5','1e-4','0.001','0.005','0.01','0.05','0.1','0.5','1','2','5','10'], key = 'Blasso_λ', index = 2)
                          

        if st.session_state['Blasso_λ'] != 'None':
        
            if st.session_state['Blasso_dict'] == "Gaussian 1 param (variance : 0.05)":
                res_first_name = "blass_res_"
                prop_first_name = "prop_BLASSO_"
            
            if st.session_state['Blasso_dict'] == "Gaussian 2 param":
                res_first_name = "blasso_res_gauss2p_"
                prop_first_name = "prop_BLASSO_gauss2p_"
            
            if st.session_state['Blasso_dict'] == "Skew gaussian 2 param (shape fixed)":
                res_first_name = "blasso_res_skewgaussian2p_"
                prop_first_name = "prop_BLASSO_skewgaussian2p_"

            import_directory = "B-LASSO_imports/"
            # Select which approx to use
            if st.session_state['Blasso_λ'] == '10':
                csv_name = res_first_name+"lambda10.csv"
                json_name = prop_first_name+"lambda10.json"
            if st.session_state['Blasso_λ'] == '5':
                csv_name = res_first_name+"lambda5.csv"
                json_name = prop_first_name+"lambda5.json"
            if st.session_state['Blasso_λ'] == '2':
                csv_name = res_first_name+"lambda2.csv"
                json_name = prop_first_name+"lambda2.json"
            if st.session_state['Blasso_λ'] == '1':
                csv_name = res_first_name+"lambda1.csv"
                json_name = prop_first_name+"lambda1.json"
            if st.session_state['Blasso_λ'] == '0.5':
                csv_name = res_first_name+"lambda05.csv"
                json_name = prop_first_name+"lambda05.json"
            if st.session_state['Blasso_λ'] == '0.1':
                csv_name = res_first_name+"lambda01.csv"
                json_name = prop_first_name+"lambda01.json"
            if st.session_state['Blasso_λ'] == '0.05':
                csv_name = res_first_name+"lambda005.csv"
                json_name = prop_first_name+"lambda005.json"
            if st.session_state['Blasso_λ'] == '0.01':
                csv_name = res_first_name+"lambda001.csv"
                json_name = prop_first_name+"lambda001.json"
            if st.session_state['Blasso_λ'] == '0.005':
                csv_name = res_first_name+"lambda0005.csv"
                json_name = prop_first_name+"lambda0005.json"
            if st.session_state['Blasso_λ'] == '0.001':
                csv_name = res_first_name+"lambda0001.csv"
                json_name = prop_first_name+"lambda0001.json"
            if st.session_state['Blasso_λ'] == '1e-4':
                csv_name = res_first_name+"lambda_1e-4.csv"
                json_name = prop_first_name+"lambda_1e-4.json"
            if st.session_state['Blasso_λ'] == '1e-5':
                csv_name = res_first_name+"lambda_1e-5.csv"
                json_name = prop_first_name+"lambda_1e-5.json"

            # Importing approx -> throw an exception if the file hasn't been computed 
            try:
                blasso_approx = pd.read_csv(import_directory+csv_name, index_col = 0)
                with open(import_directory+json_name, 'r') as file:
                    st.session_state['blasso_Prop'] = json.load(file)
            except Exception as e:
                st.error("⚠️⚠️⚠️ There is no results available for this λ with this dictionary. Select another dictionary or an other λ ⚠️⚠️⚠️")
    
            st.session_state['cd_flag'] = True
            for label, approx in blasso_approx.iterrows():

                # saving approx to use in results section
                st.session_state["X-X_hat-X_ref"].loc[f"[CD]-{label}"] = np.array(approx)
                # errors :
                sample = st.session_state['granulometrics'].loc[label]
                l2 = np.linalg.norm(np.array(approx)-np.array(sample), 2)
                L1_rel = L1_relative(np.array(approx), label)
                nb_comp = int(len(st.session_state['blasso_Prop'][label]))
                new_row = {"L1 relative error": L1_rel,"l2 error":  l2, "nb components": nb_comp}
                st.session_state["cd_errors"].loc[label] = new_row
            
            
            st.session_state['L1_mean_cd'] = st.session_state['cd_errors']["L1 relative error"].mean()
            st.session_state['l2_mean_cd'] = st.session_state['cd_errors']["l2 error"].mean()

        else:
            st.session_state['cd_flag'] = False


        st.markdown("----")
        # region comparaison with dd
        st.subheader("Graphique exemple of the method's interest")

        st.markdown(
            """ In order to show the interest of the B-Lasso method we're gonna try to decompose a signal that is made of two gaussians."""
        )

        st.markdown(
            """ The first plot is the best approximation that is possible if we use a Discrete Dictionary made
                    by replicating curve and translate them by step : $\\Delta = 1$. We can see that the approximation can't
                    overlap the observation because of because of this non-continuity."""
        )

        st.markdown(
            """On the other hand in the second plot we can see that the B-Lasso approximation is perfect."""
        )

        if not st.session_state["flag_comparaison_curves_importation"]:

            with open("ref_curves_&_exemples/ex_continuous_adv.json", "r") as file:
                json_data = json.load(file)

            st.session_state["comp_t"] = json_data["abscisses"]
            st.session_state["comp_y"] = json_data["y"]
            st.session_state["comp_hat_continuous"] = json_data["y_hat_continuous"]
            st.session_state["comp_hat_discrete"] = json_data["y_hat_discrete"]
            st.session_state["flag_comparaison_curves_importation"] = True

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=st.session_state["comp_t"],
                y=st.session_state["comp_y"],
                mode="lines",
                line=dict(color="red", width=3, dash="dot"),
                name="Observation",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=st.session_state["comp_t"],
                y=st.session_state["comp_hat_discrete"],
                mode="lines",
                line=dict(color="green", width=3),
                name="Discrete appoximation",
            )
        )
        fig.update_xaxes(tickformat=".0", dtick=1, showgrid=True)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(
            title="Lasso on a Discrete Dictionary",
            height=500,
            width=700,
        )
        fig.update_traces(
            hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>")
        st.plotly_chart(fig)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=st.session_state["comp_t"],
                y=st.session_state["comp_hat_continuous"],
                mode="lines",
                line=dict(color="lightblue", width=5),
                name="Blasso appoximation",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=st.session_state["comp_t"],
                y=st.session_state["comp_y"],
                mode="lines",
                line=dict(color="red", width=3, dash="dot"),
                name="Observation",
            )
        )
        fig.update_xaxes(tickformat=".0", dtick=1, showgrid=True)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(
            title="B-Lasso",
            height=500,
            width=700,
            # showlegend=False,
        )
        fig.update_traces(
            hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>")
        st.plotly_chart(fig)
        # endregion


with tab_discrete_dict:
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:

        # region input param

        st.markdown("<h1 style='text-align: center;'>Discrete Dictionary</h1>", unsafe_allow_html=True)
        st.markdown("---")                

        st.markdown(
            r"""In this section we try do decompose our observations with a Discrete Dictionary 
                    of unimodale curves $(\mathcal{M})$ obtained by duplicating the reference curves and shifting the spike
                     locations. To obtain these decomposition we're going to resolve the following optimization 
                    problem for each observation $x$ :"""
        )
        st.latex(
            r"""\arg \min_{a \in \mathbb{R}_+^{\vert \mathcal{M} \vert}} \frac{1}{2}\Vert x-\mathcal{M}a \Vert_2^2 + \lambda\Vert a\Vert_1"""
        )

        st.markdown(
            r"""Where $\mathcal{M}$ the dictionnary is presentend as a matrix where each column contains an unimodal curve.
                    We reconize here a LASSO problem except that the variable $a$ is non-negative. For this kind of problem (NN-LASSO), there
                     are several methods we can use. Please select bellow wich one you want."""
        )
        st.selectbox(
            label="Choose the resolution method",
            options=[
                "NN Frank-Wolfe",
                "NN gready algo",
                "FISTA with backtracking",
                "Proximal Gradient with constant step-size",
                "Projected gradient",
            ],
            key="nn_lasso_method",
        )

        col2, col3, col4 = st.columns([1.8, 1.2, 2])

        with col2:
            st.number_input(
                "Coefficient of penalization (lambda)",
                key="lambda_nn_lasso",
                value=2.3,
                min_value=0.0,
                step=0.5,
            )
        with col3:
            st.number_input("Precision for dual gap", key="p_dg", value=0.5)
        with col4:
            st.number_input(
                "Precision for complementary slackness", key="p_cs", value=0.5
            )

        # endregion

        if st.button("Run decomposition"):

            # region Creation of the Discrete Dictionary
            mesurement_points = st.session_state["granulometrics"].columns
            st.session_state["discrete_dictionnary"] = pd.DataFrame(
                columns=mesurement_points
            )

            for rc_name, rc in st.session_state["scaled_ref_curves"].items():

                # Find the first and last index of the measurement points of the peak-interval for the material of the reference curve
                peak_ind = np.argmax(rc[:])
                peak_loc = mesurement_points[peak_ind]
                mat_rc = (
                    ""  # name of the material for the interval of our ref-curve's peak
                )
                first_ind = 0
                last_ind = 0
                mat_keys = list(materials.keys())
                for mat_prec, mat in zip(mat_keys, mat_keys[1:]):
                    if peak_loc < materials[mat]:
                        mat_rc = mat
                        # index of the first mesurement point for the interval of this rc
                        first_ind = mesurement_points.get_loc(
                            materials[mat_prec])
                        # index of the last mesurement point for the interval of this rc
                        last_ind = mesurement_points.get_loc(
                            materials[mat]) - 1
                        rel_peak_ind = peak_ind - first_ind
                        rel_last_ind = last_ind - first_ind
                        break

                for i in range(rel_peak_ind + 1):
                    # Shifting ref curve to the left by dropping the i first values and adding i zeros at the end
                    duplicate = np.pad(
                        rc[i:], (0, i), mode="constant", constant_values=0
                    )
                    st.session_state["discrete_dictionnary"].loc[
                        f"{rc_name} ({mesurement_points[peak_ind-i]})"
                    ] = duplicate

                for i in range(1, rel_last_ind - rel_peak_ind + 1):
                    # Shifting ref curve to the right by dropping the i last values and adding i zeros at the begining
                    duplicate = np.pad(
                        rc[:-i], (i, 0), mode="constant", constant_values=0
                    )
                    # st.write(peak_ind+i)
                    st.session_state["discrete_dictionnary"].loc[
                        f"{rc_name} ({mesurement_points[peak_ind+i]})"
                    ] = duplicate

                # Precompute areas of curves in order to calculate proportions after
                st.session_state["aeras_dd_curves"] = np.zeros(
                    st.session_state["discrete_dictionnary"].shape[0]
                )
                for i in range(st.session_state["discrete_dictionnary"].shape[0]):
                    st.session_state["aeras_dd_curves"][i] = trapeze_areas(
                        st.session_state["discrete_dictionnary"].iloc[i].to_numpy(
                        )
                    )

            # endregion

            # region functions

            M = np.transpose(
                st.session_state["discrete_dictionnary"].to_numpy())
            
            # hyper-parameters
            it_max = 2e3
            MtM = np.dot(M.T, M)  # saving result to optimize
            eta = 2
            # Lipschitz constant of our objective function
            L = 2 * np.real(np.max(np.linalg.eigvals(MtM)))

            # Objective function
            def f_global(a, x_, lambda_):
                return 0.5 * np.linalg.norm(
                    x_ - np.dot(M, a), 2
                ) ** 2 + lambda_ * np.linalg.norm(a, 1)

            def h(a, x):
                return 0.5 * np.linalg.norm(x - np.dot(M, a), 2) ** 2

            # hat objective function
            def f_hat(a1, a, x, lambda_):
                # return 0.5*np.linalg.norm(x-np.dot(M,a),2)**2+np.dot(a1-a,np.dot(MtM,a)-np.dot(M.T,x))+1/(2*lambda_)*np.linalg.norm(a1-a,2)**2
                return (
                    f_global(a, x, lambda_)
                    + np.dot(np.dot(MtM, a) - np.dot(M.T, x), a1 - a)
                    # + 1 / (2 * lambda_) * np.linalg.norm(a1 - a, 2) ** 2
                )

            # Stop-criterions functions
            def rho(a, x, lambda_):
                return (
                    lambda_
                    * (x - np.dot(M, a))
                    / np.max(np.abs(np.dot(M.T, x) - np.dot(MtM, a)))
                )

            def CS(a, u, lambda_):
                return lambda_ * np.linalg.norm(a, 1) - np.dot(np.dot(M.T, u), a)

            def DG(a, x, u, lambda_):
                return (
                    f_global(a, x, lambda_) + 0.5 * (np.linalg.norm(x -
                                                                    u, 2) ** 2 - np.linalg.norm(x, 2) ** 2)
                )

            def stop_criterions(a, x, lambda_):
                u = rho(a, x, lambda_)
                cs = CS(a, u, lambda_)
                dg = DG(a, x, u, lambda_)
                # st.write(f"cs : {cs}")
                # st.write(f"dg : {dg}")
                return (
                    dg <= st.session_state["p_dg"]
                    and cs <= st.session_state["p_cs"]
                )

            # Non-negative least square with projected gradient method
            def NN_LS_proj_grad(Z, x_obs):
                a_ls = np.ones(Z.shape[1])
                prec_LS = 1e-3
                it_max_LS = 1e4
                err = prec_LS+1
                it = 0
                ZtZ = np.dot(Z.T, Z)
                Zx = np.dot(Z.T, x_obs)
                # 1 / Lipschitz constant of Z
                rho_LS = 1 / (2 * np.real(np.max(np.linalg.eigvals(ZtZ))))

                while err > prec_LS and it < it_max_LS:
                    a_ls_1 = np.maximum(
                        0.0, a_ls - rho_LS * (np.dot(ZtZ, a_ls)-Zx))
                    err = np.linalg.norm(a_ls_1-a_ls)
                    a_ls = a_ls_1
                    it += 1

                if it == it_max_LS:
                    st.warning(
                        'Non convergence of NN-LS for approximation reconstruction ')

                return a_ls, it

            # Reconstruction of the observation with least square problem to avoid bias due to l1 penality
            def reconstruction_LS(a, x_obs):

                # construction of Z matix in the ||x-Zc||^2 minimisation
                Z = M[:, a > 0.0]
                if Z.shape[1] == 0:
                    # case of empty solution
                    return a, np.zeros_like(x_obs), 0
                # resolving least-square problem (a_tmp is a small vector)
                a_tmp, it_ls = NN_LS_proj_grad(Z, x_obs)
                # approximation construction
                approx_ls = np.dot(Z, a_tmp)
                # spare vector, usefull to label our reconstruction
                a_ls = np.zeros(a.shape)
                k = 0                                      #
                for i in range(len(a_ls)):
                    if a[i] > 0.0:
                        a_ls[i] = a_tmp[k]
                        if a_tmp[k] < 0.0:
                            st.error(
                                f"Warning !!!! : ls reconstruction produced negativ coefficient {a_tmp[k]=}")
                        k += 1

                return a_ls, approx_ls, it_ls

            # endregion

            # region algorithms

            if st.session_state["nn_lasso_method"] == "NN gready algo":

                def decomposition_algo(x, lambda_):

                    # initialization
                    a0 = np.zeros(M.shape[1])
                    a0[random.randint(0, len(a0)-1)] = lambda_
                    a = a0
                    # a = np.zeros(M.shape[1])
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    it = 0

                    for i in range(M.shape[1]):

                        # STEP 1
                        j_star = np.argmin(np.dot(MtM, a)-Mx)
                        a_1 = np.zeros(a.shape)
                        a_1[j_star] = 1

                        # STEP 2
                        q = (lambda_ * a_1 - a)
                        if np.linalg.norm(q) == 0:
                            Γ = 0   # case where λ*a_1 = a
                        Γ = - np.dot(q, a) / (np.linalg.norm(q, 2) ** 2)
                        if Γ < 0:
                            Γ_t = 0
                        elif Γ > 1:
                            Γ_t = 1
                        else:
                            Γ_t = Γ

                        # STEP 3
                        a = a + Γ_t * q

                        it += 1

                    if it == it_max:
                        st.warning("Non-convergence for NN gready method")

                    # reconstruction with least-square problem to cancel bias
                    a_ls, approx_ls, it_ls = reconstruction_LS(a, x)
                    # argmin, approx, and nb of iterations
                    return a_ls, approx_ls.flatten(), it, it_ls

            if st.session_state["nn_lasso_method"] == "FISTA with backtracking":

                def decomposition_algo(x, lambda_):

                    # Initialization
                    a = np.zeros(M.shape[1])
                    z = a
                    it = 0
                    Li = L  # Lipschitz constant of ||x-Ma||2
                    eta = 2
                    t = 1

                    Mx = np.dot(M.T, x).reshape(a.shape)
                    f = partial(f_global, x_=x, lambda_=lambda_)

                    def q(a, z, l):
                        return (
                            f(z)
                            + np.dot(a - z, np.dot(MtM, z) - Mx)
                            + 0.5 * l * np.linalg.norm(a - z, 2) ** 2
                            + lambda_ * np.linalg.norm(a, 1)
                        )

                    # Fonction of the l1_prox of a-gradh(a)
                    # also the argmin of the approximation of F(x) at the given point y
                    def p_L(a, l):
                        z = a - (np.dot(MtM, a) - Mx) / l
                        res_neg = np.sign(z) * np.maximum(
                            np.abs(z) - np.full(z.shape, lambda_ / l),
                            np.full(z.shape, 0),
                        )
                        return np.maximum(res_neg, 0.0)

                    while not stop_criterions(a, x, lambda_) and it < it_max:
                        a1 = p_L(z, Li)
                        while f(a1) > q(a1, z, Li):
                            Li = eta * Li
                            a1 = p_L(z, Li)
                        t1 = 0.5 * (1 + np.sqrt(1 + 4 * (t**2)))
                        z = a1 + (t - 1) / t1 * (a1 - a)
                        a = a1  # update of a
                        it += 1

                    if it == it_max:
                        st.warning(
                            "Non-convergence for projected gradient method")

                    # reconstruction with least-square problem to cancel bias
                    a_ls, approx_ls, it_ls = reconstruction_LS(a, x)
                    # argmin, approx, and nb of iterations
                    return a_ls, approx_ls.flatten(), it, it_ls

            if st.session_state['nn_lasso_method'] == "NN Frank-Wolfe":
                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])
                    w = np.linalg.norm(x, 2) ** 2 / (2 * lambda_)
                    w_bar = w
                    it = 0

                    f = partial(f_global, x_=x, lambda_=lambda_)

                    # avoid to compute it every time
                    Mx = np.dot(M.T, x).reshape(a.shape)

                    # case where || M^t x ||_infty <= lambda
                    if np.max(np.abs(Mx)) <= lambda_:
                        return a, np.zeros(x.shape), it

                    while not stop_criterions(a, x, lambda_) and it < it_max:

                        # abbrevations
                        i_star = np.argmax(Mx - np.dot(MtM, a))
                        m_i_star = M[:, i_star]
                        r = x - np.dot(M, a)

                        # STEP 1:
                        if np.max(np.abs(Mx - np.dot(MtM, a))) <= lambda_:
                            a_pre = np.zeros(a.shape)
                            w_pre = 0
                        else:
                            canonic_vec = np.zeros(a.shape)
                            canonic_vec[i_star] = 1
                            a_pre = canonic_vec * w_bar
                            w_pre = w_bar

                        # STEP2:
                        v = m_i_star * w_pre - np.dot(
                            M, a
                        )
                        gamma_pre = (np.dot(v, r) + lambda_ * (w - w_pre)) / (
                            np.linalg.norm(v, 2) ** 2
                        )
                        if gamma_pre < 0:
                            gamma = 0
                        elif gamma_pre > 1:
                            gamma = 1
                        else:
                            gamma = gamma_pre

                        # STEP3:
                        a = gamma * a_pre + (1 - gamma) * a
                        w = gamma * w_pre + (1 - gamma) * w
                        w_bar = np.min([w_bar, f(a) / lambda_])

                        it += 1

                    if it == it_max:
                        st.warning("Non-convergence (Frank-Wolfe method) for one observation")

                    # reconstruction with least-square problem to cancel bias
                    a_ls, approx_ls, it_ls = reconstruction_LS(a, x)
                    # argmin, approx, and nb of iterations
                    return a_ls, approx_ls.flatten(), it, it_ls

            if st.session_state["nn_lasso_method"] == "NN FW (proj with max 0)":

                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])
                    w = np.linalg.norm(x, 2) ** 2 / (2 * lambda_)
                    w_bar = w
                    it = 0

                    f = partial(f_global, x_=x, lambda_=lambda_)

                    # avoid to compute it every time
                    Mx = np.dot(M.T, x).reshape(a.shape)

                    # case where || M^t x ||_infty <= lambda
                    if np.max(np.abs(Mx)) <= lambda_:
                        return a, np.zeros(x.shape), it

                    while not stop_criterions(a, x, lambda_) and it < it_max:

                        # abbrevations
                        i_star = np.argmax(Mx - np.dot(MtM, a))
                        m_i_star = M[:, i_star]
                        r = x - np.dot(M, a)

                        # STEP 1:
                        if np.sum(Mx - np.dot(MtM, a)) <= lambda_:
                            a_pre = np.zeros(a.shape)
                            w_pre = 0
                        else:
                            canonic_vec = np.zeros(a.shape)
                            canonic_vec[i_star] = 1
                            if not np.array_equal(np.maximum(canonic_vec * np.sign(np.dot(m_i_star, r)) * w_bar, 0), canonic_vec * np.sign(np.dot(m_i_star, r)) * w_bar):
                                st.write(" Difference")
                            a_pre = np.maximum(
                                canonic_vec * np.sign(np.dot(m_i_star, r)) * w_bar, 0)
                            w_pre = w_bar

                        # STEP2:
                        v = m_i_star * np.sign(np.dot(m_i_star, r)) * w_pre - np.dot(
                            M, a
                        )
                        gamma_pre = (np.dot(v, r) + lambda_ * (w - w_pre)) / (
                            np.linalg.norm(v, 2) ** 2
                        )
                        if gamma_pre < 0:
                            gamma = 0
                        elif gamma_pre > 1:
                            gamma = 1
                        else:
                            gamma = gamma_pre

                        # STEP3:
                        a = gamma * a_pre + (1 - gamma) * a
                        w = gamma * w_pre + (1 - gamma) * w
                        w_bar = np.min([w_bar, f(a) / lambda_])

                        it += 1

                    if it == it_max:
                        st.warning("Non-convergence for Frank-Wolfe method")

                    # reconstruction with least-square problem to cancel bias
                    a_ls, approx_ls, it_ls = reconstruction_LS(a, x)
                    # argmin, approx, and nb of iterations
                    return a_ls, approx_ls.flatten(), it, it_ls

            if st.session_state["nn_lasso_method"] == "Frank-Wolfe":

                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])
                    w = np.linalg.norm(x, 2) ** 2 / (2 * lambda_)
                    w_bar = w
                    it = 0

                    f = partial(f_global, x_=x, lambda_=lambda_)

                    # avoid to compute it every time
                    Mx = np.dot(M.T, x).reshape(a.shape)

                    # case where || M^t x ||_infty <= lambda
                    if np.max(np.abs(Mx)) <= lambda_:
                        return a, np.zeros(x.shape), it

                    while not stop_criterions(a, x, lambda_) and it < it_max:

                        # abbrevations
                        i_star = np.argmax(Mx - np.dot(MtM, a))
                        m_i_star = M[:, i_star]
                        r = x - np.dot(M, a)

                        # STEP 1:
                        if np.max(np.abs(Mx - np.dot(MtM, a))) <= lambda_:
                            a_pre = np.zeros(a.shape)
                            w_pre = 0
                        else:
                            canonic_vec = np.zeros(a.shape)
                            canonic_vec[i_star] = 1
                            a_pre = canonic_vec * \
                                np.sign(np.dot(m_i_star, r)) * w_bar
                            w_pre = w_bar

                        # STEP2:
                        v = m_i_star * np.sign(np.dot(m_i_star, r)) * w_pre - np.dot(
                            M, a
                        )
                        gamma_pre = (np.dot(v, r) + lambda_ * (w - w_pre)) / (
                            np.linalg.norm(v, 2) ** 2
                        )
                        if gamma_pre < 0:
                            gamma = 0
                        elif gamma_pre > 1:
                            gamma = 1
                        else:
                            gamma = gamma_pre

                        # STEP3:
                        a = gamma * a_pre + (1 - gamma) * a
                        w = gamma * w_pre + (1 - gamma) * w
                        w_bar = np.min([w_bar, f(a) / lambda_])

                        it += 1

                    if it == it_max:
                        st.warning("Non-convergence for Frank-Wolfe method")

                    # reconstruction with least-square problem to cancel bias
                    a_ls, approx_ls, it_ls = reconstruction_LS(a, x)
                    # argmin, approx, and nb of iterations
                    return a_ls, approx_ls.flatten(), it, it_ls

            if (
                st.session_state["nn_lasso_method"]
                == "Proximal Gradient with backtracking"
            ):


                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    it = 0
                    Li = 1
                    
                    f = partial(f_global, x_=x, lambda_=lambda_)


                    def prox_l1(z, t):
                        return np.sign(z) * np.maximum(
                            np.abs(z) - np.full(z.shape,
                                                t), np.full(z.shape, 0)
                        )

                    while not stop_criterions(a, x, lambda_) and it < it_max:
                        a1 = prox_l1(a - (np.dot(MtM, a) - Mx) /
                                     Li, lambda_ / Li)
                        while f(a1, x, lambda_) > f_hat(a1, a, x, lambda_):
                            # st.write(f"Multplying Li by {eta}")
                            Li = eta * Li
                            a1 = prox_l1(
                                a - (np.dot(MtM, a) - Mx) / Li, lambda_ / Li)
                        st.write(f"{Li=}")
                        a = a1
                        it += 1

                    if it == it_max:
                        st.warning(
                            "Non-convergence for projected gradient method")

                    # reconstruction with least-square problem to cancel bias
                    a_ls, approx_ls, it_ls = reconstruction_LS(a, x)
                    # argmin, approx, and nb of iterations
                    return a_ls, approx_ls.flatten(), it, it_ls

            if st.session_state["nn_lasso_method"] == "Projected gradient":

                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])

                    # saving result to re_use it at each iterations
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    # each element of the vector is the penalization value
                    Lambda = np.full((a.shape), lambda_)
                    it = 0

                    while not stop_criterions(a, x, lambda_) and it < it_max:
                        a1 = np.maximum(
                            0, a - 1 / L * (np.dot(MtM, a) - Mx + Lambda))
                        # st.write(a1)
                        err = np.linalg.norm(a1 - a)
                        a = a1.copy()
                        it += 1

                    if it == it_max:
                        fegh = 3
                        st.warning(
                            "Non-convergence for projected gradient method")

                    # reconstruction with least-square problem to cancel bias
                    a_ls, approx_ls, it_ls = reconstruction_LS(a, x)
                    # argmin, approx, and nb of iterations
                    return a_ls, approx_ls.flatten(), it, it_ls

            if (
                st.session_state["nn_lasso_method"]
                == "Proximal Gradient with constant step-size"
            ):

                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])

                    # saving result to re_use it at each iterations
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    # each element of the vector is the penalization value
                    it = 0

                    def prox_l1(z, t):
                        return np.sign(z) * np.maximum(
                            np.abs(z) - np.full(z.shape,
                                                t), np.full(z.shape, 0)
                        )

                    while not stop_criterions(a, x, lambda_) and it < it_max:
                        a1 = prox_l1(
                            a - (np.dot(MtM, a) - Mx) / L, lambda_ / L)
                        a = a1
                        it += 1

                    if it == it_max:
                        st.warning(
                            "Non-convergence for projected gradient method")

                    # reconstruction with least-square problem to cancel bias
                    a_ls, approx_ls, it_ls = reconstruction_LS(a, x)
                    # argmin, approx, and nb of iterations
                    return a_ls, approx_ls.flatten(), it, it_ls

            # endregion

            # region Decomposition

            st.session_state["Prop_nn_lasso"] = {}
            nb_it_total = 0
            nb_it_total_ls = 0
            start_time = time.time()

            

            # progress bar
            prog_bar = st.progress(0,"")
            compute_advancement = st.empty()
            nb_done = 0
            nb_curves = len(st.session_state["granulometrics"])
            frac_progress = 1/nb_curves


            for index, x_values in st.session_state["granulometrics"].iterrows():
                with compute_advancement.container():
                    st.write(f"approximation ({nb_done+1} over {nb_curves}) -> label : {index}")
                    prog_bar.progress(frac_progress*(nb_done+1))
                # compute decomposition for our observation x_i
                a_i, approx_i, it_i, it_ls_i = decomposition_algo(
                    x_values.to_numpy(), st.session_state["lambda_nn_lasso"]
                )
                nb_it_total += it_i
                nb_it_total_ls += it_ls_i
                st.session_state["X-X_hat-X_ref"].loc[f"[DD]-{index}"] = approx_i

                # saving coefficient that are non-zero
                prop_dict_i = {}
                sum_aera_i = 0.0
                for i in range(a_i.shape[0]):
                    if a_i[i] != 0:
                        prop_dict_i[
                            st.session_state["discrete_dictionnary"].index[i]
                        ] = (a_i[i] * st.session_state["aeras_dd_curves"][i])
                        sum_aera_i = (
                            sum_aera_i
                            + prop_dict_i[
                                st.session_state["discrete_dictionnary"].index[i]
                            ]
                        )
                for curve in prop_dict_i:
                    prop_dict_i[curve] = prop_dict_i[curve] * 100 / sum_aera_i
                
                # merging proportions of the same component
                merged_prop_dict_i = {}
                for comp_i_name in st.session_state['scaled_ref_curves'].keys():
                    curve_of_comp_i = {}
                    comp_flag = False
                    for curve_name in prop_dict_i.keys():
                        if curve_name.startswith(comp_i_name):
                            # set componentt existence flag 
                            comp_flag = True
                            # identify peak location of the duplicata with regex
                            peak_loc_pattern = r"\((\d*\.?\d+)\)"
                            peak = re.findall(peak_loc_pattern, curve_name)
                            # saving peak and prop
                            curve_of_comp_i[f"{peak[0]}"] = prop_dict_i[curve_name]
                    
                    if comp_flag:
                        # avergage of peak location weighted with area
                        merged_peak = np.average([float(peak) for peak in curve_of_comp_i.keys()], weights = list(curve_of_comp_i.values()))
                        sum_aera = np.sum(list(curve_of_comp_i.values()))

                        # saving result 
                        merged_prop_dict_i[f"{comp_i_name} ({round(merged_peak,2)})"] = sum_aera

                # proportions
                st.session_state["Prop_nn_lasso"][index] = merged_prop_dict_i
                #st.session_state["Prop_nn_lasso"][index] = prop_dict_i

                # errors
                L1_rel = L1_relative(approx_i, index)
                l2 = np.linalg.norm(approx_i - x_values,2)
                new_row = {"L1 relative error": L1_rel,"l2 error":  l2, "nb components": int(len(merged_prop_dict_i))}
                st.session_state['dd_errors'].loc[index] = new_row
                nb_done += 1

            end_time = time.time()

            st.session_state['L1_mean_dd'] = np.mean(st.session_state['dd_errors']["L1 relative error"])
            st.session_state['l2_mean_dd'] = np.mean(st.session_state['dd_errors']["l2 error"])


            mean_it = (1.0 * nb_it_total) / \
                len(st.session_state["granulometrics"])
            mean_it_ls = (1.0 * nb_it_total_ls) / \
                len(st.session_state["granulometrics"])
            with compute_advancement.container():
                st.success("Decomposition computed with success")
                st.write(
                    f"mean of iterations (decomposition algorithm) : {mean_it}")
                st.write(f"mean of iterations (reconstruction algorithm) : {mean_it_ls}")
                st.write(f"Execution time : {end_time-start_time:.2f} seconds")

            st.session_state["dd_flag"] = True
            # endregion

        # region scaled reference curves
        st.markdown("---")
        st.subheader("Scaled reference curves")
        st.markdown(
            """Reference curves are scaled so they all have an aera under the curve of one (with logarithmic abscisses). 
                    This result in reference curves' peak height that are way greater than the observations. Then the coefficient
                    for this reference curve in NN-LASSO is shrinked because it is very low in order to have a good reconstruction 
                    with the good peak height. To fix that we re-scaled the references curves so they have the same peak height as 
                    observation mean. We have then the following observations-scaled referenced curves :"""
        )
        fig = go.Figure()
        abscisses = st.session_state["ref_curves"]["ref_ArgilesFines"][
            0, :
        ]  # for abscisses
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["Argiles Fines"],
                mode="lines",
                name="Argiles fines (<1 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["Argiles Classiques"],
                mode="lines",
                name="Argiles grossieres (1-7 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["Limons fins"],
                mode="lines",
                name="Limons fins (7-20 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["Limons Grossiers"],
                mode="lines",
                name="Limons grossiers (20-50 microns)",
            )
        )
        # fig.add_trace(
        #     go.Scatter(
        #         x=abscisses,
        #         y=st.session_state["scaled_ref_curves"]["ref_Loess"],
        #         mode="lines",
        #         name="Loess (20-50 microns)",
        #     )
        # )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["Loess"],
                mode="lines",
                name="Loess sans limons (20-50 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["Sables Fins"],
                mode="lines",
                name="Sables fins (50-100 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["Sables Grossiers"],
                mode="lines",
                name="Sables grossiers (>100 microns)",
            )
        )

        fig.update_xaxes(type="log", tickformat=".1f", dtick=1, showgrid=True)
        fig.update_layout(
            height=500,
            showlegend=True,
            legend = {'font':{'size' : 20}},
            xaxis={'title': {'text': "grain size (micrometers, log-scale)", 'font': {'size': 20}}}
        )
        fig.update_traces(
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>")

        st.plotly_chart(fig)
        # endregion


with tab_NMF:
    col01, col02, col03 = st.columns([1, 3, 1])

    with col02:
        st.markdown("<h1 style='text-align: center;'>Unsupervised (NMF)</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            """ Perform the following factorisation :  $ Y \\thickapprox XM ~ $
                    by minimising the following expression :
                    """
        )
        penalization = r'''\begin{align*}
        \arg \min_{X,M}~ & D_\beta(Y\vert XM)\\ &+2\alpha_M l_{1_\text{ratio}} m\Vert M \Vert_1 \\& +2 \alpha_X l_{1_\text{ratio}} n\Vert X \Vert_1 \\ &+ \alpha_M (1-l_1{_\text{ratio}}) m\Vert M \Vert_F^2 \\ &+\alpha_X(1-l_1{_\text{ratio}}) n\Vert X \Vert_F^2 
        \end{align*}'''
        st.latex(penalization)

        st.write("")
        st.write(
            """ You can choose the values of the parameters : $~\\beta,~l_1{_\\text{ratio}},~\\alpha_M,~\\alpha_X$
                    """
        )

        st.header("Parameters for basic NMF")

        col1, col2, col3, col4, col5 = st.columns([1, 1.5, 1, 1, 1])

        with col1:
            st.number_input(
                "nb of components",
                key="nb_end_members",
                min_value=2,
                max_value=100,
                step=1,
                value=7,        # index of the default value to take in the list [min_value:max_value]
                format="%d",
            )

        with col2:
            loss_choice = st.selectbox(
                "$\\beta$ (divergence)",
                ("Frobenius (CD)", "Frobenius (MU)", "Kullback-Leibler (MU)"),
                index = 2
            )
            if loss_choice == "Kullback-Leibler (MU)":
                st.session_state["loss"] = "kullback-leibler"
                st.session_state["solver"] = "mu"
            elif loss_choice == "Frobenius (CD)":
                st.session_state["loss"] = "frobenius"
                st.session_state["solver"] = "cd"
            else:
                st.session_state["loss"] = "frobenius"
                st.session_state["solver"] = "mu"

        with col3:
            st.number_input(
                "l1-l2 ratio", key="ratio_l1",value = 1.0, min_value=0.0, max_value=1.0, format="%f"
            )
        with col4:
            st.number_input("penalization coef M", format="%f", key="a_W", value = 0.02, min_value = 0.0)
        with col5:
            st.number_input("penalization coef X", format="%f", key="a_A", value = 0.02, min_value = 0.0)

        st.header("Algorithm")

        if st.button("Lunch basic factorization"):
            begin_nmf = time.time()
            X = st.session_state["granulometrics"].to_numpy()
            model = NMF(
                n_components=st.session_state["nb_end_members"],
                solver=st.session_state["solver"],
                beta_loss=st.session_state["loss"],
                init="random",
                l1_ratio=st.session_state["ratio_l1"],
                alpha_W=st.session_state["a_W"],
                alpha_H=st.session_state["a_A"],
                random_state=0,
                tol=1e-6,
                max_iter=15000,
            )
            # Increase max_iter to get convergence
            A = model.fit_transform(X)
            M = model.components_

            # df for the approximation
            # Estimations of our observations with only the EMs
            X_nmf = pd.DataFrame(
                A @ M,
                columns=st.session_state["granulometrics"].columns,
                index=st.session_state["granulometrics"].index,
            )

            # df for the proportions
            EM_areas = np.apply_along_axis(trapeze_areas, 1, M).reshape(
                (A.shape[1])
            )  # compute areas of each EM
            Prop = A * EM_areas
            Prop = np.apply_along_axis(lambda x: x / np.sum(x) * 100, 1, Prop)

            # naming the columns of Prop with regards of where the peak is located for each EM
            # We have to use temporary variable because we can't change columns name one by one
            prop_col_label = [0] * Prop.shape[1]
            for i in range(M.shape[0]):
                peak = st.session_state['granulometrics'].columns[np.argmax(M[i, :])]
                for key, values in materials.items():
                    if peak < values:
                        prop_col_label[i] = key + f" ({peak})"
                        break
            st.session_state["Prop_nmf"] = pd.DataFrame(
                Prop,
                index=st.session_state["granulometrics"].index,
                columns=prop_col_label,
            )
            end_nmf = time.time()

            # saving nmf erros and nb components
            nb_components = (np.round(Prop,1) != 0.0).sum(axis = 1)
            l2_nmf = np.linalg.norm(X_nmf - st.session_state["granulometrics"], axis=1)
            l1_rel_nmf = X_nmf.apply(lambda row: L1_relative(row.values, row.name), axis=1)
            errors_series_nmf = pd.Series({})
            st.session_state['nmf_errors'] = pd.DataFrame({"L1 relative error": l1_rel_nmf,"l2 error":  l2_nmf, "nb components": nb_components}, index = st.session_state['granulometrics'].index)

            # mean of errors
            st.session_state['L1_mean_nmf'] = np.mean(l1_rel_nmf)
            st.session_state['l2_mean_nmf'] = np.mean(l2_nmf)

            # adding approximation to our result df
            X_nmf.index = X_nmf.index.map(
                lambda x: f"[NMF]-{x}")  # adding "[NMF]-" before

            # in this case we replace the old nmf approximation
            if st.session_state["nmf_flag"]:
                for ind in X_nmf.index:
                    st.session_state["X-X_hat-X_ref"].loc[ind] = X_nmf.loc[ind]

            else:  # easier case : there isn't already a nmf approximation
                st.session_state["X-X_hat-X_ref"] = pd.concat(
                    [st.session_state["X-X_hat-X_ref"], X_nmf], axis=0
                )
                st.session_state["nmf_flag"] = True  # They are now result

            st.success(f"Approximation succeed ! exec time = {round(end_nmf-begin_nmf,3)} seconds")
            # Displaying approx errors
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r""" \frac{1}{n}\sum_{i=1}^{n} \Vert x_i-\hat{x}_i \Vert_2 """)
            with col2:
                st.metric(
                    "mean of quadratic (l2) errors",
                    value=f"{st.session_state['l2_mean_nmf']:.4}",
                    label_visibility="visible",
                )
            col1, col2 = st.columns(2)
            with col1:
                st.latex(
                    r""" \frac{1}{n}\sum_{i=1}^{n} \frac{\Vert x_i-\hat{x}_i \Vert_{L1}}{\Vert x_i \Vert_{L1}} """
                )
            with col2:
                st.metric(
                    "mean of L1-relative errors (%)",
                    value=f"{st.session_state['L1_mean_nmf']:.3}%",
                    label_visibility="visible",
                )

            st.header("Visualization")

            with st.expander("End-Members"):

                fig = make_subplots(
                    rows=st.session_state["nb_end_members"] // 2
                    + st.session_state["nb_end_members"] % 2,
                    cols=2,
                    subplot_titles=[
                        f"End-Member {i}"
                        for i in range(1, st.session_state["nb_end_members"] + 1)
                    ],
                )
                for i in range(st.session_state["nb_end_members"]):
                    row = (i // 2) + 1
                    col = (i % 2) + 1
                    fig.add_trace(
                        go.Scatter(
                            x=st.session_state["granulometrics"].columns,
                            y=M[i, :],
                            mode="lines",
                        ),
                        row=row,
                        col=col,
                    )

                fig.update_xaxes(type="log", tickformat=".1f",
                                 dtick=1, showgrid=True)
                fig.update_yaxes(showgrid=True)
                fig.update_layout(
                    height=1300,
                    width=700,
                    title_text="End-members curves",
                    showlegend=False,
                )
                fig.update_traces(
                    hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>"
                )

                st.plotly_chart(fig)

            with st.expander("Proportions of EM in our observations"):
                st.session_state["A_df"] = pd.DataFrame(
                    A,
                    index=st.session_state["granulometrics"].index,
                    columns=[
                        f"EM{i}"
                        for i in range(1, st.session_state["nb_end_members"] + 1)
                    ],
                )
                st.session_state["A_df"]["label"] = st.session_state[
                    "granulometrics"
                ].index
                fig = make_subplots(
                    rows=st.session_state["nb_end_members"] // 2,
                    cols=1,
                    vertical_spacing=0.05,
                )
                for i in range(st.session_state["nb_end_members"] // 2):

                    first_em = 2 * i + 1
                    second_em = 2 * (i + 1)

                    fig.add_trace(
                        go.Scatter(
                            x=st.session_state["A_df"][f"EM{first_em}"],
                            y=st.session_state["A_df"][f"EM{second_em}"],
                            mode="markers",
                            marker=dict(
                                size=10,
                                color=st.session_state["A_df"]["label"]
                                .astype("category")
                                .cat.codes,
                                colorscale="rainbow",
                            ),
                            text=st.session_state["A_df"]["label"],
                        ),
                        row=i + 1,
                        col=1,
                    )
                    fig.update_xaxes(
                        title_text=f"End-member {first_em}",
                        showgrid=False,
                        gridcolor="LightGray",
                        row=i + 1,
                        col=1,
                    )
                    fig.update_yaxes(
                        title_text=f"End-member {second_em}",
                        showgrid=False,
                        gridcolor="LightGray",
                        row=i + 1,
                        col=1,
                    )

                fig.update_layout(
                    # Ajuster la hauteur de la figure en fonction du nombre de plots
                    height=700 * st.session_state["nb_end_members"] // 2,
                    title_text="Proprotions of End-members",
                    showlegend=False,  # Masquer la légende pour simplifier l'affichage
                )

                st.plotly_chart(fig)


with tab_result:

    # region plot


    st.markdown("<h1 style='text-align: center;'>Results</h1>", unsafe_allow_html=True)
    st.markdown("---")
    labels_obs = st.session_state["granulometrics"].index
    labels_approx_nmf = st.session_state["X-X_hat-X_ref"].index[
        st.session_state["X-X_hat-X_ref"].index.str.startswith(("^"))
    ]
    labels_approx_rc = st.session_state["X-X_hat-X_ref"].index[
        st.session_state["X-X_hat-X_ref"].index.str.startswith(("r"))
    ]

    # Selection of curves to plot
    st.multiselect(
            ":red[**Enter observation label :**]",
            options=labels_obs,
            key="selected_obs_labels",
        )
    col2, col3, col4 = st.columns(3)
    with col2:
        st.toggle(
            "Plot NMF approximations",
            key="flag_nmf_approx",
            value=True,
            disabled=not st.session_state["nmf_flag"],
        )
        st.toggle(
            "Display NMF component proportions and errors",
            key = 'flag_nmf_prop',
            value = True,
            disabled=not st.session_state["nmf_flag"]
        )
        
    # with col3:
    #     st.toggle(
    #         "Display approximations with reference curves",
    #         key="flag_rc_approx",
    #         value=False,
    #         disabled=not st.session_state["rc_flag"],
    #     )
    with col3:
        st.toggle(
            "Plot Discrete Dictionary approximations",
            key="flag_nnlasso_approx",
            value=True,
            disabled=not st.session_state["dd_flag"],
        )
        st.toggle(
            "Display Discrete Dictionary component proportions and errors",
            key = 'flag_dd_prop',
            value = True,
            disabled=not st.session_state["dd_flag"]
        )
    with col4:
        st.toggle(
            "Plot Continuous Dictionary approximations",
            key = "flag_blasso_approx",
            value = True,
            disabled= not st.session_state["cd_flag"] or st.session_state['flag_other_dataset']
        )
        st.toggle(
            "Display Continuous Dictionary component proportions and errors",
            key = 'flag_cd_prop',
            value = True,
            disabled = not st.session_state["cd_flag"] or st.session_state['flag_other_dataset']
        )

    st.markdown("---")
    st.info("Clic bellow to expand and view plot")
    with st.expander("**Plot**", icon ="📈",expanded = False):
        if st.session_state['selected_obs_labels']:

            curves_and_approx = st.session_state["X-X_hat-X_ref"]
            fig = go.Figure()
            for label in st.session_state["selected_obs_labels"]:
                fig.add_trace(
                    go.Scatter(
                        x=curves_and_approx.columns,
                        y=curves_and_approx.loc[label],
                        mode="lines",
                        name=label,
                        line = {'width' : 3}
                    )
                )
                if st.session_state["flag_nmf_approx"] and st.session_state['nmf_flag']:
                    fig.add_trace(
                        go.Scatter(
                            x=curves_and_approx.columns,
                            y=curves_and_approx.loc[f"[NMF]-{label}"],
                            mode="lines",
                            name=f"[NMF]-{label}",
                            line = {'width' : 3}                            
                        )
                    )
                # if st.session_state["flag_rc_approx"]:
                #     fig.add_trace(
                #         go.Scatter(
                #             x=curves_and_approx.columns,
                #             y=curves_and_approx.loc[f"r{label}"],
                #             mode="lines",
                #             name=f"r{label}",
                #         )
                #     )
                if st.session_state["flag_nnlasso_approx"] and st.session_state['dd_flag']:
                    fig.add_trace(
                        go.Scatter(
                            x=curves_and_approx.columns,
                            y=curves_and_approx.loc[f"[DD]-{label}"],
                            mode="lines",
                            name=f"[DD]-{label}",
                            line = {'width' : 3}
                        )
                    )
                if not st.session_state['flag_other_dataset'] and st.session_state['cd_flag']:
                    fig.add_trace(
                        go.Scatter(
                            x = curves_and_approx.columns,
                            y = curves_and_approx.loc[f"[CD]-{label}"],
                            mode="lines",
                            name=f"[CD]-{label}",
                            line = {'width' : 3}
                        )
                    )

            fig.update_xaxes(type="log", tickformat='.1f', dtick=1, showgrid=True)
            fig.update_layout(
                height=800,
                width=1200,
                showlegend=True,
                legend = {'font':{'size' : 40}},
                xaxis={'title': {'text': "grain size (micrometers, log-scale)", 'font': {'size': 35}}, 'tickfont' : {'size' : 20}}
            )
            fig.update_traces(
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>")

            st.plotly_chart(fig)
        else:
            st.warning("**Please select labels to plot")
    #endregion plot
    
    st.info("Clic bellow to expand and export results")
    with st.expander("Export results"):
        st.markdown("<h5 style='text-align: center;'>Result exportation</h5>", unsafe_allow_html=True)
        st.segmented_control("**What result to save ?**", 
                            options = ["Proportions of decomposition","Approximations","Quality of approximation"],
                            default="Proportions of decomposition",key='result_type_export',
                            help = "|-| **Proportions** : Dictionnary with prop of components (values) for each sample (key)         |-| "
                                  +"**Approximations** : Tab with approximation's discretized curves of each sample         |-|"
                                  +"**Quality of approximation** : Tab with errors metrics (norms) of each sample approximaton"
                            )
        col1, col2 = st.columns(2)
        if st.session_state['result_type_export'] == "Proportions of decomposition" or st.session_state['result_type_export'] == "Quality of approximation":
            methods_options = [
                option for flag, option in [
                    (st.session_state['cd_flag'], "Continuous Dictionary"),
                    (st.session_state['dd_flag'], "Discrete Dictionary"),
                    (st.session_state['nmf_flag'], "NMF")
                ] if flag
            ]
        else :
            methods_options = ['All'] 
        
        
        with col1:
            st.radio("**Which method ?**", options = methods_options,index = 0,key='method_result')
            
        if st.session_state['result_type_export'] == "Proportions of decomposition" and (st.session_state['method_result'] == "Continuous Dictionary" or st.session_state['method_result'] == "Discrete Dictionary"):
            export_extensions = [".json"]
        else :
            export_extensions = [".csv",".xlsx",".txt"]

        with col2:
            st.segmented_control("**Exportation format**",options = export_extensions,selection_mode="single",default=export_extensions[0],key='export_result_format')
        
        # Dictionnary to handle name of result file for each method
        methods_name = {
            "All" : 'all_methods',
            "Continuous Dictionary" : 'CD',
            "Discrete Dictionary" : 'DD',
            "NMF" : 'NMF'
        }
        
        if st.button("Export result"):

            if st.session_state['result_type_export'] == "Approximations":

                st.session_state['name_result_file'] = "Approx_and_original_curves"
                if st.session_state['export_result_format'] == '.xlsx':
                    st.session_state["X-X_hat-X_ref"].to_excel("exports/"+st.session_state['name_result_file']+'.xlsx',sheet_name='Feuil1',index=True)
                if st.session_state['export_result_format'] == '.csv':
                    st.session_state["X-X_hat-X_ref"].to_csv("exports/"+st.session_state['name_result_file']+'.csv', float_format='%.4f', index=True)
                if st.session_state['export_result_format'] == '.txt':
                    st.session_state["X-X_hat-X_ref"].to_csv("exports/"+st.session_state['name_result_file']+'.txt', float_format='%.4f', index=True)

            elif st.session_state['result_type_export'] == "Proportions of decomposition":
                st.session_state['name_result_file'] = "Proportion_"+methods_name[st.session_state['method_result']]+"_granulo_analysis"
                
                if st.session_state['method_result'] == "NMF":
                    dict_export = st.session_state['Prop_nmf']
                    if st.session_state['export_result_format'] == '.xlsx':
                        dict_export.to_excel("exports/"+st.session_state['name_result_file']+'.xlsx',sheet_name='Feuil1',index=True)
                    if st.session_state['export_result_format'] == '.csv':
                        dict_export.to_csv("exports/"+st.session_state['name_result_file']+'.csv', float_format='%.4f', index=True)
                    if st.session_state['export_result_format'] == '.txt':
                        dict_export.to_csv("exports/"+st.session_state['name_result_file']+'.txt', float_format='%.4f', index=True)
                else :
                    if st.session_state['method_result'] == "Continuous Dictionary":
                        dict_export = st.session_state['blasso_Prop']
                    else:
                        dict_export = st.session_state['Prop_nn_lasso']
                    with open("exports/"+st.session_state['name_result_file']+".json", "w") as json_file:
                        json.dump(dict_export, json_file, indent=4)

            else :
                st.session_state['name_result_file'] = "Quality_approx_"+methods_name[st.session_state['method_result']]+"_granulo_analysis"

                # construction of a data frame that contains every quality measurement
                qualities_approx = pd.DataFrame()
                qualities_approx.index = st.session_state['granulometrics'].index

                if st.session_state['method_result'] == "Continuous Dictionary":
                    qualities_approx = pd.concat([qualities_approx,st.session_state['cd_errors']], axis=1)
                elif st.session_state['method_result'] == "Discrete Dictionary":
                    qualities_approx = pd.concat([qualities_approx,st.session_state['dd_errors']], axis=1)
                else :
                    qualities_approx = pd.concat([qualities_approx,st.session_state['nmf_errors']], axis=1)

                # calculat average norms over all data
                mean_unsorted = qualities_approx.mean()
                mean_sorted = [mean_unsorted["l2 error"],mean_unsorted["nb components"],mean_unsorted["L1 relative error"]]
                df_mean = pd.DataFrame([mean_sorted],columns= ["l2 error","nb components","L1 relative error"], index=["AVERAGE"])
                qualities_approx = pd.concat([df_mean,qualities_approx])
                
                if st.session_state['export_result_format'] == '.xlsx':
                    qualities_approx.to_excel("exports/"+st.session_state['name_result_file']+'.xlsx',sheet_name='Feuil1',index=True)
                if st.session_state['export_result_format'] == '.csv':
                    qualities_approx.to_csv("exports/"+st.session_state['name_result_file']+'.csv', float_format='%.4f', index=True)
                if st.session_state['export_result_format'] == '.txt':
                    qualities_approx.to_csv("exports/"+st.session_state['name_result_file']+'.txt', float_format='%.4f', index=True)
               
            
            st.success("Export file has been created")
            
            # Display button to download export file  
            result_file_name = st.session_state['name_result_file'] + st.session_state['export_result_format']
            if st.session_state['export_result_format'] == '.xlsx':
                st.download_button("📂 Download result file (.xlsx)", data=get_export_file(
                    "exports/"+result_file_name), mime="application/octet-stream", file_name=result_file_name)
            if st.session_state['export_result_format'] == '.csv':
                st.download_button("📂 Download result file (.csv)", data=get_export_file(
                    "exports/"+result_file_name), mime="text", file_name=result_file_name)
            if st.session_state['export_result_format'] == '.txt':
                st.download_button("📂 Download result file (.txt)", data=get_export_file(
                    "exports/"+result_file_name), mime="text", file_name=result_file_name)
            if st.session_state['export_result_format'] == '.json':
                st.download_button("📂 Download result file (.json)", data=get_export_file(
                    "exports/"+result_file_name),mime="text", file_name=result_file_name)
    


    
    if st.session_state['selected_obs_labels']:
        col_nmf, col_dd, col_cd = st.columns(3)
        with col_nmf:
            if st.session_state["flag_nmf_prop"] and st.session_state["nmf_flag"]:
                st.markdown("<h3 style='text-align: center;'>[NMF]</h3>", unsafe_allow_html=True)
                st.markdown("---")
                for label in st.session_state["selected_obs_labels"]:
                    prop = st.session_state['Prop_nmf'].loc[label]
                    fig = go.Figure(data=[go.Bar(
                        x=list(prop.index),
                        y=list(prop.values),
                        text = np.round(list(prop.values),2),
                        marker=dict(
                            color=np.round(list(prop.values),2),  
                            colorscale='Inferno'
                        )
                    )])
                    fig.update_layout(
                        title = "Proportions of components (NMF)", 
                        xaxis = {'tickfont' : {'size' : 15}}, 
                        height = 400,
                        margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig)

            # if st.session_state["rc_flag"]:
            #     st.subheader(
            #         "Proportions of reference curve (approximations) for selected observation"
            #     )
            #     st.dataframe(
            #         st.session_state["Prop_rc"].loc[st.session_state["selected_obs_labels"]]
            #     )
        with col_dd:
            if st.session_state["flag_dd_prop"] and st.session_state["dd_flag"]:
                st.markdown("<h3 style='text-align: center;'>[Discrete Dictionary]</h3>", unsafe_allow_html=True)
                st.markdown("---")
                for label in st.session_state["selected_obs_labels"]:
                    prop = st.session_state['Prop_nn_lasso'][label]
                    fig = go.Figure(data=[go.Bar(
                        x=list(prop.keys()),
                        y=list(prop.values()),
                        text = np.round(list(prop.values()),2),
                        marker=dict(
                            color=np.round(list(prop.values()),2),  
                            colorscale='Inferno'
                        )
                    )])
                    fig.update_layout(
                        title = "Proportions of components (DD)", 
                        xaxis = {'tickfont' : {'size' : 15}}, 
                        height = 400,
                        margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig)
                    
    
        with col_cd:
            if st.session_state["flag_cd_prop"] and st.session_state["cd_flag"]:
                st.markdown("<h3 style='text-align: center;'>[Continuous Dictionary]</h3>", unsafe_allow_html=True)
                st.markdown("---")
                for label in st.session_state["selected_obs_labels"]:
                    prop = st.session_state['blasso_Prop'][label]
                    fig = go.Figure(data=[go.Bar(
                        x=list(prop.keys()),
                        y=list(prop.values()),
                        text = np.round(list(prop.values()),2),
                        marker=dict(
                            color=np.round(list(prop.values()),2),  
                            colorscale='Inferno'
                        )
                    )])
                    fig.update_layout(
                        title = "Proportions of components (CD)", 
                        xaxis = {'tickfont' : {'size' : 15}}, 
                        height = 400,
                        margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig)
                
        
        col_nmf, col_dd, col_cd = st.columns(3)

        with col_nmf:
            if st.session_state["flag_nmf_prop"] and st.session_state["nmf_flag"]:
                st.write("**Errors (NMF)**")
                st.table(st.session_state['nmf_errors'].loc[st.session_state["selected_obs_labels"]].transpose())
                st.write("**Average errors on all data**")
                col1, col2 = st.columns(2)
                with col1:
                    st.latex(
                        r""" \frac{1}{n}\sum_{i=1}^{n} \frac{\Vert x_i-\hat{x}_i \Vert_{L1}}{\Vert x_i \Vert_{L1}} """
                    )
                with col2:
                    st.metric(
                        "mean of L1-relative errors (%)",
                        value=f"{st.session_state['L1_mean_nmf']:.3}%",
                        label_visibility="visible",
                    )
                with col1:
                    st.latex(r""" \frac{1}{n}\sum_{i=1}^{n} \Vert x_i-\hat{x}_i \Vert_2 """)
                with col2:
                    st.metric(
                        "mean of quadratic (l2) errors",
                        value=f"{st.session_state['l2_mean_nmf']:.4}",
                        label_visibility="visible",
                    )

                st.markdown("<h5 style='text-align: center;'>[NMF]<br>Nb of components distribution</h5>", unsafe_allow_html=True)
                fig = go.Figure(data=[go.Histogram(x=st.session_state['nmf_errors']["nb components"])])
                fig.update_layout(bargap=0.1, xaxis = {'tickfont' : {'size' : 30}}, height = 300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig)
            
                

        with col_dd:
            if st.session_state["flag_dd_prop"] and st.session_state["dd_flag"]:
                st.write("**Errors (DD)**")
                for label in st.session_state["selected_obs_labels"]:
                    st.table(st.session_state['dd_errors'].loc[label])
                
                st.write("**Average errors on all data**")
                col1, col2 = st.columns(2)
                with col1:
                    st.latex(
                        r""" \frac{1}{n}\sum_{i=1}^{n} \frac{\Vert x_i-\hat{x}_i \Vert_{L1}}{\Vert x_i \Vert_{L1}} """
                    )
                with col2:
                    st.metric(
                        "mean of L1-relative errors (%)",
                        value=f"{st.session_state['L1_mean_dd']:.3}%",
                        label_visibility="visible",
                    )
                with col1:
                    st.latex(r""" \frac{1}{n}\sum_{i=1}^{n} \Vert x_i-\hat{x}_i \Vert_2 """)
                with col2:
                    st.metric(
                        "mean of quadratic (l2) errors",
                        value=f"{st.session_state['l2_mean_dd']:.4}",
                        label_visibility="visible",
                    )
                
                st.markdown("<h5 style='text-align: center;'>[Discrete Dictionary]<br>Nb of components distribution</h5>", unsafe_allow_html=True)
                fig = go.Figure(data=[go.Histogram(x=st.session_state['dd_errors']["nb components"])])
                fig.update_layout(bargap=0.1, xaxis = {'tickfont' : {'size' : 30}}, height = 300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig)
            
        with col_cd:
            if st.session_state["flag_cd_prop"] and st.session_state["cd_flag"]:
                st.write("**Errors (CD)**")
                for label in st.session_state["selected_obs_labels"]:
                    st.table(st.session_state['cd_errors'].loc[label])
                
                st.write("**Average errors on all data**")
                col1, col2 = st.columns(2)
                with col1:
                    st.latex(
                        r""" \frac{1}{n}\sum_{i=1}^{n} \frac{\Vert x_i-\hat{x}_i \Vert_{L1}}{\Vert x_i \Vert_{L1}} """
                    )
                with col2:
                    st.metric(
                        "mean of L1-relative errors (%)",
                        value=f"{st.session_state['L1_mean_cd']:.3}%",
                        label_visibility="visible",
                    )
                with col1:
                    st.latex(r""" \frac{1}{n}\sum_{i=1}^{n} \Vert x_i-\hat{x}_i \Vert_2 """)
                with col2:
                    st.metric(
                        "mean of quadratic (l2) errors",
                        value=f"{st.session_state['l2_mean_cd']:.4}",
                        label_visibility="visible",
                    )                
                st.markdown("<h5 style='text-align: center;'>[Continuous Dictionary]<br>Nb of components distribution</h5>", unsafe_allow_html=True)
                fig = go.Figure(data=[go.Histogram(x=st.session_state['cd_errors']["nb components"])])
                fig.update_layout(bargap=0.1, xaxis = {'tickfont' : {'size' : 30}}, height = 300, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig)
                
                

# region tab_rc
# with tab_rc:
#     col01, col02, col03 = st.columns([1, 3, 1])
#     with col02:
#         st.header("Approximation of our observation by reference curves")
#         st.markdown(
#             """In this section we don't use any NMF algorithm. Instead we use reference curves 
#                     that has been build from various curves of our data set that has been certified as 
#                     pure by experts. We're going to use these curves to compare them with the end-members
#                     we find and also to build differents approximations."""
#         )

#         st.subheader("Scaled reference curves")
#         fig = go.Figure()
#         fig.add_trace(
#             go.Scatter(
#                 x=st.session_state["ref_curves"]["ref_ArgilesFines"][0, :],
#                 y=st.session_state["scaled_ref_curves"]["Argiles Fines"],
#                 mode="lines",
#                 name="Argiles Fines (<1 microns)",
#             )
#         )
#         fig.update_xaxes(type="log", tickformat=".1e", dtick=1, showgrid=True)
#         fig.update_layout(
#             height=500,
#             showlegend=True,
#             xaxis_title=" grain diameter (micrometers, log-scale)",
#         )
#         fig.update_traces(
#             hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>")

#         fig.add_trace(
#             go.Scatter(
#                 x=st.session_state["ref_curves"]["ref_ArgilesClassiques"][0, :],
#                 y=st.session_state["scaled_ref_curves"]["Argiles Classiques"],
#                 mode="lines",
#                 name="Argiles Grossieres (1-7 microns)",
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=st.session_state["ref_curves"]["ref_Alterites"][0, :],
#                 y=st.session_state["scaled_ref_curves"]["Limons fins"],
#                 mode="lines",
#                 name="Alterites (7-20 microns)",
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=st.session_state["ref_curves"]["ref_SablesFins"][0, :],
#                 y=st.session_state["scaled_ref_curves"]["Sables Fins"],
#                 mode="lines",
#                 name="Sables Fins (50-100 microns)",
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=st.session_state["ref_curves"]["ref_SablesGrossiers"][0, :],
#                 y=st.session_state["scaled_ref_curves"]["Sables Grossiers"],
#                 mode="lines",
#                 name="Sables Grossiers (>100 microns)",
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=st.session_state["ref_curves"]["ref_LimonsGrossiers"][0, :],
#                 y=st.session_state["scaled_ref_curves"]["Limons Grossiers"],
#                 mode="lines",
#                 name="Limons Grossiers",
#             )
#         )
#         fig.add_trace(
#             go.Scatter(
#                 x=st.session_state["ref_curves"]["ref_Loess_without_residules"][0, :],
#                 y=st.session_state["scaled_ref_curves"]["Loess (without residues)"],
#                 mode="lines",
#                 name="Loess (without residues)",
#             )
#         )
#         # fig.add_trace(
#         #     go.Scatter(
#         #         x=st.session_state["ref_curves"]["ref_Loess"][0, :],
#         #         y=st.session_state["scaled_ref_curves"]["ref_Loess"],
#         #         mode="lines",
#         #         name="Loess",
#         #     )
#         # )

#         st.plotly_chart(fig)

#         st.subheader("List of reference curves")
#         st.markdown(f"""There are 8 differents reference curves that are mainly characterised by the location
#                     of the peak on the x axis (diametre in $\\mu m$). You can see their plots below.""")

#         with st.expander("List of reference curves :"):

#             st.markdown(
#                 "We plot first the reference curve of the Argiles Fines (fine clay) because its peak is much greater than the others"
#             )

#             fig = go.Figure()
#             fig.add_trace(
#                 go.Scatter(
#                     x=st.session_state["ref_curves"]["ref_ArgilesFines"][0, :],
#                     y=st.session_state["ref_curves"]["ref_ArgilesFines"][1, :],
#                     mode="lines",
#                     name="Argiles Fines (<1 microns)",
#                 )
#             )
#             fig.update_xaxes(type="log", tickformat=".1e",
#                              dtick=1, showgrid=True)
#             fig.update_layout(
#                 height=500,
#                 showlegend=True,
#                 xaxis_title=" grain diametere (micrometers, log-scale)",
#             )
#             fig.update_traces(
#                 hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>")

#             st.plotly_chart(fig)

#             fig = go.Figure()
#             fig.add_trace(
#                 go.Scatter(
#                     x=st.session_state["ref_curves"]["ref_ArgilesClassiques"][0, :],
#                     y=st.session_state["ref_curves"]["ref_ArgilesClassiques"][1, :],
#                     mode="lines",
#                     name="Argiles Grossieres (1-7 microns)",
#                 )
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=st.session_state["ref_curves"]["ref_Alterites"][0, :],
#                     y=st.session_state["ref_curves"]["ref_Alterites"][1, :],
#                     mode="lines",
#                     name="Alterites (7-20 microns)",
#                 )
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=st.session_state["ref_curves"]["ref_SablesFins"][0, :],
#                     y=st.session_state["ref_curves"]["ref_SablesFins"][1, :],
#                     mode="lines",
#                     name="Sables Fins (50-100 microns)",
#                 )
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=st.session_state["ref_curves"]["ref_SablesGrossiers"][0, :],
#                     y=st.session_state["ref_curves"]["ref_SablesGrossiers"][1, :],
#                     mode="lines",
#                     name="Sables Grossiers (>100 microns)",
#                 )
#             )

#             fig.update_xaxes(type="log", tickformat=".1e", showgrid=True)
#             fig.update_layout(
#                 # Ajuster la hauteur de la figure en fonction du nombre de plots
#                 height=500,
#                 xaxis_title=" grain diametere (micrometers, log-scale)",
#             )
#             fig.update_traces(
#                 hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>")

#             st.plotly_chart(fig)

#             st.markdown(
#                 "For the peak located between 20 and 50 $\\mu m$ we can choose between 3 different reference curves :"
#             )
#             st.markdown(" - Limons Grossier")
#             st.markdown(" - Limons Grossier-Loess")
#             st.markdown(" - Limons Loess")

#             fig = go.Figure()
#             fig.add_trace(
#                 go.Scatter(
#                     x=st.session_state["ref_curves"]["ref_LimonsGrossiers"][0, :],
#                     y=st.session_state["ref_curves"]["ref_LimonsGrossiers"][1, :],
#                     mode="lines",
#                     name="Limons Grossiers",
#                 )
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=st.session_state["ref_curves"]["ref_Loess_without_residules"][0, :],
#                     y=st.session_state["ref_curves"]["ref_Loess_without_residules"][1, :],
#                     mode="lines",
#                     name="Limons Grossiers-Loess",
#                 )
#             )
#             fig.add_trace(
#                 go.Scatter(
#                     x=st.session_state["ref_curves"]["ref_Loess"][0, :],
#                     y=st.session_state["ref_curves"]["ref_Loess"][1, :],
#                     mode="lines",
#                     name="Loess",
#                 )
#             )

#             fig.update_xaxes(type="log", tickformat=".1e",
#                              dtick=1, showgrid=True)
#             fig.update_layout(
#                 # Ajuster la hauteur de la figure en fonction du nombre de plots
#                 height=500,
#                 xaxis_title=" grain diametere (micrometers, log-scale)",
#             )
#             fig.update_traces(
#                 hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>")

#             st.plotly_chart(fig)

#         st.subheader(
#             "Choice of the reference curve for the peak between 20 and 50 microns"
#         )
#         st.markdown(
#             """As explainend in the list of reference curves we can choose between three reference curves 
#                     (Limons grossier, Limon grossier-loess, Loess) for the peak between 20 and 50 $\\mu m$. Please 
#                     select bellow which reference curve to use in approximation."""
#         )
#         st.session_state["ref_20_50"] = st.radio(
#             "do no show",
#             [
#                 "All 3 at the same time",
#                 "Limons Grossiers",
#                 "Limons Grossiers-Loess",
#                 "Loess",
#             ],
#             label_visibility="hidden",
#         )

#         st.subheader(
#             "Algorithm to perform an approximation of X from the reference curves"
#         )
#         st.markdown(
#             """We're now going to find the best combinaisons of our reference curves to approximate 
#                     our observation X."""
#         )
#         st.markdown(
#             "- $M_{ref}$ is the matrix that contains the 8 reference curves.")
#         st.markdown(
#             "- $A_{ref}$ is the matrix that contains the best combinaisons to approximate each observation."
#         )
#         st.markdown("So we have the following problem :")
#         st.latex(
#             r""" A_{ref} = \arg \min_{A\geq 0} \Vert X-AM_{ref} \Vert_F^2 """)

#         if st.button("Perform estimations with reference curves"):

#             # Deleting other 20-50 microns that have not been selected
#             st.session_state["ref_curves_selected"] = st.session_state[
#                 "ref_curves"
#             ].copy()
#             st.session_state["rc_label"] = [
#                 "Argiles Fines",
#                 "Argiles Grossier",
#                 "Limons fins",
#                 "Sables Fins",
#                 "Sables grossiers",
#                 "Loess",
#                 "Limon grossiers",
#                 "Limons grossiers Loess",
#             ]
#             if st.session_state["ref_20_50"] == "Limons Grossiers":
#                 del st.session_state["ref_curves_selected"]["ref_LimonsGrossiersLoess"]
#                 del st.session_state["ref_curves_selected"]["ref_Loess"]
#                 st.session_state["rc_label"][5] = "Limon grossiers"
#                 st.session_state["rc_label"] = st.session_state["rc_label"][0:6]
#             elif st.session_state["ref_20_50"] == "Limons Grossiers-Loess":
#                 del st.session_state["ref_curves_selected"]["ref_LimonsGrossiers"]
#                 del st.session_state["ref_curves_selected"]["ref_Loess"]
#                 st.session_state["rc_label"][5] = "Limon grossiers Loess"
#                 st.session_state["rc_label"] = st.session_state["rc_label"][0:6]
#             elif st.session_state["ref_20_50"] == "Limons Grossiers-Loess":
#                 del st.session_state["ref_curves_selected"]["ref_LimonsGrossiersLoess"]
#                 del st.session_state["ref_curves_selected"]["ref_LimonsGrossiers"]
#                 st.session_state["rc_label"][5] = "Loess"
#                 st.session_state["rc_label"] = st.session_state["rc_label"][0:6]
#             # Do nothing if all 3 at the same time selected

#             # Gathering y from every reference curve into our M_ref matrix
#             M_ref = np.zeros(
#                 (
#                     len(st.session_state["ref_curves_selected"]),
#                     st.session_state["ref_curves_selected"][
#                         "ref_ArgilesClassiques"
#                     ].shape[1],
#                 )
#             )
#             for i, ref_curve in enumerate(st.session_state["ref_curves_selected"]):
#                 M_ref[int(i), :] = st.session_state["ref_curves_selected"][ref_curve][
#                     1, :
#                 ]

#             # A_ref is the mimimal argument of the optimisation problem
#             X = st.session_state["granulometrics"].to_numpy()
#             A_ref = X @ M_ref.T @ np.linalg.inv(M_ref @ M_ref.T)

#             # Performing minimalization with CVXPY to compare
#             # Declaration of our minimization variable A
#             A = cp.Variable((X.shape[0], M_ref.shape[0]))
#             # Constraint A to be positive
#             constraints = [A >= 0]
#             objective = cp.Minimize(
#                 cp.norm(X - A @ M_ref, "fro") ** 2
#             )  # Objective function
#             # problem = cp.Problem(objective)                        # optim without constraint to compare with our direct solution
#             # Definition of our problem
#             problem = cp.Problem(objective, constraints)
#             problem.solve(
#                 solver=cp.SCS, verbose=True, eps=1e-10, max_iters=10000
#             )  # Calling solver
#             A_ref_solv = A.value  # We get the result

#             # X_ref the approximations of our observations with ref_curves
#             X_ref = pd.DataFrame(
#                 A_ref_solv @ M_ref,
#                 columns=st.session_state["granulometrics"].columns,
#                 index=st.session_state["granulometrics"].index,
#             )

#             # df for the proportions
#             RC_areas = np.apply_along_axis(trapeze_areas, 1, M_ref).reshape(
#                 (A_ref_solv.shape[1])
#             )  # compute areas of each EM
#             Prop = A_ref_solv * RC_areas
#             Prop = np.apply_along_axis(lambda x: x / np.sum(x) * 100, 1, Prop)

#             # naming the columns of Prop with regards of where the peak is located for each EM
#             st.session_state["Prop_rc"] = pd.DataFrame(
#                 Prop,
#                 index=st.session_state["granulometrics"].index,
#                 columns=st.session_state["rc_label"],
#             )

#             # Approximation errors l2
#             err2_approx_rc = np.sum(
#                 np.linalg.norm(
#                     X_ref - st.session_state["granulometrics"], axis=1)
#             )
#             # L1-relativ norm of each approximations
#             st.session_state["Prop_rc"]["L1_rel_norm (%)"] = X_ref.apply(
#                 lambda row: L1_relative(row.values, row.name), axis=1
#             )
#             # L1-relativ mean
#             errL1_approx_rc = np.mean(
#                 st.session_state["Prop_rc"]["L1_rel_norm (%)"])

#             X_ref.index = X_ref.index.map(
#                 lambda x: f"r{x}")  # adding "r" before

#             # in this case we replace the old reference curves approximation
#             if st.session_state["rc_flag"]:
#                 for ind in X_ref.index:
#                     st.session_state["X-X_hat-X_ref"].loc[ind] = X_ref.loc[ind]

#             else:  # easier case : there isn't already a reference curves approximation
#                 st.session_state["X-X_hat-X_ref"] = pd.concat(
#                     [st.session_state["X-X_hat-X_ref"], X_ref], axis=0
#                 )
#                 st.session_state["rc_flag"] = True  # They are now result

#             st.success("Approximation succeed")
#             # Displaying approx errors
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.latex(r""" \sum_{i=1}^{n} \Vert x_i-{x_{ref,i}} \Vert_2 """)
#             with col2:
#                 st.metric(
#                     "sum of quadratic errors",
#                     value=f"{err2_approx_rc:.4}",
#                     label_visibility="visible",
#                 )
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.latex(
#                     r""" \sum_{i=1}^{n} \frac{\Vert x_i-{x_{ref,i}} \Vert_{L1}}{\Vert x_i \Vert_{L1}} """
#                 )
#             with col2:
#                 st.metric(
#                     "mean of L1-relative errors (%)",
#                     value=f"{errL1_approx_rc:.3}%",
#                     label_visibility="visible",
#                 )
# endregion


# region tab_robust
# with tab_robust:

#     col1_r, col2_r = st.columns(2)
#     with col1_r:
#         st.session_state['lambda_robust'] = st.number_input(
#             "penalization regularization weigth for L-2,1 norm of the residuals R", step=0.000001, value=st.session_state['lambda_robust'])
#     with col2_r:
#         st.session_state['beta_r'] = st.number_input(
#             "beta param for the beta-divergence, 0 : Itakura-Saito, 1 : Kullback-Leibler, 2 : Euclidean", step=0.1, value=st.session_state['beta_r'])

#     st.header("Algorithm")

#     if st.button("Lunch robust factorization"):
#         X = st.session_state['granulometrics'].to_numpy()
#         A, M, R, obj = robust_nmf(X,
#                                   rank=st.session_state['nb_end_members'],
#                                   beta=st.session_state['beta_r'],
#                                   init='random',
#                                   reg_val=st.session_state['lambda_robust'],
#                                   sum_to_one=0,
#                                   tol=1e-7,
#                                   max_iter=200)

#         # Estimations of our observations with only 8 EM
#         X_hat = pd.DataFrame(
#             A @ M, columns=st.session_state['granulometrics'].columns, index=st.session_state['granulometrics'].index)
#         X_hat.index = X_hat.index.map(lambda x: f"^{x}")  # adding "^-" before
#         st.session_state['X-X_hat-X_ref'] = pd.concat(
#             [st.session_state['granulometrics'], X_hat], axis=0)

#         st.success("Robust NMF succeed")

#         st.session_state['nmf_flag'] = True  # They are now result

#         st.header("Visualisaiton")

#         with st.expander("End-Members"):
#             fig = make_subplots(rows=4, cols=2, subplot_titles=[
#                                 f"End-Member {i}" for i in range(1, 9)])
#             for i in range(8):
#                 row = (i // 2) + 1
#                 col = (i % 2) + 1
#                 fig.add_trace(
#                     go.Scatter(
#                         x=st.session_state['granulometrics'].columns, y=M[i, :], mode='lines'),
#                     row=row, col=col
#                 )

#             fig.update_xaxes(type='log', tickformat=".1e", dtick=1,showgrid=True)
#             fig.update_yaxes(showgrid=True)
#             fig.update_layout(height=1300, width=700,
#                               title_text="End-members curves", showlegend=False)

#             st.plotly_chart(fig)

#         with st.expander("Proportions of EM in our observations"):
#             st.session_state['A_df'] = pd.DataFrame(A, index=st.session_state['granulometrics'].index, columns=[
#                                                     f'EM{i}' for i in range(1, st.session_state['nb_end_members']+1)])
#             st.session_state['A_df']['label'] = st.session_state['granulometrics'].index
#             fig = make_subplots(rows=st.session_state['nb_end_members']//2,
#                                 cols=1, vertical_spacing=0.05)
#             for i in range(st.session_state['nb_end_members']//2):

#                 first_em = 2*i+1
#                 second_em = 2*(i+1)

#                 fig.add_trace(
#                     go.Scatter(
#                         x=st.session_state['A_df'][f'EM{first_em}'],
#                         y=st.session_state['A_df'][f'EM{second_em}'],
#                         mode='markers',
#                         marker=dict(size=10, color=st.session_state['A_df']['label'].astype(
#                             'category').cat.codes, colorscale='rainbow'),
#                         text=st.session_state['A_df']['label'],
#                     ),
#                     row=i+1, col=1
#                 )
#                 fig.update_xaxes(
#                     title_text=f'End-member {first_em}', showgrid=False, gridcolor='LightGray', row=i+1, col=1)
#                 fig.update_yaxes(
#                     title_text=f'End-member {second_em}', showgrid=False, gridcolor='LightGray', row=i+1, col=1)

#             fig.update_layout(
#                 # Ajuster la hauteur de la figure en fonction du nombre de plots
#                 height=700 * st.session_state['nb_end_members']//2,
#                 title_text='Proprotions of End-members',
#                 showlegend=False  # Masquer la légende pour simplifier l'affichage
#             )

#             st.plotly_chart(fig)
# endregion
