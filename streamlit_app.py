import numpy as np
import pandas as pd
import streamlit as st
import time
import json
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import plotly.graph_objects as go
import cvxpy as cp
from multiprocessing import Pool
from functools import partial
import openpyxl

# from backends.numpy_functions import robust_nmf # --> for robust nmf algorithm
from plotly.subplots import make_subplots
import sys

sys.path.append("..")

st.set_page_config(page_title="NMF test", layout="wide")


st.title("Methods for gromulometric analysis")

# data_granulometry_03_06_24
# Loading observation data :
if "granulometrics" not in st.session_state:
    data = pd.read_excel(
        "data_granulometry_03_06_24.xlsx", sheet_name=0, header=0, index_col=2, engine='openpyxl' 
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
        "data_granulometry_03_06_24.xlsx", sheet_name=0, header=0, engine='openpyxl')    
       
# region initialisation of session variables
if "flag_comparaison_curves_importation" not in st.session_state:
    st.session_state["flag_comparaison_curves_importation"] = False
if "dd_flag" not in st.session_state:
    st.session_state["dd_flag"] = False
if "nb_end_members" not in st.session_state:
    st.session_state["nb_end_members"] = 8
if "X-X_hat-X_ref" not in st.session_state:
    st.session_state["X-X_hat-X_ref"] = st.session_state["granulometrics"].copy()
    # to have the same columns as approximations
    # st.session_state['X-X_hat-X_ref']['L1_rel_norm'] = '-'
if "rc_flag" not in st.session_state:
    st.session_state["rc_flag"] = False
if "nmf_flag" not in st.session_state:
    st.session_state["nmf_flag"] = False
if "a_W" not in st.session_state:
    st.session_state["a_W"] = 0.0
if "a_H" not in st.session_state:
    st.session_state["a_H"] = 0.0
if "ratio_l1" not in st.session_state:
    st.session_state["ratio_l1"] = 0.0
if "lambda_robust" not in st.session_state:
    st.session_state["lambda_robust"] = 1.0
if "beta_r" not in st.session_state:
    st.session_state["beta_r"] = 1.5
if "selected_label" not in st.session_state:
    st.session_state["selected_label"] = []
if "A_df" not in st.session_state:
    st.session_state["A_df"] = pd.DataFrame(
        np.zeros(
            (
                st.session_state["granulometrics"].to_numpy().shape[0],
                st.session_state["nb_end_members"],
            )
        )
    )
if "X-X_hat-X_ref" not in st.session_state:
    st.session_state["X-X_hat-X_ref"] = st.session_state["granulometrics"]
# endregion

# Loading reference curves
if "ref_curves" not in st.session_state:
    st.session_state["ref_curves"] = {}  # empty initialization
    st.session_state["ref_curves"]["ref_ArgilesFines"] = np.genfromtxt(
        "ref_curves/ref_ArgilesFines.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_ArgilesClassiques"] = np.genfromtxt(
        "ref_curves/ref_ArgilesClassiques.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_Alterites"] = np.genfromtxt(
        "ref_curves/ref_Alterites.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_SablesFins"] = np.genfromtxt(
        "ref_curves/ref_SablesFins.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_SablesGrossiers"] = np.genfromtxt(
        "ref_curves/ref_SablesGrossiers.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_Loess"] = np.genfromtxt(
        "ref_curves/ref_Loess.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_LimonsGrossiers"] = np.genfromtxt(
        "ref_curves/ref_LimonsGrossiers.csv", delimiter=","
    )
    st.session_state["ref_curves"]["ref_LimonsGrossiersLoess"] = np.genfromtxt(
        "ref_curves/ref_LimonsGrossiersLoess.csv", delimiter=","
    )
    st.session_state["scaled_ref_curves"] = (
        {}
    )  # for ref curves that will be on the same scale as observations (for NN-LASSO)
    # st.session_state['scaled_ref_curves']['abscisses'] = st.session_state["ref_curves"]["ref_ArgilesFines"][0,:] # --> abscisses not necessary
    st.session_state["scaled_ref_curves"]["ref_ArgilesFines"] = (
        st.session_state["ref_curves"]["ref_ArgilesFines"][1, :] * 0.00101501
    )
    st.session_state["scaled_ref_curves"]["ref_ArgilesClassiques"] = (
        st.session_state["ref_curves"]["ref_ArgilesClassiques"][1, :] * 0.0353269
    )
    st.session_state["scaled_ref_curves"]["ref_Alterites"] = (
        st.session_state["ref_curves"]["ref_Alterites"][1, :] * 0.107063
    )
    st.session_state["scaled_ref_curves"]["ref_SablesFins"] = (
        st.session_state["ref_curves"]["ref_SablesFins"][1, :] * 0.135512
    )
    st.session_state["scaled_ref_curves"]["ref_SablesGrossiers"] = (
        st.session_state["ref_curves"]["ref_SablesGrossiers"][1, :] * 0.0903764
    )
    st.session_state["scaled_ref_curves"]["ref_Loess"] = (
        st.session_state["ref_curves"]["ref_Loess"][1, :] * 0.107131
    )
    st.session_state["scaled_ref_curves"]["ref_LimonsGrossiers"] = (
        st.session_state["ref_curves"]["ref_LimonsGrossiers"][1, :] * 0.141878
    )
    st.session_state["scaled_ref_curves"]["ref_LimonsGrossiersLoess"] = (
        st.session_state["ref_curves"]["ref_LimonsGrossiersLoess"][1, :] * 0.0747438
    )

# region Other variables / functions
# Integral approximations with trapeze method for every observation


def trapeze_areas(x):
    return 0.5 * np.sum(
        (
            st.session_state["granulometrics"].columns[1:]
            - st.session_state["granulometrics"].columns[:-1]
        )
        * (x[1:] + x[:-1])
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
    "Argile Grossières": 7,
    "Alterites": 20,
    "Limons/Loess": 50,
    "Sables Fin": 100,
    "Sables Grossiers": 2400,
}
# endregion


tab_data,tab_continous_dict, tab_discrete_dict, tab_basic, tab_result = st.tabs(
    [
        "Granulometric data",
        "Continuous dictionary",
        "Discrete dictionnary",
        "Basic NMF (with penalization)",
        "Results",
    ]
)

with tab_data:
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:
        st.header("Presentation of our granulometric data")

        st.subheader("Our Data :s")
        st.dataframe(st.session_state['granulometrics'])

        st.subheader("Add new observation")

        with st.form(key='input_obs_form'):
            col1, col2 = st.columns([1,3])
            with col1:
                nb_line = st.number_input('Number of observations :', value = 1, min_value= 1, max_value = 10000)
            with col2:
                sep = st.radio("Separator", options = ['tabulation', 'space', 'comma', 'semicolon'])
            vecteur = st.text_area('Raw data (with metadata), separated by tabulations :', height=150)  # Utiliser un textarea pour plus de commodité
            submit_button = st.form_submit_button(label='Add')

            if submit_button:
                None

        st.subheader("Remove observation")
        st.markdown("Choose which label to remove and then click on \"Confirm\". Please reload the page to save change !")
        col1, col2 = st.columns(2)

        with col1:
            st.multiselect("",options=st.session_state["granulometrics"].index, key = "labels_to_remove", label_visibility='collapsed')

        with col2:
            if st.button("Confirm"):
                    # Select observation execpt those to be removed
                    st.session_state['raw_data'] = st.session_state['raw_data'][~st.session_state['raw_data']['Echt'].isin(st.session_state['labels_to_remove'])]
                    # Update the excel file 
                    st.session_state['raw_data'].to_excel("data_granulometry_03_06_24.xlsx", sheet_name=0, index = False)
                    st.success("Removing asked, now reload the page")
                    st.dataframe(st.session_state['raw_data'])

        st.header("Approximation of our observation by reference curves")
        st.markdown(
            """In this section we don't use any NMF algorithm. Instead we use reference curves 
                    that has been build from various curves of our data set that has been certified as 
                    pure by experts. We're going to use these curves to compare them with the end-members
                    we find and also to build differents approximations."""
        )

        st.subheader("List of reference curves")
        st.markdown(
            f"""There are 8 differents reference curves that are mainly characterised by the location 
                    of the peak on the x axis (diametre in $\\mu m$). You can see their plots below."""
        )

        with st.expander("List of reference curves :"):

            st.markdown(
                "We plot first the reference curve of the Argiles Fines (fine clay) because its peak is much greater than the others"
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_ArgilesFines"][0, :],
                    y=st.session_state["ref_curves"]["ref_ArgilesFines"][1, :],
                    mode="lines",
                    name="Argiles Fines (<1 microns)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_ArgilesClassiques"][0, :],
                    y=st.session_state["ref_curves"]["ref_ArgilesClassiques"][1, :],
                    mode="lines",
                    name="Argiles Grossières (1-7 microns)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_Alterites"][0, :],
                    y=st.session_state["ref_curves"]["ref_Alterites"][1, :],
                    mode="lines",
                    name="Alterites (7-20 microns)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_SablesFins"][0, :],
                    y=st.session_state["ref_curves"]["ref_SablesFins"][1, :],
                    mode="lines",
                    name="Sables Fins (50-100 microns)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_SablesGrossiers"][0, :],
                    y=st.session_state["ref_curves"]["ref_SablesGrossiers"][1, :],
                    mode="lines",
                    name="Sables Grossiers (>100 microns)",
                )
            )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_ArgilesFines"][0, :],
                    y=st.session_state["ref_curves"]["ref_ArgilesFines"][1, :],
                    mode="lines",
                    name="Argiles Fines (<1 microns)",
                )
            )
            fig.update_xaxes(type="log", tickformat=".1e", dtick=1, showgrid=True)
            fig.update_layout(
                height=500,
                showlegend=True,
                xaxis_title=" grain diametere (micrometers, log-scale)",
            )
            fig.update_traces(hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>")

            st.plotly_chart(fig)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_ArgilesClassiques"][0, :],
                    y=st.session_state["ref_curves"]["ref_ArgilesClassiques"][1, :],
                    mode="lines",
                    name="Argiles Grossières (1-7 microns)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_Alterites"][0, :],
                    y=st.session_state["ref_curves"]["ref_Alterites"][1, :],
                    mode="lines",
                    name="Alterites (7-20 microns)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_SablesFins"][0, :],
                    y=st.session_state["ref_curves"]["ref_SablesFins"][1, :],
                    mode="lines",
                    name="Sables Fins (50-100 microns)",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_SablesGrossiers"][0, :],
                    y=st.session_state["ref_curves"]["ref_SablesGrossiers"][1, :],
                    mode="lines",
                    name="Sables Grossiers (>100 microns)",
                )
            )

            fig.update_xaxes(type="log", tickformat=".1e", showgrid=True)
            fig.update_layout(
                # Ajuster la hauteur de la figure en fonction du nombre de plots
                height=500,
                xaxis_title=" grain diametere (micrometers, log-scale)",
            )
            fig.update_traces(hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>")

            st.plotly_chart(fig)

            st.markdown(
                "For the peak located between 20 and 50 $\\mu m$ we can choose between 3 different reference curves :"
            )
            st.markdown(" - Limons Grossier")
            st.markdown(" - Limons Grossier-Loess")
            st.markdown(" - Limons Loess")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_LimonsGrossiers"][0, :],
                    y=st.session_state["ref_curves"]["ref_LimonsGrossiers"][1, :],
                    mode="lines",
                    name="Limons Grossiers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_LimonsGrossiersLoess"][0, :],
                    y=st.session_state["ref_curves"]["ref_LimonsGrossiersLoess"][1, :],
                    mode="lines",
                    name="Limons Grossiers-Loess",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=st.session_state["ref_curves"]["ref_Loess"][0, :],
                    y=st.session_state["ref_curves"]["ref_Loess"][1, :],
                    mode="lines",
                    name="Loess",
                )
            )

            fig.update_xaxes(type="log", tickformat=".1e", dtick=1, showgrid=True)
            fig.update_layout(
                # Ajuster la hauteur de la figure en fonction du nombre de plots
                height=500,
                xaxis_title=" grain diametere (micrometers, log-scale)",
            )
            fig.update_traces(hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>")

            st.plotly_chart(fig)

        st.subheader(
            "Choice of the reference curve for the peak between 20 and 50 microns"
        )
        st.markdown(
            """As explainend in the list of reference curves we can choose between three reference curves 
                    (Limons grossier, Limon grossier-loess, Loess) for the peak between 20 and 50 $\\mu m$. Please 
                    select bellow which reference curve to use in approximation."""
        )
        st.session_state["ref_20_50"] = st.radio(
            "do no show",
            [
                "All 3 at the same time",
                "Limons Grossiers",
                "Limons Grossiers-Loess",
                "Loess",
            ],
            label_visibility="hidden",
        )

        st.subheader(
            "Algorithm to perform an approximation of X from the reference curves"
        )
        st.markdown(
            """We're now going to find the best combinaisons of our reference curves to approximate 
                    our observation X."""
        )
        st.markdown("- $M_{ref}$ is the matrix that contains the 8 reference curves.")
        st.markdown(
            "- $A_{ref}$ is the matrix that contains the best combinaisons to approximate each observation."
        )
        st.markdown("So we have the following problem :")
        st.latex(r""" A_{ref} = \arg \min_{A\geq 0} \Vert X-AM_{ref} \Vert_F^2 """)

        if st.button("Perform estimations with reference curves"):

            # Deleting other 20-50 microns that have not been selected
            st.session_state["ref_curves_selected"] = st.session_state[
                "ref_curves"
            ].copy()
            st.session_state["rc_label"] = [
                "Argiles Fines",
                "Argiles Grossier",
                "Alterites",
                "Sables Fins",
                "Sables grossiers",
                "Loess",
                "Limon grossiers",
                "Limons grossiers Loess",
            ]
            if st.session_state["ref_20_50"] == "Limons Grossiers":
                del st.session_state["ref_curves_selected"]["ref_LimonsGrossiersLoess"]
                del st.session_state["ref_curves_selected"]["ref_Loess"]
                st.session_state["rc_label"][5] = "Limon grossiers"
                st.session_state["rc_label"] = st.session_state["rc_label"][0:6]
            elif st.session_state["ref_20_50"] == "Limons Grossiers-Loess":
                del st.session_state["ref_curves_selected"]["ref_LimonsGrossiers"]
                del st.session_state["ref_curves_selected"]["ref_Loess"]
                st.session_state["rc_label"][5] = "Limon grossiers Loess"
                st.session_state["rc_label"] = st.session_state["rc_label"][0:6]
            elif st.session_state["ref_20_50"] == "Limons Grossiers-Loess":
                del st.session_state["ref_curves_selected"]["ref_LimonsGrossiersLoess"]
                del st.session_state["ref_curves_selected"]["ref_LimonsGrossiers"]
                st.session_state["rc_label"][5] = "Loess"
                st.session_state["rc_label"] = st.session_state["rc_label"][0:6]
            # Do nothing if all 3 at the same time selected

            # Gathering y from every reference curve into our M_ref matrix
            M_ref = np.zeros(
                (
                    len(st.session_state["ref_curves_selected"]),
                    st.session_state["ref_curves_selected"][
                        "ref_ArgilesClassiques"
                    ].shape[1],
                )
            )
            for i, ref_curve in enumerate(st.session_state["ref_curves_selected"]):
                M_ref[int(i), :] = st.session_state["ref_curves_selected"][ref_curve][
                    1, :
                ]

            # A_ref is the mimimal argument of the optimisation problem
            X = st.session_state["granulometrics"].to_numpy()
            A_ref = X @ M_ref.T @ np.linalg.inv(M_ref @ M_ref.T)

            # Performing minimalization with CVXPY to compare
            # Declaration of our minimization variable A
            A = cp.Variable((X.shape[0], M_ref.shape[0]))
            # Constraint A to be positive
            constraints = [A >= 0]
            objective = cp.Minimize(
                cp.norm(X - A @ M_ref, "fro") ** 2
            )  # Objective function
            # problem = cp.Problem(objective)                        # optim without constraint to compare with our direct solution
            # Definition of our problem
            problem = cp.Problem(objective, constraints)
            problem.solve(
                solver=cp.SCS, verbose=True, eps=1e-10, max_iters=10000
            )  # Calling solver
            A_ref_solv = A.value  # We get the result

            # X_ref the approximations of our observations with ref_curves
            X_ref = pd.DataFrame(
                A_ref_solv @ M_ref,
                columns=st.session_state["granulometrics"].columns,
                index=st.session_state["granulometrics"].index,
            )

            # df for the proportions
            RC_areas = np.apply_along_axis(trapeze_areas, 1, M_ref).reshape(
                (A_ref_solv.shape[1])
            )  # compute areas of each EM
            Prop = A_ref_solv * RC_areas
            Prop = np.apply_along_axis(lambda x: x / np.sum(x) * 100, 1, Prop)

            # naming the columns of Prop with regards of where the peak is located for each EM
            st.session_state["Prop_rc"] = pd.DataFrame(
                Prop,
                index=st.session_state["granulometrics"].index,
                columns=st.session_state["rc_label"],
            )

            # Approximation errors l2
            err2_approx_rc = np.sum(
                np.linalg.norm(X_ref - st.session_state["granulometrics"], axis=1)
            )
            # L1-relativ norm of each approximations
            st.session_state["Prop_rc"]["L1_rel_norm (%)"] = X_ref.apply(
                lambda row: L1_relative(row.values, row.name), axis=1
            )
            # L1-relativ mean
            errL1_approx_rc = np.mean(st.session_state["Prop_rc"]["L1_rel_norm (%)"])

            X_ref.index = X_ref.index.map(lambda x: f"r{x}")  # adding "r" before

            # in this case we replace the old reference curves approximation
            if st.session_state["rc_flag"]:
                for ind in X_ref.index:
                    st.session_state["X-X_hat-X_ref"].loc[ind] = X_ref.loc[ind]

            else:  # easier case : there isn't already a reference curves approximation
                st.session_state["X-X_hat-X_ref"] = pd.concat(
                    [st.session_state["X-X_hat-X_ref"], X_ref], axis=0
                )
                st.session_state["rc_flag"] = True  # They are now result

            st.success("Approximation succeed")
            # Displaying approx errors
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r""" \sum_{i=1}^{n} \Vert x_i-{x_{ref,i}} \Vert_2 """)
            with col2:
                st.metric(
                    "sum of quadratic errors",
                    value=f"{err2_approx_rc:.4}",
                    label_visibility="visible",
                )
            col1, col2 = st.columns(2)
            with col1:
                st.latex(
                    r""" \sum_{i=1}^{n} \frac{\Vert x_i-{x_{ref,i}} \Vert_{L1}}{\Vert x_i \Vert_{L1}} """
                )
            with col2:
                st.metric(
                    "mean of L1-relative errors (%)",
                    value=f"{errL1_approx_rc:.3}%",
                    label_visibility="visible",
                )


with tab_continous_dict:
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:
        st.header("Decomposition onto a continuous dictionnary")
        st.subheader("Graphique exemple of the method's interest")

        st.markdown(
            """ In order to show the interest of the B-Lasso method we're gonna try to decompose a signal that is made of two gaussians."""
        )

        st.markdown(
            f""" The first plot is the best approximation that is possible if we use a discrete dictionnary made 
                    by replicating curve and translate them by step : $\Delta = 1$. We can see that the approximation can't
                    overlap the observation because of because of this non-continuity."""
        )

        st.markdown(
            f"""On the other hand in the second plot we can see that the B-Lasso approximation is perfect."""
        )

        if not st.session_state["flag_comparaison_curves_importation"]:

            with open("ex_continuous_adv.json", "r") as file:
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
            title="Lasso on a discrete dictionnary",
            height=500,
            width=700,
        )
        fig.update_traces(hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>")
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
        fig.update_traces(hovertemplate="X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>")
        st.plotly_chart(fig)


with tab_discrete_dict:
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:
        st.header("Decomposition onto a discrete dictionnary")
        st.markdown(
            r"""In this section we try do decompose our observations with a discrete dictionnary 
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
                "Frank-Wolfe",
                "NN gready algo",
                "NN FW (proj with max 0)",
                "NN FW step 1 modification",
                "FISTA with backtracking",
                "Proximal Gradient with backtracking",
                "Proximal Gradient with constant step-size",
                "Projected gradient",
            ],
            key="nn_lasso_method",
        )

        col2, col3, col4 = st.columns([2, 1, 2])

        with col2:
            st.number_input(
                "Coefficient of penalization (lambda)",
                key="lambda_nn_lasso",
                value=10.0,
                min_value=0.0,
                step=1.0,
            )
        with col3:
            st.number_input("Precision for dual gap", key="p_dg", value=5.0)
        with col4:
            st.number_input(
                "Precision for complementary slackness", key="p_cs", value=5.0
            )

        if st.button("Run decomposition"):

            # region Creation of the discrete dictionnary
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
                        first_ind = mesurement_points.get_loc(materials[mat_prec])
                        # index of the last mesurement point for the interval of this rc
                        last_ind = mesurement_points.get_loc(materials[mat]) - 1
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
                        st.session_state["discrete_dictionnary"].iloc[i].to_numpy()
                    )

            # endregion

            # region functions

            M = np.transpose(st.session_state["discrete_dictionnary"].to_numpy())
            # hyper-parameters
            p_dg = 1
            p_cs = 1
            it_max = 1e3
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
                    0.5 * np.linalg.norm(x - np.dot(M, a), 2) ** 2
                    + lambda_ * np.linalg.norm(a, 1)
                    + 0.5 * (np.linalg.norm(x - u, 2) ** 2 - np.linalg.norm(x, 2) ** 2)
                )

            def stop_criterions(a, x, lambda_):
                u = rho(a, x, lambda_)
                cs = CS(a, u, lambda_)
                dg = DG(a, x, u, lambda_)
                # st.write(f"cs : {cs}")
                # st.write(f"dg : {dg}")
                return (
                    dg <= st.session_state["p_dg"]
                    and cs <= dg <= st.session_state["p_cs"]
                )
            
            # Non-negative least square with projected gradient method
            def NN_LS_proj_grad(Z, x_obs): 
                a_ls = np.ones(Z.shape[1])
                prec_LS = 1e-3
                it_max_LS = 1e4
                err = prec_LS+1
                it = 0
                ZtZ = np.dot(Z.T,Z)
                Zx = np.dot(Z.T,x_obs)
                rho_LS = 1 / (2 * np.real(np.max(np.linalg.eigvals(ZtZ)))) # 1 / Lipschitz constant of Z

                while err > prec_LS and it < it_max_LS:
                    a_ls_1 = np.maximum(0.0 ,a_ls - rho_LS * (np.dot(ZtZ,a_ls)-Zx))
                    err = np.linalg.norm(a_ls_1-a_ls)
                    a_ls = a_ls_1
                    it += 1

                if it == it_max_LS:
                    st.warning('Non convergence of NN-LS for approximation reconstruction ')

                return a_ls,it

            # Reconstruction of the observation with least square problem to avoid bias due to l1 penality
            def reconstruction_LS(a, x_obs):

                Z = M[:, a > 0.0]                        # construction of Z matix in the ||x-Zc||^2 minimisation
                if Z.shape[1] == 0:                      
                    return a, np.zeros_like(x_obs), 0    # case of empty solution
                a_tmp,it_ls = NN_LS_proj_grad(Z,x_obs)   # resolving least-square problem (a_tmp is a small vector)
                approx_ls = np.dot(Z,a_tmp)              # approximation construction
                a_ls = np.zeros(a.shape)                 # spare vector, usefull to label our reconstruction 
                k=0                                      #
                for i in range(len(a_ls)):
                    if a[i] > 0.0:
                        a_ls[i] = a_tmp[k]
                        if a_tmp[k] < 0.0 :
                            st.error(f"Warning !!!! : ls reconstruction produced negativ coefficient {a_tmp[k]=}")
                        k += 1

                return a_ls,approx_ls,it_ls

            # endregion

            # region algorithms

            if st.session_state["nn_lasso_method"] == "NN gready algo":

                def decomposition_algo(x, lambda_):
                    
                    # initialization
                    a0 = np.zeros(M.shape[1]) 
                    a0[random.randint(0,len(a0)-1)] = lambda_
                    a = a0
                    #a = np.zeros(M.shape[1])
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    it = 0

                    for i in range(M.shape[1]):
                    
                        # STEP 1
                        j_star = np.argmin(np.dot(MtM,a)-Mx)
                        a_1 = np.zeros(a.shape)
                        a_1[j_star] = 1

                        # STEP 2
                        q = (lambda_ * a_1 - a) 
                        if np.linalg.norm(q) == 0:
                            Γ = 0   # case where λ*a_1 = a
                        Γ = - np.dot(q,a) / (np.linalg.norm(q,2) ** 2)
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

                    a_ls, approx_ls, it_ls = reconstruction_LS(a,x)# reconstruction with least-square problem to cancel bias
                    return a_ls, approx_ls.flatten(), it, it_ls    # argmin, approx, and nb of iterations
                        
                        


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
                        return np.sign(z) * np.maximum(
                            np.abs(z) - np.full(z.shape, lambda_ / l),
                            np.full(z.shape, 0),
                        )

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
                        st.warning("Non-convergence for projected gradient method")

                    
                    a_ls, approx_ls, it_ls = reconstruction_LS(a,x)# reconstruction with least-square problem to cancel bias
                    return a_ls, approx_ls.flatten(), it, it_ls    # argmin, approx, and nb of iterations

            if st.session_state['nn_lasso_method'] == "NN FW step 1 modification":
                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])
                    w = np.linalg.norm(x, 2) ** 2 / (2 * lambda_)
                    w_bar = w
                    it = 0
                    M_prime = np.hstack((M,-M))
                    f = partial(f_global, x_=x, lambda_=lambda_)

                    # avoid having to compute it every time
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    # M_prime_x = np.dot(M_prime.T, x)
                    # MtM_prime = np.dot(M_prime.T,M_prime)

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
                            st.write("Zero case")
                            a_pre = np.zeros(a.shape)
                            w_pre = 0
                        else:
                            st.write("Non-zero case")
                            canonic_vec = np.zeros(a.shape)
                            canonic_vec[i_star] = 1
                            a_pre = canonic_vec * w_bar
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

                    a_ls, approx_ls, it_ls = reconstruction_LS(a,x)# reconstruction with least-square problem to cancel bias
                    return a_ls, approx_ls.flatten(), it, it_ls    # argmin, approx, and nb of iterations

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
                            if not np.array_equal(np.maximum(canonic_vec * np.sign(np.dot(m_i_star, r)) * w_bar,0),canonic_vec * np.sign(np.dot(m_i_star, r)) * w_bar) :
                                st.write(" Difference")
                            a_pre = np.maximum(canonic_vec * np.sign(np.dot(m_i_star, r)) * w_bar,0)
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
                        fegh = 3
                        st.warning("Non-convergence for Frank-Wolfe method")

                    a_ls, approx_ls, it_ls = reconstruction_LS(a,x)# reconstruction with least-square problem to cancel bias
                    return a_ls, approx_ls.flatten(), it, it_ls    # argmin, approx, and nb of iterations

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
                            a_pre = canonic_vec * np.sign(np.dot(m_i_star, r)) * w_bar
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
                        fegh = 3
                        st.warning("Non-convergence for Frank-Wolfe method")

                    a_ls, approx_ls, it_ls = reconstruction_LS(a,x)# reconstruction with least-square problem to cancel bias
                    return a_ls, approx_ls.flatten(), it, it_ls    # argmin, approx, and nb of iterations

            if (
                st.session_state["nn_lasso_method"]
                == "Proximal Gradient with backtracking"
            ):

                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    it = 0
                    Li = 1

                    def prox_l1(z, t):
                        return np.sign(z) * np.maximum(
                            np.abs(z) - np.full(z.shape, t), np.full(z.shape, 0)
                        )

                    while not stop_criterions(a, x, lambda_) and it < it_max:
                        a1 = prox_l1(a - (np.dot(MtM, a) - Mx) / Li, lambda_ / Li)
                        while f(a1, x, lambda_) > f_hat(a1, a, x, lambda_):
                            # st.write(f"Multplying Li by {eta}")
                            Li = eta * Li
                            a1 = prox_l1(a - (np.dot(MtM, a) - Mx) / Li, lambda_ / Li)
                        st.write(f"{Li = }")
                        a = a1
                        it += 1

                    if it == it_max:
                        st.warning("Non-convergence for projected gradient method")

                    a_ls, approx_ls, it_ls = reconstruction_LS(a,x)# reconstruction with least-square problem to cancel bias
                    return a_ls, approx_ls.flatten(), it, it_ls    # argmin, approx, and nb of iterations


            if st.session_state["nn_lasso_method"] == "Projected gradient":

                def decomposition_algo(x, lambda_):
                    a = np.zeros(M.shape[1])

                    # saving result to re_use it at each iterations
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    # each element of the vector is the penalization value
                    Lambda = np.full((a.shape), lambda_)
                    it = 0

                    while not stop_criterions(a, x, lambda_) and it < it_max:
                        a1 = np.maximum(0, a - 1 / L * (np.dot(MtM, a) - Mx + Lambda))
                        # st.write(a1)
                        err = np.linalg.norm(a1 - a)
                        a = a1.copy()
                        it += 1

                    if it == it_max:
                        fegh = 3
                        st.warning("Non-convergence for projected gradient method")

                    a_ls, approx_ls, it_ls = reconstruction_LS(a,x)# reconstruction with least-square problem to cancel bias
                    return a_ls, approx_ls.flatten(), it, it_ls    # argmin, approx, and nb of iterations


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
                            np.abs(z) - np.full(z.shape, t), np.full(z.shape, 0)
                        )

                    while not stop_criterions(a, x, lambda_) and it < it_max:
                        a1 = prox_l1(a - (np.dot(MtM, a) - Mx) / L, lambda_ / L)
                        a = a1
                        it += 1

                    if it == it_max:
                        st.warning("Non-convergence for projected gradient method")

                    a_ls, approx_ls, it_ls = reconstruction_LS(a,x)# reconstruction with least-square problem to cancel bias
                    return a_ls, approx_ls.flatten(), it, it_ls    # argmin, approx, and nb of iterations


            # endregion

            # region Decomposition

            st.session_state["Prop_nn_lasso"] = {}
            nb_it_total = 0
            nb_it_total_ls = 0
            start_time = time.time()

            compute_advancement = st.empty()
            k = 1
            nb_curves = len(st.session_state["granulometrics"])
            for index, row in st.session_state["granulometrics"].iterrows():
                with compute_advancement.container():
                    st.write(
                        f"approximation ({k} over {nb_curves}) -> sample : {index} "
                    )
                # compute decomposition for our observation x_i
                a_i, approx_i, it_i, it_ls_i = decomposition_algo(
                    row.to_numpy(), st.session_state["lambda_nn_lasso"]
                )
                nb_it_total += it_i
                nb_it_total_ls += it_ls_i
                st.session_state["X-X_hat-X_ref"].loc[f"dd-{index}"] = approx_i

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
                st.session_state["Prop_nn_lasso"][index] = prop_dict_i
                k += 1

            end_time = time.time()

            mean_it = (1.0 * nb_it_total) / len(st.session_state["granulometrics"])
            mean_it_ls = (1.0 * nb_it_total_ls) / len(st.session_state["granulometrics"])
            with compute_advancement.container():
                st.success("Decomposition computed with success")
                st.write(f"mean of iterations (decomposition algorithm) : {mean_it}")
                st.write(f"mean of iterations (reconstruction algorithm) : {mean_it_ls}")
                st.write(f"Execution time : {end_time-start_time:.2f} seconds")

            st.session_state["dd_flag"] = True
            # endregion

        # region scaled reference curves
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
                y=st.session_state["scaled_ref_curves"]["ref_ArgilesFines"],
                mode="lines",
                name="Argiles fines (<1 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["ref_ArgilesClassiques"],
                mode="lines",
                name="Argiles grossières (1-7 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["ref_Alterites"],
                mode="lines",
                name="Limons d'altération (7-20 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["ref_LimonsGrossiers"],
                mode="lines",
                name="Limons grossiers (20-50 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["ref_Loess"],
                mode="lines",
                name="Loess (20-50 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["ref_LimonsGrossiersLoess"],
                mode="lines",
                name="Loess sans limons (20-50 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["ref_SablesFins"],
                mode="lines",
                name="Sables fins (50-100 microns)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=abscisses,
                y=st.session_state["scaled_ref_curves"]["ref_SablesGrossiers"],
                mode="lines",
                name="Sables grossires (>100 microns)",
            )
        )

        fig.update_xaxes(type="log", tickformat=".1e", dtick=1, showgrid=True)
        fig.update_layout(
            height=500,
            showlegend=True,
            xaxis_title=" grain diametere (micrometers, log-scale)",
        )
        fig.update_traces(hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>")

        st.plotly_chart(fig)
        # endregion


with tab_basic:
    col01, col02, col03 = st.columns([1, 3, 1])

    with col02:
        st.header("Basic NMF")
        st.markdown(
            """ Perform the following factorisation :  $$ X \\thickapprox AM + \\varepsilon $$ 
                    by minimising the following expression as a function of $M$ and $A$
                    """
        )
        st.latex(
            r"""
                \Vert X-AM\Vert^2_{\beta-loss}+2l_{1_\text{ratio}}\left(\alpha_M m\Vert M \Vert_1 +\alpha_A n\Vert A \Vert_1 \right)+(1-l_1{_\text{ratio}})\left(\alpha_M m\Vert M \Vert_F^2 +\alpha_A n\Vert A \Vert_F^2 \right)
                """
        )

        st.markdown(
            """ You can choose the values of the parameters : $\\beta-loss$, $l_1{_\\text{ratio}}$, $\\alpha_M$ and $\\alpha_A$
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
                format="%d",
            )

        with col2:
            loss_choice = st.selectbox(
                "Beta_loss :",
                ("Frobenius (CD)", "Frobenius (MU)", "Kullback-Leibler (MU)"),
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
                "l1-l2 ratio", key="ratio_l1", min_value=0.0, max_value=1.0, format="%f"
            )
        with col4:
            st.number_input("penalization coef M", format="%f", key="a_W")
        with col5:
            st.number_input("penalization coef A", format="%f", key="a_A")

        st.header("Algorithm")

        if st.button("Lunch basic factorization"):
            X = st.session_state["granulometrics"].to_numpy()
            model = NMF(
                n_components=st.session_state["nb_end_members"],
                solver=st.session_state["solver"],
                beta_loss=st.session_state["loss"],
                init="random",
                l1_ratio=st.session_state["ratio_l1"],
                alpha_W=st.session_state["a_W"],
                alpha_H=st.session_state["a_H"],
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
                peak = np.argmax(M[i, :])
                for key, values in materials.items():
                    if peak < values:
                        prop_col_label[i] = key + f" ({peak})"
                        break
            st.session_state["Prop_nmf"] = pd.DataFrame(
                Prop,
                index=st.session_state["granulometrics"].index,
                columns=prop_col_label,
            )

            # Approximation errors l2
            err2_nmf = np.sum(
                np.linalg.norm(X_nmf - st.session_state["granulometrics"], axis=1)
            )
            # L1-relativ norm of each approximations
            st.session_state["Prop_nmf"]["L1_rel_norm (%)"] = X_nmf.apply(
                lambda row: L1_relative(row.values, row.name), axis=1
            )
            # L1-relativ mean
            errL1_nmf = np.mean(st.session_state["Prop_nmf"]["L1_rel_norm (%)"])

            # adding approximation to our result df
            X_nmf.index = X_nmf.index.map(lambda x: f"^{x}")  # adding "^-" before

            # in this case we replace the old nmf approximation
            if st.session_state["nmf_flag"]:
                for ind in X_nmf.index:
                    st.session_state["X-X_hat-X_ref"].loc[ind] = X_nmf.loc[ind]

            else:  # easier case : there isn't already a nmf approximation
                st.session_state["X-X_hat-X_ref"] = pd.concat(
                    [st.session_state["X-X_hat-X_ref"], X_nmf], axis=0
                )
                st.session_state["nmf_flag"] = True  # They are now result

            st.success("Approximation succeed")
            # Displaying approx errors
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r""" \sum_{i=1}^{n} \Vert x_i-{x_{ref,i}} \Vert_2 """)
            with col2:
                st.metric(
                    "sum of quadratic errors",
                    value=f"{err2_nmf:.4}",
                    label_visibility="visible",
                )
            col1, col2 = st.columns(2)
            with col1:
                st.latex(
                    r""" \sum_{i=1}^{n} \frac{\Vert x_i-{x_{ref,i}} \Vert_{L1}}{\Vert x_i \Vert_{L1}} """
                )
            with col2:
                st.metric(
                    "mean of L1-relative errors (%)",
                    value=f"{errL1_nmf:.3}%",
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

                fig.update_xaxes(type="log", tickformat=".1e", dtick=1, showgrid=True)
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
    st.header("Display observations to compare them")
    st.markdown(
        "##### Please perform approximation before trying to plot curves of approximations"
    )

    labels_obs = st.session_state["granulometrics"].index
    labels_approx_nmf = st.session_state["X-X_hat-X_ref"].index[
        st.session_state["X-X_hat-X_ref"].index.str.startswith(("^"))
    ]
    labels_approx_rc = st.session_state["X-X_hat-X_ref"].index[
        st.session_state["X-X_hat-X_ref"].index.str.startswith(("r"))
    ]

    # Selection of curves to plot
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.multiselect(
            "labels of the observations to diplay",
            options=labels_obs,
            key="selected_obs_labels",
        )
    with col2:
        st.toggle(
            "Display NMF-approximations",
            key="flag_nmf_approx",
            value=False,
            disabled=not st.session_state["nmf_flag"],
        )
    with col3:
        st.toggle(
            "Display approximations with reference curves",
            key="flag_rc_approx",
            value=False,
            disabled=not st.session_state["rc_flag"],
        )
    with col4:
        st.toggle(
            "Display approximations with discrete dictionnary (NN-LASSO)",
            key="flag_nnlasso_approx",
            value=False,
            disabled=not st.session_state["dd_flag"],
        )

    if st.session_state["nmf_flag"]:
        st.subheader("Proportions of EM (NMF-approximations) for selected observation")
        st.dataframe(
            st.session_state["Prop_nmf"].loc[st.session_state["selected_obs_labels"]]
        )

    if st.session_state["rc_flag"]:
        st.subheader(
            "Proportions of reference curve (approximations) for selected observation"
        )
        st.dataframe(
            st.session_state["Prop_rc"].loc[st.session_state["selected_obs_labels"]]
        )

    if st.session_state["dd_flag"]:
        st.subheader(
            "Proportions of curve in the discrete dicitonnary (approximations) for selected observation"
        )
        for label in st.session_state["selected_obs_labels"]:
            st.table(st.session_state["Prop_nn_lasso"][label])

    if st.button("Plots curves"):
        curves_and_approx = st.session_state["X-X_hat-X_ref"]
        fig = go.Figure()
        for label in st.session_state["selected_obs_labels"]:
            fig.add_trace(
                go.Scatter(
                    x=curves_and_approx.columns,
                    y=curves_and_approx.loc[label],
                    mode="lines",
                    name=label,
                )
            )
            if st.session_state["flag_nmf_approx"]:
                fig.add_trace(
                    go.Scatter(
                        x=curves_and_approx.columns,
                        y=curves_and_approx.loc[f"^{label}"],
                        mode="lines",
                        name=f"^{label}",
                    )
                )
            if st.session_state["flag_rc_approx"]:
                fig.add_trace(
                    go.Scatter(
                        x=curves_and_approx.columns,
                        y=curves_and_approx.loc[f"r{label}"],
                        mode="lines",
                        name=f"r{label}",
                    )
                )
            if st.session_state["flag_nnlasso_approx"]:
                fig.add_trace(
                    go.Scatter(
                        x=curves_and_approx.columns,
                        y=curves_and_approx.loc[f"dd-{label}"],
                        mode="lines",
                        name=f"dd-{label}",
                    )
                )

        fig.update_xaxes(type="log", tickformat=".1e", dtick=1, showgrid=True)
        fig.update_layout(
            height=800,
            width=1000,
            showlegend=True,
            xaxis_title=" grain diametere (micrometers, log-scale)",
        )
        fig.update_traces(hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>")

        st.plotly_chart(fig)


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
