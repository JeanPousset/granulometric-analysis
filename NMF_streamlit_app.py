import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import plotly.graph_objects as go
# from backends.numpy_functions import robust_nmf # --> for robust nmf algorithm
from plotly.subplots import make_subplots
import sys
sys.path.append("..")

st.set_page_config(page_title='NMF test', layout='centered')


st.title('Comparaison of differents NMF')

nb_end_members = 8  # to choose ?


# data_granulometry_03_06_24
# Loading observation data :
if 'granulometrics' not in st.session_state:
    data = pd.read_excel("data_granulometry_03_06_24.xls",
                         sheet_name=0, header=0, index_col=2)
    # Deletion of additional information
    data = data.drop(columns=['Dept', 'Commune', 'Type'])
    data = data.diff(axis=1)  # Cumulative difference on line item
    # Dividing by the log division of mesurement steps
    data.loc[:, data.columns != 0.03] = data.loc[:, data.columns != 0.03].div(np.log10(np.array(
        [float(col) for col in data.columns])[1:]/np.array([float(col) for col in data.columns])[:-1]), axis=1)
    data[[0.03]] = 0
    data = data.div(data.sum(axis=1), axis=0)*100  # Norming curves
    st.session_state['granulometrics'] = data

# region initialisation of session variables
if 'X-X_hat-X_ref' not in st.session_state:
    st.session_state['X-X_hat-X_ref'] = st.session_state['granulometrics']
if 'rc_flag' not in st.session_state:
    st.session_state['rc_flag'] = False
if 'nmf_flag' not in st.session_state:
    st.session_state['nmf_flag'] = False
if 'a_W' not in st.session_state:
    st.session_state['a_W'] = 0.0
if 'a_H' not in st.session_state:
    st.session_state['a_H'] = 0.0
if 'ratio_l1' not in st.session_state:
    st.session_state['ratio_l1'] = 0.0
if 'lambda_robust' not in st.session_state:
    st.session_state['lambda_robust'] = 1.0
if 'beta_r' not in st.session_state:
    st.session_state['beta_r'] = 1.5
if 'selected_label' not in st.session_state:
    st.session_state['selected_label'] = []
if 'A_df' not in st.session_state:
    st.session_state['A_df'] = pd.DataFrame(np.zeros(
        (st.session_state['granulometrics'].to_numpy().shape[0], nb_end_members)))
if 'X-X_hat-X_ref' not in st.session_state:
    st.session_state['X-X_hat-X_ref'] = st.session_state['granulometrics']
# endregion

# Loading reference curves
if 'ref_curves' not in st.session_state:
    st.session_state['ref_curves'] = {}  # empty initialization
    st.session_state['ref_curves']['ref_Alterites'] = np.genfromtxt(
        "ref_curves/ref_Alterites.csv", delimiter=',')
    st.session_state['ref_curves']['ref_ArgilesFines'] = np.genfromtxt(
        "ref_curves/ref_ArgilesFines.csv", delimiter=',')
    st.session_state['ref_curves']['ref_ArgilesClassiques'] = np.genfromtxt(
        "ref_curves/ref_ArgilesClassiques.csv", delimiter=',')
    st.session_state['ref_curves']['ref_LimonsGrossiers'] = np.genfromtxt(
        "ref_curves/ref_LimonsGrossiers.csv", delimiter=',')
    st.session_state['ref_curves']['ref_LimonsGrossiersLoess'] = np.genfromtxt(
        "ref_curves/ref_LimonsGrossiersLoess.csv", delimiter=',')
    st.session_state['ref_curves']['ref_Loess'] = np.genfromtxt(
        "ref_curves/ref_Loess.csv", delimiter=',')
    st.session_state['ref_curves']['ref_SablesFins'] = np.genfromtxt(
        "ref_curves/ref_SablesFins.csv", delimiter=',')
    st.session_state['ref_curves']['ref_SablesGrossiers'] = np.genfromtxt(
        "ref_curves/ref_SablesGrossiers.csv", delimiter=',')

tab_basic, tab_ref_expert, tab_result = st.tabs(
    ['basic NMF (with penalization)', 'Experimental references', 'Results'])

with tab_basic:

    st.header("Basic NMF")
    st.markdown(""" Perform the following factorisation :  $$ X \\thickapprox AM + \\varepsilon $$ 
                by minimising the following expression as a function of $M$ and $A$
                """)
    st.latex(r'''
            \Vert X-MA\Vert^2_{\beta-loss}+2l_{1_\text{ratio}}\left(\alpha_M m\Vert M \Vert_1 +\alpha_A n\Vert A \Vert_1 \right)+(1-l_1{_\text{ratio}})\left(\alpha_M m\Vert M \Vert_F^2 +\alpha_A n\Vert A \Vert_F^2 \right)
             ''')

    st.markdown(""" You can choose the values of the parameters : $\\beta-loss$, $l_1{_\\text{ratio}}$, $\\alpha_M$ and $\\alpha_A$
                """)

    st.header("Parameters for basic NMF")

    col1, col2 = st.columns([1, 2])

    with col1:
        loss_choice = st.selectbox(
            "Beta_loss :", ("Frobenius", "kullback-leibler"))
        if loss_choice == "kullback-leibler":
            st.session_state['loss'] = "kullback-leibler"
            st.session_state['solver'] = "mu"

        else:
            st.session_state['loss'] = "frobenius"
            st.session_state['solver'] = "cd"

    with col2:
        param = pd.DataFrame({'l1-l2 ratio': [st.session_state['ratio_l1']],
                              'penalization coef M': [st.session_state['a_W']],
                              'penalization coef A': [st.session_state['a_H']], })

        param_choosen = st.data_editor(param, hide_index=True, column_config={
            "l1-l2 ratio": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, format="%f"),
            "penalization coef M": st.column_config.NumberColumn(format="%f"),
            "penalization coef A": st.column_config.NumberColumn(format="%f")
        }
        )

    st.header("Algorithm")

    if st.button("Lunch basic factorization"):

        # parameters recovery
        st.session_state['ratio_l1'] = param_choosen.loc[0, 'l1-l2 ratio']
        st.session_state['a_W'] = param_choosen.loc[0, 'penalization coef M']
        st.session_state['a_H'] = param_choosen.loc[0, 'penalization coef A']

        X = st.session_state['granulometrics'].to_numpy()
        model = NMF(n_components=nb_end_members,
                    solver=st.session_state['solver'],
                    beta_loss=st.session_state['loss'],
                    init='random',
                    l1_ratio=st.session_state['ratio_l1'],
                    alpha_W=st.session_state['a_W'],
                    alpha_H=st.session_state['a_H'],
                    random_state=0, max_iter=15000)
        # Increase max_iter to get convergence
        A = model.fit_transform(X)
        M = model.components_

        # df for the approximation
        # Estimations of our observations with only 8 EM
        X_hat = pd.DataFrame(
            A @ M, columns=st.session_state['granulometrics'].columns, index=st.session_state['granulometrics'].index)

        # Approximation error calculation with sum of euclidean norm of Xi-Xi_hat
        err_approx = np.sum(np.linalg.norm(
            X_hat-st.session_state['granulometrics'], axis=1))

        # adding approximation to our result df
        X_hat.index = X_hat.index.map(lambda x: f"^{x}")  # adding "^-" before

        if st.session_state['nmf_flag']:    # in this case we replace the old nmf approximation
            for ind in X_hat.index:
                st.session_state['X-X_hat-X_ref'].loc[ind] = X_hat.loc[ind]

        else :  # easier case : there isn't already a nmf approximation
            st.session_state['X-X_hat-X_ref'] = pd.concat(
                [st.session_state['X-X_hat-X_ref'], X_hat], axis=0)
            st.session_state['nmf_flag'] = True  # They are now result


        st.success("NMF succeed")

        # Displaying approx error
        col1, col2 = st.columns(2)
        with col2:
            st.latex(r''' \sum_{i=1}^{853} \Vert x_i-\hat{x_i} \Vert_2 ''')
        with col1:
            st.metric("Approximation error", err_approx,
                      label_visibility="visible")

        st.header("Visualization")

        with st.expander("End-Members"):

            fig = make_subplots(rows=4, cols=2, subplot_titles=[
                                f"End-Member {i}" for i in range(1, 9)])
            for i in range(8):
                row = (i // 2) + 1
                col = (i % 2) + 1
                fig.add_trace(
                    go.Scatter(
                        x=st.session_state['granulometrics'].columns, y=M[i, :], mode='lines'),
                    row=row, col=col
                )

            fig.update_xaxes(type='log', tickformat=".1e",
                             dtick=1, showgrid=True)
            fig.update_yaxes(showgrid=True)
            fig.update_layout(height=1300, width=700,
                              title_text="End-members curves", showlegend=False)

            st.plotly_chart(fig)

        with st.expander("Proportions of EM in our observations"):
            st.session_state['A_df'] = pd.DataFrame(A, index=st.session_state['granulometrics'].index, columns=[
                                                    f'EM{i}' for i in range(1, nb_end_members+1)])
            st.session_state['A_df']['label'] = st.session_state['granulometrics'].index
            fig = make_subplots(rows=nb_end_members//2,
                                cols=1, vertical_spacing=0.05)
            for i in range(nb_end_members//2):

                first_em = 2*i+1
                second_em = 2*(i+1)

                fig.add_trace(
                    go.Scatter(
                        x=st.session_state['A_df'][f'EM{first_em}'],
                        y=st.session_state['A_df'][f'EM{second_em}'],
                        mode='markers',
                        marker=dict(size=10, color=st.session_state['A_df']['label'].astype(
                            'category').cat.codes, colorscale='rainbow'),
                        text=st.session_state['A_df']['label'],
                    ),
                    row=i+1, col=1
                )
                fig.update_xaxes(
                    title_text=f'End-member {first_em}', showgrid=False, gridcolor='LightGray', row=i+1, col=1)
                fig.update_yaxes(
                    title_text=f'End-member {second_em}', showgrid=False, gridcolor='LightGray', row=i+1, col=1)

            fig.update_layout(
                # Ajuster la hauteur de la figure en fonction du nombre de plots
                height=700 * nb_end_members//2,
                title_text='Proprotions of End-members',
                showlegend=False  # Masquer la légende pour simplifier l'affichage
            )

            st.plotly_chart(fig)


with tab_ref_expert:
    st.header("Approximation of our observation by reference curves")
    st.markdown("""In this section we don't use any NMF algorithm. Instead we use reference curves 
                that has been build from various curves of our data set that has been certified as 
                pure by experts. We're going to use these curves to compare them with the end-members
                we find and also to build differents approximations. """)

    st.subheader("List of reference curves")
    st.markdown(f"""There are 8 differents reference curves that are mainly characterised by the location 
                of the peak on the x axis (diametre in $\\mu m$). You can see their plots below.""")

    with st.expander("List of reference curves :"):

        st.markdown(
            "We plot first the reference curve of the Argiles Fines (fine clay) because its peak is much greater than the others")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_ArgilesFines'][0, :], y=st.session_state['ref_curves']
                                 ['ref_ArgilesFines'][1, :], mode='lines', name='Argiles Fines (<1 microns)'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_ArgilesClassiques'][0, :], y=st.session_state['ref_curves']
                                 ['ref_ArgilesClassiques'][1, :], mode='lines', name='Argiles Grossières (1-7 microns)'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_Alterites'][0, :], y=st.session_state['ref_curves']
                                 ['ref_Alterites'][1, :], mode='lines', name='Alterites (7-20 microns)'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_SablesFins'][0, :], y=st.session_state['ref_curves']
                                 ['ref_SablesFins'][1, :], mode='lines', name='Sables Fins (50-100 microns)'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_SablesGrossiers'][0, :], y=st.session_state['ref_curves']
                                 ['ref_SablesGrossiers'][1, :], mode='lines', name='Sables Grossiers (>100 microns)'))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_ArgilesFines'][0, :], y=st.session_state['ref_curves']
                                 ['ref_ArgilesFines'][1, :], mode='lines', name='Argiles Fines (<1 microns)'))
        fig.update_xaxes(type="log", tickformat=".1e", dtick=1)
        fig.update_layout(
            height=500,
            showlegend=True,
            xaxis_title=" grain diametere (micrometers, log-scale)"
        )
        st.plotly_chart(fig)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_ArgilesClassiques'][0, :], y=st.session_state['ref_curves']
                                 ['ref_ArgilesClassiques'][1, :], mode='lines', name='Argiles Grossières (1-7 microns)'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_Alterites'][0, :], y=st.session_state['ref_curves']
                                 ['ref_Alterites'][1, :], mode='lines', name='Alterites (7-20 microns)'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_SablesFins'][0, :], y=st.session_state['ref_curves']
                                 ['ref_SablesFins'][1, :], mode='lines', name='Sables Fins (50-100 microns)'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_SablesGrossiers'][0, :], y=st.session_state['ref_curves']
                                 ['ref_SablesGrossiers'][1, :], mode='lines', name='Sables Grossiers (>100 microns)'))

        fig.update_xaxes(type="log", tickformat=".1e")
        fig.update_layout(
            # Ajuster la hauteur de la figure en fonction du nombre de plots
            height=500,
            xaxis_title=" grain diametere (micrometers, log-scale)"
        )
        st.plotly_chart(fig)

        st.markdown(
            "For the peak located between 20 and 50 $\\mu m$ we can choose between 3 different reference curves :")
        st.markdown(" - Limons Grossier")
        st.markdown(" - Limons Grossier-Loess")
        st.markdown(" - Limons Loess")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_LimonsGrossiers'][0, :], y=st.session_state['ref_curves']
                                 ['ref_LimonsGrossiers'][1, :], mode='lines', name='Limons Grossiers'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_LimonsGrossiersLoess'][0, :], y=st.session_state['ref_curves']
                                 ['ref_LimonsGrossiersLoess'][1, :], mode='lines', name='Limons Grossiers-Loess'))
        fig.add_trace(go.Scatter(x=st.session_state['ref_curves']['ref_Loess'][0, :], y=st.session_state['ref_curves']
                                 ['ref_Loess'][1, :], mode='lines', name='Loess'))

        fig.update_xaxes(type="log", tickformat=".1e", dtick=1)
        fig.update_layout(
            # Ajuster la hauteur de la figure en fonction du nombre de plots
            height=500,
            xaxis_title=" grain diametere (micrometers, log-scale)"
        )
        st.plotly_chart(fig)

    st.subheader("Choice of the reference curve for the peak between 20 and 50 microns")
    st.markdown("""As explainend in the list of reference curves we can choose between three reference curves 
                (Limons grossier, Limon grossier-loess, Loess) for the peak between 20 and 50 $\\mu m$. Please 
                select bellow which reference curve to use in approximation.""")
    st.session_state['ref_20_50'] = st.radio("",['Limons Grossiers','Limons Grossiers-Loess','Loess'])

    st.subheader(
        "Algorithm to perform an approximation of X from the reference curves")
    st.markdown("""We're now going to find the best combinaisons of our reference curves to approximate 
                our observation X.""")
    st.markdown(
        "- $M_{ref}$ is the matrix that contains the 8 reference curves.")
    st.markdown(
        "- $A_{ref}$ is the matrix that contains the best combinaisons to approximate each observation.")
    st.markdown("So we have the following problem :")
    st.latex(r''' A_{ref} = \arg \min_A \Vert X-AM_{ref} \Vert_F^2 ''')

    if st.button('Perform estimations with reference curves'):

        # Deleting other 20-50 microns that have not been selected
        st.session_state['ref_curves_selected'] = st.session_state['ref_curves'].copy()
        if st.session_state['ref_20_50'] == 'Limons Grossiers':
            del st.session_state['ref_curves_selected']["ref_LimonsGrossiersLoess"]
            del st.session_state['ref_curves_selected']["ref_Loess"]
        elif st.session_state['ref_20_50'] == 'Limons Grossiers-Loess':
            del st.session_state['ref_curves_selected']["ref_LimonsGrossiers"]
            del st.session_state['ref_curves_selected']["ref_Loess"]
        else :
            del st.session_state['ref_curves_selected']["ref_LimonsGrossiersLoess"]
            del st.session_state['ref_curves_selected']["ref_LimonsGrossiers"]


        # Gathering y from every reference curve into our M_ref matrix
        M_ref = np.zeros(
            (len(st.session_state['ref_curves_selected']), st.session_state['ref_curves_selected']['ref_ArgilesClassiques'].shape[1]))
        for i, ref_curve in enumerate(st.session_state['ref_curves_selected']):
            M_ref[int(i), :] = st.session_state['ref_curves_selected'][ref_curve][1, :]

        # A_ref is the mimimal argument of the optimisation problem
        X = st.session_state['granulometrics'].to_numpy()
        A_ref = X @ M_ref.T @ np.linalg.inv(M_ref @ M_ref.T)

        # X_ref the approximations of our observations with ref_curves
        X_ref = pd.DataFrame(
            A_ref @ M_ref, columns=st.session_state['granulometrics'].columns, index=st.session_state['granulometrics'].index)

        # Approximation error calculation with sum of euclidean norm of Xi-Xi_hat
        err_approx_rc = np.sum(np.linalg.norm(
            X_ref-st.session_state['granulometrics'], axis=1))
        X_ref.index = X_ref.index.map(lambda x: f"r{x}")  # adding "r" before

        if st.session_state['rc_flag']:    # in this case we replace the old reference curves approximation
            for ind in X_ref.index:
                st.session_state['X-X_hat-X_ref'].loc[ind] = X_ref.loc[ind]

        else :  # easier case : there isn't already a reference curves approximation
            st.session_state['X-X_hat-X_ref'] = pd.concat(
                [st.session_state['X-X_hat-X_ref'], X_ref], axis=0)
            st.session_state['rc_flag'] = True  # They are now result
        

        st.success("Approximation succeed")
        # Displaying approx error
        col1, col2 = st.columns(2)
        with col2:
            st.latex(r''' \sum_{i=1}^{853} \Vert x_i-{x_{ref,i}} \Vert_2 ''')
        with col1:
            st.metric("Approximation error", err_approx_rc ,
                        label_visibility="visible")

with tab_result:
    st.header("Display observations to compare them")
    st.markdown(
        "##### Please perform approximation before trying to plot curves of approximations")

    labels_obs = st.session_state['granulometrics'].index
    labels_approx = st.session_state['X-X_hat-X_ref'].index[st.session_state['X-X_hat-X_ref'].index.str.startswith((
        '^', 'r'))]

    # Selection of curves to plot
    col1, col2 = st.columns(2)
    with col1:
        st.session_state['selected_obs_labels'] = st.multiselect(
            "labels of the observations to diplay", options=labels_obs)
    with col2:
        st.session_state['selected_approx_labels'] = st.multiselect(
            "labels of the approximations to diplay", options=labels_approx)

    if st.session_state['nmf_flag']:
        st.subheader("Proportions of EM for selected observations")
        st.table(
            st.session_state['A_df'].loc[st.session_state['selected_obs_labels']])

    if st.button('Plots curves'):
        fig, ax = plt.subplots()

        for label in st.session_state['selected_obs_labels']+st.session_state['selected_approx_labels']:
            ax.semilogx(st.session_state['X-X_hat-X_ref'].columns,
                        st.session_state['X-X_hat-X_ref'].loc[label], label=label)

        ax.set_xlabel('micrometers')
        ax.set_title('granulometrics curves of selected observations')
        ax.legend()
        st.pyplot(fig)

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
#                                   rank=nb_end_members,
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
#                                                     f'EM{i}' for i in range(1, nb_end_members+1)])
#             st.session_state['A_df']['label'] = st.session_state['granulometrics'].index
#             fig = make_subplots(rows=nb_end_members//2,
#                                 cols=1, vertical_spacing=0.05)
#             for i in range(nb_end_members//2):

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
#                 height=700 * nb_end_members//2,
#                 title_text='Proprotions of End-members',
#                 showlegend=False  # Masquer la légende pour simplifier l'affichage
#             )

#             st.plotly_chart(fig)
# endregion
