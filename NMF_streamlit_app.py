import numpy as np
import pandas as pd
import streamlit as st
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import plotly.graph_objects as go
import cvxpy as cp
# from backends.numpy_functions import robust_nmf # --> for robust nmf algorithm
from plotly.subplots import make_subplots
import sys
sys.path.append("..")

st.set_page_config(page_title='NMF test', layout='wide')


st.title('Comparaison of differents NMF')


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
if 'dd_flag' not in st.session_state:
    st.session_state['dd_flag'] = False
if 'nb_end_members' not in st.session_state:
    st.session_state['nb_end_members'] = 8
if 'X-X_hat-X_ref' not in st.session_state:
    st.session_state['X-X_hat-X_ref'] = st.session_state['granulometrics'].copy()
    # to have the same columns as approximations
    # st.session_state['X-X_hat-X_ref']['L1_rel_norm'] = '-'
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
        (st.session_state['granulometrics'].to_numpy().shape[0], st.session_state['nb_end_members'])))
if 'X-X_hat-X_ref' not in st.session_state:
    st.session_state['X-X_hat-X_ref'] = st.session_state['granulometrics']
# endregion

# Loading reference curves
if 'ref_curves' not in st.session_state:
    st.session_state['ref_curves'] = {}  # empty initialization
    st.session_state['ref_curves']['ref_ArgilesFines'] = np.genfromtxt(
        "ref_curves/ref_ArgilesFines.csv", delimiter=',')
    st.session_state['ref_curves']['ref_ArgilesClassiques'] = np.genfromtxt(
        "ref_curves/ref_ArgilesClassiques.csv", delimiter=',')
    st.session_state['ref_curves']['ref_Alterites'] = np.genfromtxt(
        "ref_curves/ref_Alterites.csv", delimiter=',')
    st.session_state['ref_curves']['ref_SablesFins'] = np.genfromtxt(
        "ref_curves/ref_SablesFins.csv", delimiter=',')
    st.session_state['ref_curves']['ref_SablesGrossiers'] = np.genfromtxt(
        "ref_curves/ref_SablesGrossiers.csv", delimiter=',')
    st.session_state['ref_curves']['ref_Loess'] = np.genfromtxt(
        "ref_curves/ref_Loess.csv", delimiter=',')
    st.session_state['ref_curves']['ref_LimonsGrossiers'] = np.genfromtxt(
        "ref_curves/ref_LimonsGrossiers.csv", delimiter=',')
    st.session_state['ref_curves']['ref_LimonsGrossiersLoess'] = np.genfromtxt(
        "ref_curves/ref_LimonsGrossiersLoess.csv", delimiter=',')
# region Other variables / functions
# Integral approximations with trapeze method for every observation


def trapeze_areas(x):
    return 0.5*np.sum((st.session_state['granulometrics'].columns[1:]-st.session_state['granulometrics'].columns[:-1])*(x[1:]+x[:-1]))


def L1_relative(x_approx, obs_index):
    x_obs = st.session_state['granulometrics'].loc[obs_index].to_numpy()
    numerator = trapeze_areas(np.abs(x_approx-x_obs))
    denominator = trapeze_areas(x_obs)
    return numerator/denominator*100
# Calculate quotient between ||x-x_approx||_L1 et ||x||L1


# list of labelized materials regarding the location of the peak
materials = {
    'Autre': 0.03,
    'Argile Fines': 1,
    'Argile Grossières': 7,
    'Alterites': 20,
    'Limons/Loess': 50,
    'Sables Fin': 100,
    'Sables Grossiers': 2400
}
# endregion


tab_discrete_dict, tab_basic, tab_ref_expert, tab_result = st.tabs(
    ['Discrete dictionnary', 'Basic NMF (with penalization)', 'Experimental references', 'Results'])

with tab_discrete_dict:
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:
        st.header("Decomposition onto a discrete dictionnary")
        st.markdown(r"""In this section we try do decompose our observations with a discrete dictionnary 
                    of unimodale curves $(\mathcal{M})$ obtained by duplicating the reference curves and shifting the spike
                     locations. To obtain these decomposition we're going to resolve the following optimization 
                    problem for each observation $x$ :""")
        st.latex(
            r'''\arg \min_{a \in \mathbb{R}_+^{\vert \mathcal{M} \vert}} \frac{1}{2}\Vert x-\mathcal{M}a \Vert_2^2 + \lambda\Vert a\Vert_1''')

        st.markdown(r"""Where $\mathcal{M}$ the dictionnary is presentend as a matrix where each row contains an unimodal curve.
                    We reconize here a LASSO problem except that the variable $a$ is non-negative. For this kind of problem (NN-LASSO), there
                     are several methods we can use. Please select bellow wich one you want.""")

        col1, col2 = st.columns(2)

        with col1:
            st.selectbox("Choose the resolution method", options=['Proximal Gradient with constant step-size iteration',
                         'Frank-Wolfe', 'Projected gradient', 'Proximal Gradient with backtracking iteration'], key='nn_lasso_method')
        with col2:
            st.number_input("Coefficient of penalization (lambda)",
                            key='lambda_nn_lasso', value=5.0, min_value=0.0, step=1.0)

        if st.button("Run decomposition"):

            # region Creation of the discrete dictionnary
            mesurement_points = st.session_state['granulometrics'].columns
            st.session_state['discrete_dictionnary'] = pd.DataFrame(
                columns=mesurement_points)

            for rc_name, rc in st.session_state['ref_curves'].items():

                # Find the first and last index of the measurement points of the peak-interval for the material of the reference curve
                peak_ind = np.argmax(rc[1, :])
                peak_loc = mesurement_points[peak_ind]
                mat_rc = ""         # name of the material for the interval of our ref-curve's peak
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
                        last_ind = mesurement_points.get_loc(materials[mat])-1
                        rel_peak_ind = peak_ind-first_ind
                        rel_last_ind = last_ind-first_ind
                        break

                for i in range(rel_peak_ind+1):
                    # Shifting ref curve to the left by dropping the i first values and adding i zeros at the end
                    duplicate = np.pad(rc[1, i:], (0, i),
                                       mode='constant', constant_values=0)
                    st.session_state['discrete_dictionnary'].loc[
                        f"{rc_name} ({mesurement_points[peak_ind-i]})"] = duplicate

                for i in range(1, rel_last_ind-rel_peak_ind+1):
                    # Shifting ref curve to the right by dropping the i last values and adding i zeros at the begining
                    duplicate = np.pad(
                        rc[1, :-i], (i, 0), mode='constant', constant_values=0)
                    # st.write(peak_ind+i)
                    st.session_state['discrete_dictionnary'].loc[
                        f"{rc_name} ({mesurement_points[peak_ind+i]})"] = duplicate

                # Precompute areas of curves in order to calculate proportions after
                st.session_state['aeras_dd_curves'] = np.zeros(
                    st.session_state['discrete_dictionnary'].shape[0])
                for i in range(st.session_state['discrete_dictionnary'].shape[0]):
                    st.session_state['aeras_dd_curves'][i] = trapeze_areas(
                        st.session_state['discrete_dictionnary'].iloc[i].to_numpy())

            # endregion

            # region algorithms

            M = np.transpose(
                st.session_state['discrete_dictionnary'].to_numpy())
            # hyper-parameters
            prec = 1e-3
            it_max = 1e4
            MtM = np.dot(M.T, M)  # saving result to optimize
            rho = 1/(np.real(np.max(np.linalg.eigvals(MtM))))    # 1/L

            if st.session_state['nn_lasso_method'] == 'Projected gradient':

                def decomposition_algo(x):
                    a = np.random.rand(M.shape[1])

                    # saving result to re_use it at each iterations
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    # each element of the vector is the penalization value
                    Lambda = np.full(
                        (a.shape), st.session_state['lambda_nn_lasso'])
                    err = prec+1
                    it = 0

                    while err > prec and it < it_max:
                        a1 = np.maximum(0, a-rho*(np.dot(MtM, a)-Mx+Lambda))
                        # st.write(a1)
                        err = np.linalg.norm(a1-a)
                        a = a1.copy()
                        it = it+1

                    if it == it_max:
                        fegh = 3
                        st.warning(
                            "Non-convergence for projected gradient method")

                    # argmin, approx, and nb of iterations
                    return a, np.dot(M, a).flatten(), it

            if st.session_state['nn_lasso_method'] == 'Proximal Gradient with constant step-size iteration':
                def decomposition_algo(x):
                    a = np.random.rand(M.shape[1])

                    # saving result to re_use it at each iterations
                    Mx = np.dot(M.T, x).reshape(a.shape)
                    # each element of the vector is the penalization value
                    Lambda = np.full(
                        (a.shape), st.session_state['lambda_nn_lasso'])
                    err = prec+1
                    it = 0

                    def prox_l1(z):
                        return np.sign(z)*np.maximum(np.abs(z)-Lambda*rho, 0)

                    while err > prec and it < it_max:
                        a1 = prox_l1(a-rho*(np.dot(MtM, a)-Mx))
                        err = np.linalg.norm(a1-a)
                        a = a1.copy()

                        it = it+1

                    if it == it_max:
                        st.warning(
                            "Non-convergence for projected gradient method")

                    # argmin, approx, and nb of iterations
                    return a, np.dot(M, a).flatten(), it

            # endregion

            # region Decomposition
            st.session_state['Prop_nn_lasso'] = {}
            nb_it_total = 0
            start_time = time.time()

            for index, row in st.session_state['granulometrics'].iterrows():
                # compute decomposition for our observation x_i
                a_i, approx_i, it_i = decomposition_algo(row.to_numpy())
                nb_it_total = nb_it_total + it_i
                st.session_state['X-X_hat-X_ref'].loc[f"dd-{index}"] = approx_i

                # saving coefficient that are non-zero
                prop_dict_i = {}
                sum_aera_i = 0.0
                for i in range(a_i.shape[0]):
                    if a_i[i] > 0:
                        prop_dict_i[st.session_state['discrete_dictionnary'].index[i]
                                    ] = a_i[i]*st.session_state['aeras_dd_curves'][i]
                        sum_aera_i = sum_aera_i + \
                            prop_dict_i[st.session_state['discrete_dictionnary'].index[i]]
                for curve in prop_dict_i:
                    prop_dict_i[curve] = prop_dict_i[curve]*100/sum_aera_i
                st.session_state['Prop_nn_lasso'][index]=prop_dict_i

            end_time = time.time()

            mean_it = (1.0*nb_it_total)/len(st.session_state['granulometrics'])
            st.write(mean_it)
            st.write(f"Execution time : {end_time-start_time} seconds")
            st.success("Decomposition computed with success")
            st.session_state['dd_flag'] = True


with tab_basic:
    col01, col02, col03 = st.columns([1, 3, 1])

    with col02:
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

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

        with col1:
            st.number_input("nb of components", key='nb_end_members',
                            min_value=2, max_value=100, step=1, format="%d")

        with col2:
            loss_choice = st.selectbox(
                "Beta_loss :", ("Frobenius", "kullback-leibler"))
            if loss_choice == "kullback-leibler":
                st.session_state['loss'] = "kullback-leibler"
                st.session_state['solver'] = "mu"

            else:
                st.session_state['loss'] = "frobenius"
                st.session_state['solver'] = "cd"

        with col3:
            st.number_input("l1-l2 ratio", key='ratio_l1',
                            min_value=0.0, max_value=1.0, format="%f")
        with col4:
            st.number_input("penalization coef M", format="%f", key='a_W')
        with col5:
            st.number_input("penalization coef A", format="%f", key='a_A')

        st.header("Algorithm")

        if st.button("Lunch basic factorization"):
            X = st.session_state['granulometrics'].to_numpy()
            model = NMF(n_components=st.session_state['nb_end_members'],
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
            # Estimations of our observations with only the EMs
            X_nmf = pd.DataFrame(
                A @ M, columns=st.session_state['granulometrics'].columns, index=st.session_state['granulometrics'].index)

            # df for the proportions
            EM_areas = np.apply_along_axis(trapeze_areas, 1, M).reshape(
                (A.shape[1]))  # compute areas of each EM
            Prop = A * EM_areas
            Prop = np.apply_along_axis(lambda x: x/np.sum(x)*100, 1, Prop)

            # naming the columns of Prop with regards of where the peak is located for each EM
            # We have to use temporary variable because we can't change columns name one by one
            prop_col_label = [0]*Prop.shape[1]
            for i in range(M.shape[0]):
                peak = np.argmax(M[i, :])
                for key, values in materials.items():
                    if peak < values:
                        prop_col_label[i] = key+f" ({peak})"
                        break
            st.session_state['Prop_nmf'] = pd.DataFrame(
                Prop, index=st.session_state['granulometrics'].index, columns=prop_col_label)

            # Approximation errors l2
            err2_nmf = np.sum(np.linalg.norm(
                X_nmf-st.session_state['granulometrics'], axis=1))
            # L1-relativ norm of each approximations
            st.session_state['Prop_nmf']['L1_rel_norm (%)'] = X_nmf.apply(
                lambda row: L1_relative(row.values, row.name), axis=1)
            # L1-relativ mean
            errL1_nmf = np.mean(
                st.session_state['Prop_nmf']['L1_rel_norm (%)'])

            # adding approximation to our result df
            X_nmf.index = X_nmf.index.map(
                lambda x: f"^{x}")  # adding "^-" before

            # in this case we replace the old nmf approximation
            if st.session_state['nmf_flag']:
                for ind in X_nmf.index:
                    st.session_state['X-X_hat-X_ref'].loc[ind] = X_nmf.loc[ind]

            else:  # easier case : there isn't already a nmf approximation
                st.session_state['X-X_hat-X_ref'] = pd.concat(
                    [st.session_state['X-X_hat-X_ref'], X_nmf], axis=0)
                st.session_state['nmf_flag'] = True  # They are now result

            st.success("Approximation succeed")
            # Displaying approx errors
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r''' \sum_{i=1}^{n} \Vert x_i-{x_{ref,i}} \Vert_2 ''')
            with col2:
                st.metric("sum of quadratic errors", value=f"{err2_nmf:.4}",
                          label_visibility="visible")
            col1, col2 = st.columns(2)
            with col1:
                st.latex(
                    r''' \sum_{i=1}^{n} \frac{\Vert x_i-{x_{ref,i}} \Vert_{L1}}{\Vert x_i \Vert_{L1}} ''')
            with col2:
                st.metric("mean of L1-relative errors (%)", value=f"{errL1_nmf:.3}%",
                          label_visibility="visible")

            st.header("Visualization")

            with st.expander("End-Members"):

                fig = make_subplots(rows=st.session_state['nb_end_members']//2+st.session_state['nb_end_members'] % 2, cols=2, subplot_titles=[
                                    f"End-Member {i}" for i in range(1, st.session_state['nb_end_members']+1)])
                for i in range(st.session_state['nb_end_members']):
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
                fig.update_traces(
                    hovertemplate='X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>')

                st.plotly_chart(fig)

            with st.expander("Proportions of EM in our observations"):
                st.session_state['A_df'] = pd.DataFrame(A, index=st.session_state['granulometrics'].index, columns=[
                                                        f'EM{i}' for i in range(1, st.session_state['nb_end_members']+1)])
                st.session_state['A_df']['label'] = st.session_state['granulometrics'].index
                fig = make_subplots(rows=st.session_state['nb_end_members']//2,
                                    cols=1, vertical_spacing=0.05)
                for i in range(st.session_state['nb_end_members']//2):

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
                    height=700 * st.session_state['nb_end_members']//2,
                    title_text='Proprotions of End-members',
                    showlegend=False  # Masquer la légende pour simplifier l'affichage
                )

                st.plotly_chart(fig)

with tab_ref_expert:
    col01, col02, col03 = st.columns([1, 3, 1])
    with col02:
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
            fig.update_xaxes(type="log", tickformat=".1e",
                             dtick=1, showgrid=True)
            fig.update_layout(
                height=500,
                showlegend=True,
                xaxis_title=" grain diametere (micrometers, log-scale)"
            )
            fig.update_traces(
                hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>')

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

            fig.update_xaxes(type="log", tickformat=".1e", showgrid=True)
            fig.update_layout(
                # Ajuster la hauteur de la figure en fonction du nombre de plots
                height=500,
                xaxis_title=" grain diametere (micrometers, log-scale)"
            )
            fig.update_traces(
                hovertemplate='X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>')

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

            fig.update_xaxes(type="log", tickformat=".1e",
                             dtick=1, showgrid=True)
            fig.update_layout(
                # Ajuster la hauteur de la figure en fonction du nombre de plots
                height=500,
                xaxis_title=" grain diametere (micrometers, log-scale)"
            )
            fig.update_traces(
                hovertemplate='X: %{x:.0f}<br>Y: %{y:.2f}<extra></extra>')

            st.plotly_chart(fig)

        st.subheader(
            "Choice of the reference curve for the peak between 20 and 50 microns")
        st.markdown("""As explainend in the list of reference curves we can choose between three reference curves 
                    (Limons grossier, Limon grossier-loess, Loess) for the peak between 20 and 50 $\\mu m$. Please 
                    select bellow which reference curve to use in approximation.""")
        st.session_state['ref_20_50'] = st.radio(
            "", ['All 3 at the same time', 'Limons Grossiers', 'Limons Grossiers-Loess', 'Loess'])

        st.subheader(
            "Algorithm to perform an approximation of X from the reference curves")
        st.markdown("""We're now going to find the best combinaisons of our reference curves to approximate 
                    our observation X.""")
        st.markdown(
            "- $M_{ref}$ is the matrix that contains the 8 reference curves.")
        st.markdown(
            "- $A_{ref}$ is the matrix that contains the best combinaisons to approximate each observation.")
        st.markdown("So we have the following problem :")
        st.latex(
            r''' A_{ref} = \arg \min_{A\geq 0} \Vert X-AM_{ref} \Vert_F^2 ''')

        if st.button('Perform estimations with reference curves'):

            # Deleting other 20-50 microns that have not been selected
            st.session_state['ref_curves_selected'] = st.session_state['ref_curves'].copy(
            )
            st.session_state['rc_label'] = ['Argiles Fines', 'Argiles Grossier', 'Alterites',
                                            'Sables Fins', 'Sables grossiers', 'Loess', 'Limon grossiers', 'Limons grossiers Loess']
            if st.session_state['ref_20_50'] == 'Limons Grossiers':
                del st.session_state['ref_curves_selected']["ref_LimonsGrossiersLoess"]
                del st.session_state['ref_curves_selected']["ref_Loess"]
                st.session_state['rc_label'][5] = 'Limon grossiers'
                st.session_state['rc_label'] = st.session_state['rc_label'][0:6]
            elif st.session_state['ref_20_50'] == 'Limons Grossiers-Loess':
                del st.session_state['ref_curves_selected']["ref_LimonsGrossiers"]
                del st.session_state['ref_curves_selected']["ref_Loess"]
                st.session_state['rc_label'][5] = 'Limon grossiers Loess'
                st.session_state['rc_label'] = st.session_state['rc_label'][0:6]
            elif st.session_state['ref_20_50'] == 'Limons Grossiers-Loess':
                del st.session_state['ref_curves_selected']["ref_LimonsGrossiersLoess"]
                del st.session_state['ref_curves_selected']["ref_LimonsGrossiers"]
                st.session_state['rc_label'][5] = 'Loess'
                st.session_state['rc_label'] = st.session_state['rc_label'][0:6]
            # Do nothing if all 3 at the same time selected

            # Gathering y from every reference curve into our M_ref matrix
            M_ref = np.zeros(
                (len(st.session_state['ref_curves_selected']), st.session_state['ref_curves_selected']['ref_ArgilesClassiques'].shape[1]))
            for i, ref_curve in enumerate(st.session_state['ref_curves_selected']):
                M_ref[int(
                    i), :] = st.session_state['ref_curves_selected'][ref_curve][1, :]

            # A_ref is the mimimal argument of the optimisation problem
            X = st.session_state['granulometrics'].to_numpy()
            A_ref = X @ M_ref.T @ np.linalg.inv(M_ref @ M_ref.T)

            # Performing minimalization with CVXPY to compare
            # Declaration of our minimization variable A
            A = cp.Variable((X.shape[0], M_ref.shape[0]))
            # Constraint A to be positive
            constraints = [A >= 0]
            objective = cp.Minimize(
                cp.norm(X - A @ M_ref, 'fro')**2)   # Objective function
            # problem = cp.Problem(objective)                        # optim without constraint to compare with our direct solution
            # Definition of our problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=True, eps=1e-10,
                          max_iters=10000)    # Calling solver
            A_ref_solv = A.value                                    # We get the result

            # X_ref the approximations of our observations with ref_curves
            X_ref = pd.DataFrame(
                A_ref_solv @ M_ref, columns=st.session_state['granulometrics'].columns, index=st.session_state['granulometrics'].index)

            # df for the proportions
            RC_areas = np.apply_along_axis(trapeze_areas, 1, M_ref).reshape(
                (A_ref_solv.shape[1]))  # compute areas of each EM
            Prop = A_ref_solv * RC_areas
            Prop = np.apply_along_axis(lambda x: x/np.sum(x)*100, 1, Prop)

            # naming the columns of Prop with regards of where the peak is located for each EM
            st.session_state['Prop_rc'] = pd.DataFrame(
                Prop, index=st.session_state['granulometrics'].index, columns=st.session_state['rc_label'])

            # Approximation errors l2
            err2_approx_rc = np.sum(np.linalg.norm(
                X_ref-st.session_state['granulometrics'], axis=1))
            # L1-relativ norm of each approximations
            st.session_state['Prop_rc']['L1_rel_norm (%)'] = X_ref.apply(
                lambda row: L1_relative(row.values, row.name), axis=1)
            # L1-relativ mean
            errL1_approx_rc = np.mean(
                st.session_state['Prop_rc']['L1_rel_norm (%)'])

            X_ref.index = X_ref.index.map(
                lambda x: f"r{x}")  # adding "r" before

            # in this case we replace the old reference curves approximation
            if st.session_state['rc_flag']:
                for ind in X_ref.index:
                    st.session_state['X-X_hat-X_ref'].loc[ind] = X_ref.loc[ind]

            else:  # easier case : there isn't already a reference curves approximation
                st.session_state['X-X_hat-X_ref'] = pd.concat(
                    [st.session_state['X-X_hat-X_ref'], X_ref], axis=0)
                st.session_state['rc_flag'] = True  # They are now result

            st.success("Approximation succeed")
            # Displaying approx errors
            col1, col2 = st.columns(2)
            with col1:
                st.latex(r''' \sum_{i=1}^{n} \Vert x_i-{x_{ref,i}} \Vert_2 ''')
            with col2:
                st.metric("sum of quadratic errors", value=f"{err2_approx_rc:.4}",
                          label_visibility="visible")
            col1, col2 = st.columns(2)
            with col1:
                st.latex(
                    r''' \sum_{i=1}^{n} \frac{\Vert x_i-{x_{ref,i}} \Vert_{L1}}{\Vert x_i \Vert_{L1}} ''')
            with col2:
                st.metric("mean of L1-relative errors (%)", value=f"{errL1_approx_rc:.3}%",
                          label_visibility="visible")

with tab_result:
    st.header("Display observations to compare them")
    st.markdown(
        "##### Please perform approximation before trying to plot curves of approximations")

    labels_obs = st.session_state['granulometrics'].index
    labels_approx_nmf = st.session_state['X-X_hat-X_ref'].index[st.session_state['X-X_hat-X_ref'].index.str.startswith((
        '^'))]
    labels_approx_rc = st.session_state['X-X_hat-X_ref'].index[st.session_state['X-X_hat-X_ref'].index.str.startswith((
        'r'))]

    # Selection of curves to plot
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.multiselect("labels of the observations to diplay",
                       options=labels_obs, key='selected_obs_labels')
    with col2:
        st.checkbox("Display NMF-approximations", key='flag_nmf_approx', value=False,
                    disabled=not st.session_state['nmf_flag'])
    with col3:
        st.checkbox("Display approximations with reference curves",
                    key='flag_rc_approx', value=False, disabled=not st.session_state['rc_flag'])
    with col4:
        st.checkbox("Display approximations with discrete dictionnary (NN-LASSO)",
                    key='flag_nnlasso_approx', value=False,  disabled=not st.session_state['dd_flag'])

    if st.session_state['nmf_flag']:
        st.subheader(
            "Proportions of EM (NMF-approximations) for selected observation")
        st.dataframe(
            st.session_state['Prop_nmf'].loc[st.session_state['selected_obs_labels']])

    if st.session_state['rc_flag']:
        st.subheader(
            "Proportions of reference curve (approximations) for selected observation")
        st.dataframe(
            st.session_state['Prop_rc'].loc[st.session_state['selected_obs_labels']])

    if st.session_state['dd_flag']:
        st.subheader(
            "Proportions of curve in the discrete dicitonnary (approximations) for selected observation")
        for label in st.session_state['selected_obs_labels']:
            st.table(st.session_state['Prop_nn_lasso'][label])

    if st.button('Plots curves'):
        curves_and_approx = st.session_state['X-X_hat-X_ref']
        fig = go.Figure()
        for label in st.session_state['selected_obs_labels']:
            fig.add_trace(go.Scatter(x=curves_and_approx.columns,
                          y=curves_and_approx.loc[label], mode='lines', name=label))
            if st.session_state['flag_nmf_approx']:
                fig.add_trace(go.Scatter(x=curves_and_approx.columns,
                                         y=curves_and_approx.loc[f"^{label}"], mode='lines', name=f"^{label}"))
            if st.session_state['flag_rc_approx']:
                fig.add_trace(go.Scatter(x=curves_and_approx.columns,
                                         y=curves_and_approx.loc[f"r{label}"], mode='lines', name=f"r{label}"))
            if st.session_state['flag_nnlasso_approx']:
                fig.add_trace(go.Scatter(x=curves_and_approx.columns,
                                         y=curves_and_approx.loc[f"dd-{label}"], mode='lines', name=f"dd-{label}"))

        fig.update_xaxes(type="log", tickformat=".1e", dtick=1, showgrid=True)
        fig.update_layout(
            height=800,
            width=1000,
            showlegend=True,
            xaxis_title=" grain diametere (micrometers, log-scale)"
        )
        fig.update_traces(
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>')

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
