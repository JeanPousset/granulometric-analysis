import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append("..")
from backends.numpy_functions import robust_nmf

st.set_page_config(page_title='NMF test', layout='centered')


st.title('Comparaison of differents NMF')

nb_end_members = 8 # to choose ?


# Loading data :
if 'granulometrics' not in st.session_state :
    data = pd.read_excel("data_granulometry.xls",sheet_name=0,header=0,index_col = 2)
    data = data.drop(columns=['Dept','Commune','Type'])       # Deletion of additional information
    data = data.diff(axis=1) # Cumulative difference on line item
    data[[0.03]] = 0 
    data.div(np.log10([float(col) for col in data.columns]),axis=1)
    st.session_state['granulometrics'] = data

# initialisation
if 'a_W' not in st.session_state :
    st.session_state['a_W'] = 0.0
if 'a_H' not in st.session_state :
    st.session_state['a_H'] = 0.0
if 'ratio_l1' not in st.session_state :
    st.session_state['ratio_l1'] = 0.0
if 'lambda_robust' not in st.session_state :
    st.session_state['lambda_robust'] = 1.0
if 'beta_r' not in st.session_state :
    st.session_state['beta_r'] = 1.5 
if 'selected_label' not in st.session_state :
    st.session_state['selected_label'] = []
if 'A_df' not in st.session_state :
    st.session_state['A_df'] = pd.DataFrame(np.zeros((st.session_state['granulometrics'].to_numpy().shape[0],nb_end_members)))
if 'X-X_hat' not in st.session_state :
    st.session_state['X-X_hat'] = st.session_state['granulometrics']



tab_basic, tab_robust, observations = st.tabs(['basic NMF (with penalization)', 'Robust NMF','Display observations'])

with tab_basic: 

    st.header("Basic NMF")
    st.markdown(""" Perform the following factorisation :  $$ X \\thickapprox MA + \\varepsilon $$ 
                by minimising the following expression as a function of $M$ and $A$
                """)
    st.latex(r'''
            \Vert X-MA\Vert^2_{\beta-loss}+2l_{1_\text{ratio}}\left(\alpha_M m\Vert M \Vert_1 +\alpha_A n\Vert A \Vert_1 \right)+(1-l_1{_\text{ratio}})\left(\alpha_M m\Vert M \Vert_F^2 +\alpha_A n\Vert A \Vert_F^2 \right)
             ''')
    
    st.markdown(""" You can choose the values of the parameters : $\\beta-loss$, $l_1{_\\text{ratio}}$, $\\alpha_M$ and $\\alpha_A$
                """)
    
    st.header("Parameters for basic NMF")

    col1,col2 = st.columns([1,2])

    with col1:
        loss_choice = st.selectbox("Beta_loss :",("Frobenius","kullback-leibler"))
        if loss_choice == "kullback-leibler" :
            st.session_state['loss'] = "kullback-leibler"
            st.session_state['solver'] = "mu"

        else :
            st.session_state['loss'] = "frobenius"
            st.session_state['solver'] = "cd"
    
    with col2:
        param = pd.DataFrame({'l1-l2 ratio' : [st.session_state['ratio_l1']],
                'penalization coef M' : [st.session_state['a_W']],
                'penalization coef A' : [st.session_state['a_H']],})
        
        param_choosen = st.data_editor(param,hide_index=True,column_config={
            "l1-l2 ratio": st.column_config.NumberColumn(min_value=0.0,max_value=1.0,format="%f"),
            "penalization coef M": st.column_config.NumberColumn(format="%f"),
            "penalization coef A": st.column_config.NumberColumn(format="%f")
            }
        )
        




    st.header("Algorithm")

    if st.button("Lunch basic factorization") :

        # parameters recovery
        st.session_state['ratio_l1'] = param_choosen.loc[0,'l1-l2 ratio']
        st.session_state['a_W'] = param_choosen.loc[0,'penalization coef M']
        st.session_state['a_H'] = param_choosen.loc[0,'penalization coef A']

        X = st.session_state['granulometrics'].to_numpy()
        model = NMF(n_components=nb_end_members, 
                    solver=st.session_state['solver'],
                    beta_loss=st.session_state['loss'],
                    init='random',
                    l1_ratio=st.session_state['ratio_l1'] , 
                    alpha_W=st.session_state['a_W'], 
                    alpha_H=st.session_state['a_H'],
                    random_state=0, max_iter = 15000) 
                    # Increase max_iter to get convergence
        A = model.fit_transform(X)
        M = model.components_

        # df for the approximation
        X_hat = pd.DataFrame(A @ M,columns=st.session_state['granulometrics'].columns,index=st.session_state['granulometrics'].index) # Estimations of our observations with only 8 EM 

        # Approximation error calculation with sum of euclidean norm of Xi-Xi_hat
        err_approx = np.sum(np.linalg.norm(X_hat-st.session_state['granulometrics'], axis=1))

        # adding approximation to our result df
        X_hat.index = X_hat.index.map(lambda x: f"^{x}") # adding "^-" before
        st.session_state['X-X_hat']=pd.concat([st.session_state['granulometrics'],X_hat],axis=0)

        st.success("NMF succeed")

        # Displaying approx error
        col1,col2 = st.columns(2)
        with col2 :
            st.latex(r''' \sum_{i=1}^{853} \Vert x_i-\hat{x_i} \Vert_2 ''')
        with col1 :
            st.metric("Approximation error",err_approx,label_visibility="visible")

        st.header("Visualisaiton")

        with st.expander("End-Members"):
            fig_em, axs_em = plt.subplots(nrows=nb_end_members//2, ncols=2, figsize=(10, 2*nb_end_members))

            for i in range(nb_end_members//2) :
                for j in range(2) :
                    axs_em[i,j].semilogx(st.session_state['granulometrics'].columns,M[2*i+j,:])
                    axs_em[i,j].set_title(f'End-member {2*i+j+1}') 

            plt.tight_layout()
            st.pyplot(fig_em)
        
        with st.expander("Proportions of EM in our observations"):
            st.session_state['A_df'] = pd.DataFrame(A,index=st.session_state['granulometrics'].index,columns=[f'EM{i}' for i in range(1, nb_end_members+1)])
            st.session_state['A_df']['label']=st.session_state['granulometrics'].index
            fig = make_subplots(rows=nb_end_members//2, cols=1,vertical_spacing=0.05)
            for i in range(nb_end_members//2) :

                first_em = 2*i+1
                second_em = 2*(i+1)

                fig.add_trace(
                    go.Scatter(
                        x=st.session_state['A_df'][f'EM{first_em}'],
                        y=st.session_state['A_df'][f'EM{second_em}'],
                        mode='markers',
                        marker=dict(size=10, color=st.session_state['A_df']['label'].astype('category').cat.codes, colorscale='rainbow'),
                        text=st.session_state['A_df']['label'],  
                    ),
                    row=i+1, col=1
                )
                fig.update_xaxes(title_text=f'End-member {first_em}', showgrid=False, gridcolor='LightGray', row=i+1, col=1)
                fig.update_yaxes(title_text=f'End-member {second_em}', showgrid=False, gridcolor='LightGray', row=i+1, col=1)

            
            fig.update_layout(
                height=700 * nb_end_members//2,  # Ajuster la hauteur de la figure en fonction du nombre de plots
                title_text='Proprotions of End-members',
                showlegend=False  # Masquer la légende pour simplifier l'affichage
            )
                
            st.plotly_chart(fig)       


with tab_robust:

    col1_r, col2_r = st.columns(2)
    with col1_r:
        st.session_state['lambda_robust'] = st.number_input("penalization regularization weigth for L-2,1 norm of the residuals R", step = 0.000001, value=st.session_state['lambda_robust'])
    with col2_r: 
        st.session_state['beta_r'] = st.number_input("beta param for the beta-divergence, 0 : Itakura-Saito, 1 : Kullback-Leibler, 2 : Euclidean", step = 0.1, value=st.session_state['beta_r'])
   
    st.header("Algorithm")

    if st.button("Lunch robust factorization") :
        X = st.session_state['granulometrics'].to_numpy() 
        A, M, R, obj = robust_nmf(X,
                                        rank=nb_end_members,
                                        beta=st.session_state['beta_r'],
                                        init='random',
                                        reg_val=st.session_state['lambda_robust'],
                                        sum_to_one=0,
                                        tol=1e-7,
                                        max_iter=200)
        
        X_hat = pd.DataFrame(A @ M,columns=st.session_state['granulometrics'].columns,index=st.session_state['granulometrics'].index) # Estimations of our observations with only 8 EM 
        X_hat.index = X_hat.index.map(lambda x: f"^{x}") # adding "^-" before
        st.session_state['X-X_hat']=pd.concat([st.session_state['granulometrics'],X_hat],axis=0)

        st.success("Robust NMF succeed")

        st.header("Visualisaiton")

        with st.expander("End-Members"):
            fig_em, axs_em = plt.subplots(nrows=nb_end_members//2, ncols=2, figsize=(10, 2*nb_end_members))

            for i in range(nb_end_members//2) :
                for j in range(2) :
                    axs_em[i,j].semilogx(st.session_state['granulometrics'].columns,M[2*i+j,:])
                    axs_em[i,j].set_title(f'End-member {2*i+j+1}') 

            plt.tight_layout()
            st.pyplot(fig_em)
        
        with st.expander("Proportions of EM in our observations"):
            st.session_state['A_df'] = pd.DataFrame(A,index=st.session_state['granulometrics'].index,columns=[f'EM{i}' for i in range(1, nb_end_members+1)])
            st.session_state['A_df']['label']=st.session_state['granulometrics'].index
            fig = make_subplots(rows=nb_end_members//2, cols=1,vertical_spacing=0.05)
            for i in range(nb_end_members//2) :

                first_em = 2*i+1
                second_em = 2*(i+1)

                fig.add_trace(
                    go.Scatter(
                        x=st.session_state['A_df'][f'EM{first_em}'],
                        y=st.session_state['A_df'][f'EM{second_em}'],
                        mode='markers',
                        marker=dict(size=10, color=st.session_state['A_df']['label'].astype('category').cat.codes, colorscale='rainbow'),
                        text=st.session_state['A_df']['label'],  
                    ),
                    row=i+1, col=1
                )
                fig.update_xaxes(title_text=f'End-member {first_em}', showgrid=False, gridcolor='LightGray', row=i+1, col=1)
                fig.update_yaxes(title_text=f'End-member {second_em}', showgrid=False, gridcolor='LightGray', row=i+1, col=1)

            
            fig.update_layout(
                height=700 * nb_end_members//2,  # Ajuster la hauteur de la figure en fonction du nombre de plots
                title_text='Proprotions of End-members',
                showlegend=False  # Masquer la légende pour simplifier l'affichage
            )
                
            st.plotly_chart(fig)

with observations :
    st.header("Display observations to compare them")

    labels_obs = st.session_state['granulometrics'].index
    labels_approx = st.session_state['X-X_hat'].index[st.session_state['X-X_hat'].index.str.startswith("^")]
    
    # Selection of curves to plot
    col1,col2 = st.columns(2)
    with col1:
        st.session_state['selected_obs_labels'] = st.multiselect("labels of the observations to diplay", options=labels_obs)
    with col2:
        st.session_state['selected_approx_labels'] = st.multiselect("labels of the approximations to diplay", options=labels_approx)


    st.subheader("Proportions of EM for selected observations")
    st.table(st.session_state['A_df'].loc[st.session_state['selected_obs_labels']])

    if st.button('Plots curves'):
        fig, ax = plt.subplots()

        for label in st.session_state['selected_obs_labels']+st.session_state['selected_approx_labels']:
            ax.semilogx(st.session_state['X-X_hat'].columns,st.session_state['X-X_hat'].loc[label], label=label)

        ax.set_xlabel('micrometers')
        ax.set_title('granulometrics curves of selected observations')
        ax.legend()
        st.pyplot(fig)
