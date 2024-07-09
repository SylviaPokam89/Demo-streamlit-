import streamlit as st

# Image illustrative du projet
from PIL import Image
img = Image.open("image_illustrative.jpg")
st.image(img)
st.title("Projet E_Commerce")


# Core Packages
import pandas as pd 
import seaborn as sns 
import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso


# titres sidebar
st.sidebar.title("Plan de la présentation")
menu = st.sidebar.radio(label = "", options = ["Introduction", "Jeux de données", "Preprocessing", "Analyses graphiques","Modélisations et résultats", "Interprétabilité", "Prédictions", "Conclusions et axes d’améliorations"])

# sidebar présentation de la team
auteurs = st.sidebar.info("## Auteurs")
with st.sidebar.info("## Auteurs"):
    st.write("### Sylvia POKAM \n"
             "### Elisheva BRAMI \n"
             "### Emile BEAU")




# Introduction:
if menu == "Introduction":
    st.header('Contexte')
    st.markdown('''Les consommateurs se tournent de plus en plus vers **les sites de vente en ligne**.
    Cet engouement exige des entreprises un maintient d'avantages concurrentiels pour maintenir 
    (si ce n’est augmenter) leur rentabilité. En plus du marketing, une entreprise qui gère efficacement ses coûts, génère moins de perte. 
    De ce fait, il est crucial pour une entreprise de savoir prédire avec peu d’incertitudes ses ventes 
    pour un approvisionnement optimal. Une prévision erronée conduirait à un manque à gagner, soit dû à un excédent de 
    produits en stocks, soit à une rupture de stocks.
    ''')

    st.markdown('''Ce projet a pour but d'améliorer les prévisions de ventes d'un site **E_commerce** grâce aux modèles de machine learning. 
    ''')

    st.write("### Enjeux et attentes projet:")
    st.markdown("-	Créer une base de données à partir des données brutes de ventes en ligne fournies par la start-up;")
    st.markdown("-	Faire un nettoyage des données;")
    st.markdown("-	Implémenter différents modèles de machine learning pour sélectionner ceux capables de répondre à notre problématique;")
    st.markdown("-	Optimiser les modèles sélectionnés pour obtenir de meilleurs rendements;")
    st.markdown("-	Faire une étude d’interprétabilité pour mettre en évidence les variables les plus pertinentes;")
    st.markdown("-	Prédire le niveau de ventes sur plusieurs mois à venir;")


# Jeux de données:
private_data = "private_data.csv"
final_stamp = "final_stamp.csv"
df1 = pd.read_csv(private_data)
df2 = pd.read_csv(final_stamp)

if menu == "Jeux de données":
    st.header("Jeux de données")
    st.write("Nous avons eu pour ce projet deux jeux de données:")
    st.markdown(''' - Un premier dataset appelé « private_data » qui contient les données de ventes
     telles que les quantités vendues par produits, les prix de ventes avant et après promotions, 
     le nom des sites de ventes, les dates...
    ''')
    st.markdown(''' - Et un second dataset appelé « final_stamp » qui lui contient les données sur 
    l’évaluation et la disponibilité des produits. On y retrouve dans ce dataset par exemple, 
    le nombre d’étoiles obtenu par produit, le nombre de commentaires, le stock par produits… 
    ''')
    datas = st.radio(label = " ", options = ["private_data", "final_stamp"])

    st.write("##### Déroulé dataframes et variables")
    
    if datas == "private_data":
        if st.checkbox("Dataframe private_data") :
             line_to_plot = st.slider("nombre de lignes", min_value=5, max_value=df1.shape[0])
             st.dataframe(df1.head(line_to_plot))

        if st.checkbox("Variables") : 
             variables_1 = "variables_private_data.xlsx"
             df3 = pd.read_excel(variables_1)
             st.dataframe(df3)

    if datas == "final_stamp":
        if st.checkbox("Dataframe final_stamp") :
             line_to_plot = st.slider("nombre de lignes", min_value=5, max_value=df2.shape[0])
             st.dataframe(df2.head(line_to_plot))

        if st.checkbox("Variables") : 
             variables_2 = "variables_final_stamp.xlsx"
             df4 = pd.read_excel(variables_2)
             st.dataframe(df4)
    

# Preprocessing 

if menu == "Preprocessing":
    st.header("Preprocessing")
    st.markdown('''Dans cette partie, nous allons explorer plus en détail 
    les variables des datasets pour mieux les comprendre. 
    Nous allons entre autre faire  une analyse statistique des variables, 
    une évaluation des valeurs manquantes et des doublons, 
    une étude du type de variables…
    ''')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    st.subheader("types de variables et valeurs manquantes")
    datas_1 = st.radio(label = " ", options = ["private_data", "final_stamp"])

    if datas_1 == "private_data":
        st.write("##### tableau recapitulatif private_data")
        miss_values = "valeurs manquantes_private.xlsx"
        df5 = pd.read_excel(miss_values)
        st.dataframe(df5)
        st.info('''Le dataset private_data ne contient aucune valeur manquante ni de doublons. 
        Les types de format des variables  sont également corrects, 
        excepté la variable date qui est au format « object » et non « date »
        ''')
    
    if datas_1 == "final_stamp":
        st.write("##### tableau recapitulatif final_stamp")
        miss_values_1 = "valeurs manquantes_final.xlsx"
        df6 = pd.read_excel(miss_values_1)
        st.dataframe(df6)
        st.markdown('''Le dataset final_stamp contient des variables avec des valeurs manquantes.''')
        st.warning('''Ces valeurs manquantes représentent une forte proportion (plus de 70%) des 
        variables rating_i (pour i allant de 1 à 5) et review_count.''')
    
    st.write(' ')
    st.write(' ')
    st.write(' ')



    st.write("### Analyse descriptive")
    st.write("**Tableau statistique du dataset private_data**")
    stat_private = df1.describe()
    st.dataframe(stat_private)
    st.info('''Les valeurs minimales des variables  average_retail_price  et average_selling_price sont égales à zéro. 
    Or ces variables ne sauraient être nulles car elles représentent les prix de ventes.''')
    st.write(' ')
    st.write(' ')

    st.write("**Tableau statistique du dataset final_stamp**")
    stat_final_stamp = df2.describe()
    st.dataframe(stat_final_stamp)
    st.info('''Le dataset final_stamp présente tout comme le dataset private_data, les valeurs minimales des variables 
    retail_price et selling_price == 0''')

    st.markdown('''-   Seul le site de vente 'SHP' est concerné lorsque ces deux variables sont nulles simultanément.''')
    st.markdown('''-   Egalement quand ces deux variables sont nulles, les valeurs des colonnes rating_i et rating_count 
    sont toutes manquantes.''')
    st.markdown('''On peut donc penser que ces valeurs manquantes sont dues à une mauvaise collecte des données lors de leur portabilité. ''')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    
    st.write("### Nettoyage")
    st.markdown("Le nettoyage des datasets s'est fait comme ci-dessous:")
    st.markdown("-	Suppression de la colonne « Unnamed:0 » dans les deux tableaux car identique à la colonne index.")
    st.markdown("-	Suppression des lignes avec average_retail_price == 0  et average_selling_price == 0 (dataset private_data).")
    st.markdown("-	Suppression des lignes avec les variables retail_price == 0 et selling_price == 0 (dataset final_stamp).")
    st.markdown("-	Suppression de la variable historical_sales du dataset final_stamp, car pas d’informations concernant le site LAZ.")
    st.markdown("-	Suppression des doublons (221 au total) du tableau final_stamp.")
    st.markdown("-	Suppression des lignes avec rating_count == 0 , car tous les rating_i concernés sont également = 0.")
    st.write(' ')
    st.write(' ')
    st.write(' ')


    st.write("### Ajout des variables")
    st.write("Pour alimenter notre analyse, nous avons créé les variables suivantes:")
    st.write("**Périodicité:** Décomposition de la variable datetime en variables 'year', 'month', 'weekday', et 'weekend'")
    st.write("**Promotion:** Une variable « promotion_en_% » a été ajoutée au tableau private_data. Il s'agit de la différence(en %) entre les variables average_retail_price = prix de vente avant promotion et average_selling_price = prix de vente après promotion.")
    st.write(' ')
    st.write(' ')
    st.write(' ')


    st.write("### Jointure des deux dataframes")
    
    st.write('**Merge des deux tableaux:** merge fait sur six variables')
    # rename des variables dans le tableau final_stamp pour le merge:
    df2 = df2.rename(columns={'selling_price': 'average_selling_price', 'retail_price': 'average_retail_price'})

    with st.echo():
        df = pd.merge(df1, df2, left_on=['marketplace', 'seller_id', 'product_id', 'variation_id', 'average_retail_price', 'average_selling_price'], 
              right_on=['marketplace', 'seller_id', 'product_id', 'variation_id', 'average_retail_price', 'average_selling_price'])

    st.write("**Aggregation des varibales suivi d'un groupby** uniquement avec les variables stratégiques comme ci-dessous:")
    with st.echo():
        aggregation = {"quantity": "sum",
               "order_count": "sum",
               "average_retail_price": 'first',
               "average_selling_price": "first",
               'marketplace': 'first', 
               "seller_id": "first",
               'product_id': 'first',
               'variation_id': 'first',
               "month": "nunique",
               "rating_avg": "mean",
               "rating_count": "max",
               "review_count": "max", 
               "stock": 'mean'}
    st.write(' ')
    st.write(' ')
    st.write(' ')

    st.write("**Après Dichotomisation des marketplaces, et suppression d'autres variables redondantes, nous avons obtenu le tableau final ci-dessous:**")
    data_sans_ordercount = 'data_sans_ordercount.csv'
    df7 = pd.read_csv(data_sans_ordercount)
    line_to_plot = st.slider("nombre de lignes", min_value=5, max_value=df7.shape[0])
    st.dataframe(df7.head(line_to_plot))







# Prédictions
if menu == "Prédictions":
    st.header("Prédicitons: Méthode de séries temporelles")
    st.write('''Le but final de ce projet étant la prévision des ventes sur un site de vente en ligne, 
    nous allons à présent essayer de prédire les quantités de vente sur les mois futurs. Une série temporelle 
    a été créée avec les données en notre possession, i.e. sur les **dix premiers mois de l’année 2020**
    ''')

    # chargement du fichier pour prévisions:
    data_previsions = 'data_pour_previsions.csv'
    df = pd.read_csv(data_previsions)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(df['date'])
    df = df.drop('date', axis = 1)

    if st.checkbox("data_previsions") :
             line_to_plot = st.slider("nombre de lignes", min_value=10, max_value=df.shape[0])
             st.dataframe(df.head(line_to_plot))
    
    # Dataviz
    plt.figure(figsize = (8, 4))
    plt.plot(df)
    plt.xlabel("date")
    plt.ylabel("quantité_de_ventes")
    plt.title("quantités de ventes sur l'année 2020")
    st.pyplot()
    
    # suppression warnings qui s'affichent sur la page de présentation web
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.write('''Pour analyser une série temporelle, il est important de la décomposer selon les trois 
    termes suivants : la tendance, la saisonnalité et le résidu, comme présenté sur le graphe ci-dessous.
    ''')

    # pour une bonne cohérence au niveau de la prévision avec une périodicité correctes: 
    # décision de supprimer les 13 premières lignes de df car il manque des jours
    df = df.iloc[13:]
    
    import statsmodels.api as sm
    plt.figure(figsize = (3, 2))
    decomposition = sm.tsa.seasonal_decompose(df)
    decomposition.plot()
    st.pyplot()


    st.write('''Nous observons une décomposition satisfaisante (faible variation de résidu). On observe des pics 
    d’augmentation des ventes avec le temps. Cette augmentation n’est cependant pas très significative et donc 
    difficile à quantifier. N’ayant pas des données sur plusieurs années, il n’est pas possible pour nous ici 
    d’avancer des hypothèses sur les tendances de variations saisonnières ou annuelles des ventes. Quant à la saisonnalité, 
    nous l’avons ici définie à 12 car nous travaillons en année (même si l’année sur laquelle est étendue la série de données 
    n’est pas complète).''')

    st.write("#### Test statistique:")
    st.write('Pour vérifier la stationnarité de notre série, nous avons utilisé le **test de Dickey-Fuller**')
    from statsmodels.tsa.stattools import adfuller
    _, p_value, _, _, _, _ = adfuller(df)
    st.write("p_value =", p_value)
    st.write('La p_value est bien inférieure à 5%, ainsi on peut considérer notre série comme stationnaire.')

    st.write("## Prédictions avec SARIMAX")

    # séparation des données en train et test
    train = df.iloc[:240]
    test = df.iloc[240:]

    # évaluation des prévisions avec SARIMAX
    st.write('''Sur les deux modèles de prédictions testés (ARIMA et SARIMAX), nous avons décidé de présenter 
    dans ce rapport uniquement les résultats obtenus avec SARIMAX, car c’est celui qui nous a donné des prédictions 
    on va dire « à peu près acceptables ». Les hyper paramètres utilisés sont les suivants : 
    p = 0, d = 0, q = 2 et s = 12.''')

    model = sm.tsa.SARIMAX(train, order=(0,0,2), seasonal_order=(0,0,2,12))
    result = model.fit()
     
    # Représentation graphique des valeurs prédites et celles réelles
    st.write("#### Représentation graphique des valeurs prédites et celles réelles:")
    start = len(train)
    end = len(df) - 1
    pred = result.predict(start = start, end = end, typ = 'levels')
    plt.figure(figsize = (8, 4))
    plt.plot(test, label = 'valeurs_réelles')
    plt.plot(pred, label = 'valeurs Prédites')
    plt.title('Predictions and real_values')
    plt.legend()
    st.pyplot()

    # erreur moyenne
    st.write("#### Erreur moyenne:")
    from sklearn.metrics import mean_squared_error
    RMSE = np.sqrt(mean_squared_error(test, pred))

    err = np.mean(pred) - np.mean(test)
    error = err/np.mean(test) * 100
    st.write("l'erreur quadratique moyenne =", RMSE)
    st.write("l'erreur moyenne relative en % vaut: ", error)
    st.write('''Au regard de la forte valeur de la RMSE et d'une erreur moyenne relative égale à -39 %, 
    on peut conclure à une sous-évaluation de la prédiction des ventes.''')


    # Prévisions
    st.write("#### prévisions sur les mois futurs")
    prev = result.predict(start = len(train) - 50, end = len(df) + 32)
    plt.figure(figsize = (8, 4))
    plt.plot(df, label = 'quantités_ventes')
    plt.plot(prev, label = 'prévisions', color = 'red')
    plt.legend()
    st.pyplot()

    
    st.write('''Le graphe des prévisions sur les mois futurs sont en accord avec les valeurs d'erreurs obtenues ci-dessus. Nous observons
    une forte sous évaluation sur la prévision des ventes. Ces prévisions sont d'autant plus mitigées, qu'à partir du 10 octobre, 
    on a que des valeurs nulles jusque la fin de la période prédite.''')

    st.write('''Ce résultat s'explique probablement par le fait que nous avons pris une saisonnalité = 12 sachant que nos données 
    ne s’étendent pas sur une année complète. En effet, notre modèle n'ayant pas appris sur les mois complets de octobre à décembre,
    il n'a aucune idée de la tendances de ventes sur ces mois, ce qui rend difficile une prédicition fiable.''')

