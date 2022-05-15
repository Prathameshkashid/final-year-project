import streamlit as st
import warnings                                  
warnings.filterwarnings('ignore')  
#from __future__ import division, print_function, unicode_literals
from numpy import *
import numpy as np
import matplotlib.pyplot as mplot
import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import pylab
import warnings
warnings.filterwarnings('ignore')
from itertools import cycle
from sklearn.cluster import KMeans as km
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Lasso as LS
from sklearn.linear_model import Ridge as RG
from sklearn.tree import DecisionTreeRegressor as scart
from sklearn.manifold import TSNE
#---------------------------------------------------------------------------------------------------------------------------------------
add_selectbox = st.sidebar.markdown(':sunglasses: Name: Prathamesh Laxman Kashid :sunglasses:')
add_selectbox = st.sidebar.markdown('Seat Number:____________________')
add_selectbox = st.sidebar.markdown('Project Name: DS in Agriculture Sector :sunglasses:')
st.sidebar.image('pk.JFIF', width=300)
add_selectbox = st.header('Final Year Project (2021-2022) :sunglasses:')
#============================================================================================================================================================    
click= st.checkbox('Data Reading')
if click==True:
    #add_selectbox = st.subheader('Data Reading')
    A=['Crop Production','Crop Price','Area Under Cultivation','Cultivation Cost','Mean Temperature','Rainfall','Major Crop','Indian Export']
    col=st.selectbox("Select option for fetching the data:",A)
    if col=='Crop Production':
        st.error('Fetching Crop Production Data...')
        crop_prod=pd.read_csv('apy.csv',delimiter=',')
        crop_prod
        st.success('Data Fetch Successfully!!!')
#-----------------------------------------------------------------------------------------------------------------------------------------
    if col=='Crop Price':
        st.error('Fetching Crop Price Data...')
        crop_price=pd.read_csv("Crops_price.csv",delimiter=',')
        crop_price
        st.success('Data Fetch Successfully!!!')
#-----------------------------------------------------------------------------------------------------------------------------------------
    if col=='Area Under Cultivation':
        st.error('Fecthing Area Under Cultivations Data...')
        area_cult=pd.read_csv("area_cult.csv",delimiter=',')
        area_cult
        st.success('Data Fetch Successfully!!!')
#------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Cultivation Cost':
        st.error('Fetching Cultivation Cost Data...')
        culti_cost=pd.read_csv("culti_cost.csv",delimiter=',')
        culti_cost
        st.success('Data Fetch Successfully!!!')
#-------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Mean Temperature':
        st.error('Fecthing Mean Temperature Data...')
        temperature = pd.read_csv('Mean_Temperatures.csv',delimiter=',')
        temperature
        st.success('Data Fetch Successfully!!!')
#--------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Rainfall':
        st.error('Fetching Rainfall Data...')
        rainfall =  pd.read_csv('rainfall_cleaned.csv',delimiter=',')
        rainfall
        st.success('Data Fetch Successfully!!!')
#---------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Major Crop':
        st.error('Fetching Major Crop Data...')
        growth = pd.read_csv('Avg annual Growth Rate_Major Crops.csv',delimiter = ',')
        growth
        st.success('Data Fetch Successfully!!!')
#---------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Indian Export':
        st.error('Fetching Indian Export Data...')
        exports = pd.read_csv('IndiaExport.csv',delimiter=',')
        exports
        st.success('Data Fetch Successfully!!!')
#======================================================================================================================================================
click= st.checkbox('Data Prepocessing')
if click==True:
    B=['Crop_Production','Crop Price','Area Under Cultivation','Cultivation Cost','Mean Temperature','Rainfall','Major Crop','Indian Export']
    col=st.selectbox("Select Data from drop down list:",B)
    if col=='Crop_Production':
        st.warning('Preprocessing Data...')
        crop_prod=pd.read_csv('apy.csv',delimiter=',')
        st.write(crop_prod.isnull().sum())
        if crop_prod.isnull().values.any():
            st.error('Null value found... removing it...')
            crop_prod=crop_prod.fillna(0)
            crop_prod.sort_values(by=crop_prod.columns[0])
            states = sorted(set(crop_prod.iloc[:,0].values))
            st.write(crop_prod.isnull().sum())
            st.success('Data Preprocessed Successfully!!!')
#-----------------------------------------------------------------------------------------------------------------------------------------
    if col=='Crop Price':
        st.warning('Preprocessing Data...')
        crop_price=pd.read_csv("Crops_price.csv",delimiter=',')
        for i in range(1,crop_price.shape[1]):
            crop_price.iloc[:,i]=pd.to_numeric(crop_price.iloc[:,i],errors='coerce')
            crop_price.iloc[:,i]=crop_price.iloc[:,i].fillna(0)
        crop_price = crop_price.rename(columns = {'Commodities(rs/quin)':'Commodities'})
        crop_price
        st.success('Data Preprocessed Successfully!!!')
#-----------------------------------------------------------------------------------------------------------------------------------------
    if col=='Area Under Cultivation':
        st.warning('Preprocessing Data...')
        area_cult=pd.read_csv("area_cult.csv",delimiter=',')
        area_cult
        st.success('Data Preprocessed Successfully!!!')
#------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Cultivation Cost':
        st.warning('Preprocessing Data...')
        culti_cost=pd.read_csv("culti_cost.csv",delimiter=',')
        culti_cost
        st.success('Data Preprocessed Successfully!!!')
#-------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Mean Temperature':
        st.warning('Preprocessing Data...')
        temperature = pd.read_csv('Mean_Temperatures.csv',delimiter=',')
        temperature
        st.success('Data Preprocessed Successfully!!!')
#--------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Rainfall':
        st.warning('Preprocessing Data...')
        rainfall =  pd.read_csv('rainfall_cleaned.csv',delimiter=',')
        rainfall
        st.success('Data Preprocessed Successfully!!!')
#---------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Major Crop':
        st.warning('Preprocessing Data...')
        growth = pd.read_csv('Avg annual Growth Rate_Major Crops.csv',delimiter = ',')
        growth = pd.concat([growth.iloc[:,0],growth.iloc[:,5:]],axis=1,sort=False)
        growth = growth.dropna()
        growth
        st.success('Data Preprocessed Successfully!!!')
#---------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Indian Export':
        st.warning('Preprocessing Data...')
        exports = pd.read_csv('IndiaExport.csv',delimiter=',')
        to_drop = []
        for head in exports.columns[1:]:
            if 'Unn' in head:
                to_drop.append(head)
        exports.drop(columns = to_drop,inplace=True)
        exports = exports.drop(0)
        exports.rename(columns = {'Unnamed: 0':'Product'},inplace = True)
        for_plot = exports.copy()
        exports = exports.melt(id_vars='Product')
        exports.variable = exports.variable.astype(int)
        exports.value = exports.value.astype(float)
        exports
        st.success('Data Preprocessed Successfully!!!')
#======================================================================================================================================================
click= st.checkbox('Data Visualization')
if click==True:
    C=['Correlation','Crop_Price','Area Under Cultivation','Cultivation Cost','Mean Temperature','Rainfall','Major Crop']
    col=st.selectbox("Select Data from drop down list:",C)
    if col=='Correlation':
        st.error('Visualizing Data...')
        st.image('10.PNG', width=555)
        st.success('Data Visualized Successfully!!!')
#-----------------------------------------------------------------------------------------------------------------------------------------
    if col=='Crop_Price':
        st.error('Visualizing Data...')
        st.image('2.PNG', width=777)
        st.success('Data Visualized Successfully!!!')
#-----------------------------------------------------------------------------------------------------------------------------------------
    if col=='Area Under Cultivation':
        st.error('Visualizing Data...')
        st.image('3.PNG', width=777)
        st.success('Data Visualized Successfully!!!')
#------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Cultivation Cost':
        st.error('Visualizing Data...')
        st.image('4.PNG', width=888)
        st.image('5.PNG', width=888)
        st.success('Data Visualized Successfully!!!')
#-------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Mean Temperature':
        st.error('Visualizing Data...')
        st.image('6.PNG', width=777)
        st.success('Data Visualized Successfully!!!')
#--------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Rainfall':
        st.error('Visualizing Data...')
        st.image('7.PNG', width=777)
        st.success('Data Visualized Successfully!!!')
#---------------------------------------------------------------------------------------------------------------------------------------------
    if col=='Major Crop':
        st.error('Visualizing Data...')
        st.image('8.PNG', width=777)
        st.success('Data Visualized Successfully!!!')
#===============================================================================================================================================
click= st.checkbox('Observation')
if click==True:
    C=['crops that have reduction in production','crops that have reduction in production but their price is increasing','crops which has lower increase in production but are increasing in price',
       'crops that have lower rate of increase in cost per hectare than price']
    col=st.selectbox("Select Data from drop down list:",C)
    if col=='crops that have reduction in production':
        st.image('21.PNG', width=222)
        st.markdown(' The above crops shows the crops that have negative overall slope or in other words have seen decrease in production over the years. The threshold we used is -1000.')
#-----------------------------------------------------------------------------------------------------------------------------------------
    if col=='crops that have reduction in production but their price is increasing':
        st.image('22.PNG', width=333)
        st.markdown('In this we tried to find the crops that have reduction in production of the crop but there is still increase in price of the crop. This shows that the production has been decreasing but the demand for the same crops is not as can be observed by the positive value of slope.')
#-----------------------------------------------------------------------------------------------------------------------------------------
    if col=='crops which has lower increase in production but are increasing in price':
        st.image('23.PNG', width=222)
        st.markdown('The above crops are those that have lower increase in production but has increase higher increase in price. This shows that the increase in production of that crop is not as much as 27 demand. These crops will be more profitable to produce. Threshold used for production is -10000 and for price is 100.')
#------------------------------------------------------------------------------------------------------------------------------------------
    if col=='crops that have lower rate of increase in cost per hectare than price':
        st.image('24.PNG', width=333)
        st.markdown('I tried to find out the crops that has lower rate of increase in cost per hectare but has increase in price of that crop more. This shows that these crops can give more returns.')

#====================================================================================================================================
click= st.checkbox('Suicides Analysis')
if click==True:
    st.error('Fetching Sucidies Data...')
    suicides = pd.read_csv('suicides_10-14.csv',delimiter = ',')
    suicides
    
    st.warning('Preprocessing Data...')
    suicides = pd.read_csv('suicides_10-14.csv',delimiter = ',')
    suicides.drop(columns={'Sl. No.','2014 - Labourers'},inplace=True)
    suicides.iloc[:,:].fillna(0,inplace=True)
    suicides
    
    st.success('Visualizing Data...')
    st.image('9.PNG', width=777)

    from PIL import Image, ImageOps
    image1 = Image.open("s2.JPG")
    image2 = Image.open("s5.PNG")
    fig = plt.figure()
    #First Image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    #Second Image
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    st.pyplot(fig)
#=========================================================================================================================
click= st.checkbox('Weather Forecasting')
if click==True:
    import requests
    import webbrowser
    city = st.text_input('Enter City name: ')
    if city:
        url = 'https://wttr.in/{}'.format(city)
        a=st.button('View Weather Forecast')
        if a:
            webbrowser.open(url)
        #res = requests.get(url)
        #st.write(res)
        a=st.button('Weather forecast over India')
        if a:
        #url = 'https://mausam.imd.gov.in/'
            url='https://mausam.imd.gov.in/imd_latest/contents/subdivisionwise-warning.php'
            webbrowser.open_new_tab(url)
#=======================================================================================================================================================
click= st.checkbox('Crop Recommendation')
if click==True:
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.model_selection import train_test_split
    data=pd.read_csv('cropdata.csv')
    from sklearn.preprocessing import LabelEncoder
    encod = LabelEncoder()
    data['Encoded_label'] = encod.fit_transform(data.label)
    a = pd.DataFrame(pd.unique(data.label));
    a.rename(columns={0:'label'},inplace=True)
    b = pd.DataFrame(pd.unique(data.Encoded_label));
    b.rename(columns={0:'encoded'},inplace=True)
    classes = pd.concat([a,b],axis=1).sort_values('encoded').set_index('label')
    data = data.drop_duplicates()
    import pickle
    a=st.text_input("Enter N:")
    b=st.text_input("Enter P:")
    c=st.text_input("Enter K:")
    d=st.text_input("Enter temperature:")
    e=st.text_input("Enter humidity:")
    f=st.text_input("Enter ph:")
    g=st.text_input("Enter rainfall:")
    if g:
        new_data = pd.DataFrame([{'N':a,'P':b,'K':c,'temperature' : d, 'humidity' : e, 'ph' : f, 'rainfall' :g}])
        pickle_in = open('cls.pkl','rb')
        model = pickle.load(pickle_in)
        pre = model.predict_proba(new_data)
        pre = pd.DataFrame(data = np.round(pre.T*100,2), index=classes.index,columns=['predicted_values'])
        high = pre.predicted_values.nlargest(5)
        fig=plt.figure(figsize=(15,10))
        plt.rcParams['font.size']=15
        plt.title('Crops Recommendations :',fontdict={'fontsize': 25, 'fontweight': 'medium'})
        plt.pie(x=high,labels=high.index,autopct='%1.1f%%',explode=(0.1, 0, 0, 0, 0),shadow=True,startangle=90,colors=['green','red','cyan','brown','orange'])
        plt.show()
        st.pyplot(fig)
#===============================================================================================================
click= st.checkbox('Plant Diseases Recognition')
if click==True:
    class_na = ['Apple scab', 'Apple Black rot', 'Apple Cedar rust', 'Apple healthy', 'Blueberry healthy',
            'Cherry healthy','Cherry Powdery mildew',  
            'Corn Cercospora Gray leaf spot', 'Corn Common rust', 
            'Corn healthy','Corn Northern Leaf Blight',  'Grape Black rot', 
            'Grape Esca Black Measles','Grape healthy',  'Grape Leaf blight Isariopsis', 
            'Orange Haunglongbing Citrus greening', 'Peach Bacterial spot', 'Peach healthy', 
            'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato healthy',
            'Potato Late blight', 'Raspberry healthy', 'Soybean healthy', 'Squash Powdery mildew', 
            'Strawberry healthy','Strawberry Leaf scorch', 'Tomato Bacterial spot', 'Tomato Early blight', 
            'Tomato healthy', 'Tomato Late blight', 'Tomato Leaf Mold', 
            'Tomato Septoria leaf spot', 'Tomato Two spotted spider mite', 'Tomato Target Spot',
            'Tomato mosaic virus', 'Tomato Yellow Leaf Curl Virus']
    class_fe = ['Katyayani Prozol Propiconazole 25% EC Systematic Fungicide',
            'Magic FungiX For Fungal disease',
            'Katyayani All in 1 Organic Fungicide', 
            'Tapti Booster Organic Fertilizer',
            'GreenStix Fertilizer',
            'ROM Mildew Clean',
            'Plantic Organic BloomDrop Liquid Plant Food',  
            'ANTRACOL FUNGICIDE',
            '3 STAR M45 Mancozeb 75% WP Contact Fungicide', 
            'Biomass Lab Sampoorn Fasal Ahaar',
            'QUIT (Carbendazim 12% + Mancozeb 63% WP) Protective And Curative Fungicide',
            'Southern Ag Captan 50% WP Fungicide', 
            'ALIETTE FUNGICIDE',
            'Sansar Green Grapes Fertilizer', 
            'Tebulur Tebuconazole 10% + Sulphur 65% WG , Advance Broad Spectrum Premix Fungicides', 
            'Green Dews CITRUS PLANT FOOD Fertilizer', 
            'SCORE FUNGICIDE', 
            'Jeevamrut Plant Growth Tonic', 
            'Systemic Fungicide (Domark) Tetraconazole 3.8% w/w (4% w/v) EW',
            'Casa De Amor Organic Potash Fertilizer', 
            'Parin Herbal Fungicides ', 
            'Saosis Fertilizer for potato Fertilizer',
            'Syngenta Ridomil gold Fungicide', 
            'Karens Naturals - Organic Just Raspberries', 
            'Max Crop Liquid Fertilizer', 
            'No powdery mildew 1 quart', 
            'SWISS GREEN ORGANIC PLANT GROWTH PROMOTER STRAWBERRY Fertilizer',
            'Greatindos All in 1 Organic Fungicide', 
            'CUREAL Best Fungicide & Bactericide', 
            'NATIVO FUNGICIDE', 
            'Tomato Fertilizer Organic',
            'ACROBAT FUNGICIDE', 
            'Virus Special', 
            'Roko Fungicide', 
            'OMITE INSECTICIDE', 
            'Propi Propineb 70% WP Fungicide ',
            'V Bind Viral Disease Special', 
            'Syngenta Amistor Top Fungicide']
    np.set_printoptions(suppress=True)
    import tensorflow
    from PIL import Image, ImageOps
    import cv2
    model= tensorflow.keras.models.load_model('Prathamesh.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img_path = st.file_uploader("Upload:",type=['png','jpeg','jpg'])
    if img_path:
        image = Image.open(img_path)
        fig=plt.figure()
        plt.imshow(image)
        plt.axis("off")
        st.pyplot(fig)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0)
        data[0] = normalized_image_array
        prediction = model.predict(data)
        st.write("\nPrediction:",class_na[np.argmax(prediction)])
        st.write("\nConfident:",100 * np.max(prediction))
        st.write("\nFertilizer:",class_fe[np.argmax(prediction)])
#===================================================================================================================================================

add_selectbox = st.sidebar.markdown(':sunglasses: Partner: Siddhesh Deepak Patil :sunglasses:')
add_selectbox = st.sidebar.markdown('Seat Number:____________________')


