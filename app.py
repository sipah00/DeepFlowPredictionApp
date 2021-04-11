import streamlit as st

import tempfile
import torch
import numpy as np
from imageio import imread


from DfpNet import TurbNetG 
from utils import InputData, saveOutput


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@st.cache
def getModel(expo):
    netG = TurbNetG(channelExponent=expo).to(device)
    netG.load_state_dict(torch.load(f'models/model_w_{expo}', map_location=device))
    netG.eval()
    return netG

def predictFlow(np_arr, model):
    input_data = InputData(np_arr, removePOffset=True, makeDimLess=True)
    input_arr, target_arr = input_data.input, input_data.target 
    input_tr = torch.tensor(input_arr)
    input_batch = input_tr.unsqueeze(0)

    print('Running inference')
    output_batch = model(input_batch.float())
    
    output_tr = output_batch.squeeze(0)
    output_arr = output_tr.detach().numpy()

    print('Saving output')
    saveOutput(output_arr, target_arr)

    return 1

def solver(np_arr, expo=5):
    model = getModel(expo)
    resid = predictFlow(np_arr, model)
    return resid



st.sidebar.title('Deep Flow Prediction App')

ux = st.sidebar.number_input('Enter Ux')
uy = st.sidebar.number_input('Enter Uy')
airfoil_type = st.sidebar.multiselect('Airfoil type', ['NACA 2412', 'NACA 2415'])


st.sidebar.write('OR')
st.sidebar.write('Upload input file in npz format')

upload_file = st.sidebar.file_uploader('Choose file...')

sts = st.sidebar.button('Submit')

if sts:
    if upload_file:
        file = upload_file.read()
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(file)
        np_arr = np.load(tfile.name)['a']
        resid = solver(np_arr, expo=5)
        print(resid)
        if resid == 1:
            st.image('result.png')
            st.balloons()

    elif ux and uy and airfoil_type:
        print(airfoil_type)
        im = imread(f'airfoils/{airfoil_type[0]}.png')
        np_im = np.array(im)
        np_im = np_im / np.max(np_im)

        st.markdown('## Airfoil')
        st.image(np_im)
        
        np_im = np.flipud(np_im).transpose() # in model's input format


        ux = float(ux)
        uy = float(uy)

        fx = np.full((128, 128), ux) * np_im
        fy = np.full((128, 128), uy) * np_im

        np_im = 1 - np_im

        np_arr = np.stack((fx, fy, np_im))

        resid = solver(np_arr, expo=5)
        print(resid)
        if resid == 1:
            st.image('result_velX_pred.png')
            st.image('result_velY_pred.png')
            st.image('result_pressure_pred.png')
            st.balloons()






