import streamlit as st

import tempfile
import torch
import numpy as np


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
    input_tr, target_tr = torch.tensor(input_arr), torch.tensor(target_arr)
    input_batch = input_tr.unsqueeze(0)

    print('Running inference')
    output_batch = model(input_batch.float())
    
    output_tr = output_batch.squeeze(0)
    output_arr = output_tr.detach().numpy()

    print('Saving output')
    saveOutput(output_arr, target_arr)

    return 1


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
        model = getModel(5)
        resid = predictFlow(np_arr, model)
        print(resid)
        if resid == 1:
            st.image('result.png')
            st.balloons()
