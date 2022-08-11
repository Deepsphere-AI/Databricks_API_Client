import streamlit as st

from DSAI_Utility.DSAI_Utility import All_Initialization,CSS_Property
import traceback
import sys
from PIL import Image
from DSAI_APIClient import Model_Result

if __name__ == "__main__":
    st.set_page_config(page_title="Databricks MLOps", layout="wide")
    try:
        # Applying CSS properties for web page
        CSS_Property("DSAI_Utility/style.css")
        # Initializing Basic Componentes of Web Page
        All_Initialization()
        # Calling API Endpoint for model prediction
        Model_Result()
    except BaseException as e:
        col1, col2, col3 = st.columns([1.15,7.1,1.15])
        with col2:
            st.write('')
            st.error('In Error block - '+str(e))
            traceback.print_exception(*sys.exc_info())
