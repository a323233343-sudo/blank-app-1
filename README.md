# 🎈 Blank app template

A simple Streamlit app template for you to modify!

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. # Install the python version don't newer then v3.13

    https://www.python.org/downloads/

2. # active virtual environment run on powershell

   ```
   $ cd {to where you download path}
   $ python -m venv env
   ```
   
   ```
   $ py -{python version} -m venv env   # if you more then one version of python 
   ```
   ```
   $ Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
   $ .\env\Scripts\Activate.ps1    #
   ```

3. # Install the requirements

   ```
   $ pip install wheel setuptools # for the first time
   $ pip install -r requirements.txt
   ```

4. # Run the app

   ```
   $ streamlit run streamlit_app2.py
   ```
