Granulometric analysis web app
---
- [Info](#Info)
- [Online use](#Online use)
- [Local Installation](#Installation)
- [Local use](#Locale use)
---

# Info 

Web application for Granulometric (grain-size) data analysis developped during an IRMAR internship


### Author :
Jean Pousset (4th year Applied Maths INSA Rennes)  contact : [pousset.jean1@gmail.com](pousset.jean1@gmail.com)

DON'T HESITATE TO CONTACT ME FOR HELP / MORE EXPLANATIONS

### [Internship report](IRMAR_report/Rapport_IRMAR_décomposition_courbe_granulométriques_Jean_POUSSET.pdf)
### Guidance:
- [Valérie Monbet (IRMAR)] (https://perso.univ-rennes1.fr/valerie.monbet/)
- [Fabrice Mahé (IRMAR)] (https://perso.univ-rennes1.fr/fabrice.mahe/)

### Data proveiders :
- François Pustoc'h (CreAAH)
- [Simond Puaud (CreAAH)](https://creaah.cnrs.fr/team/puaud-simon-1/) 

Code writer of the algorithme that produced the BLASSO results : [Clément Elvira (IETR / SCEE)](https://c-elvira.github.io/)

# Online use

You can easily use the online application via the link: https://granulometric-analysis.streamlit.app/ . 
When you get to the page, you may have to click on the alert to restart the server. It automatically shuts down when there has been no access for 48 hours.


You can also do some modifications on the app and run it locally on your computer. In that case follow the instruction in **Local Installation**. Then you can modify the app in the *streamlit_app.py* script and lunch it by following the instruction of **Local use**.


# Local Installation

Start by downloading or cloning the entire repository locally.

⚠️ **Warning : <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Windows_logo_-_2012_%28dark_blue%29.svg/1920px-Windows_logo_-_2012_%28dark_blue%29.svg.png" alt="Logo Windows" width="15"> WINDOWS users :** ⚠️

If you are on windows command keywords for Python may be differents. For the next steps, try replacing `python3` with `py` and `pip` with `py -m pip`.


### Python installation

First check if you have python (version 3.0 or greater) and pip installed in your computer. Open a terminal window in your computer and run the following commands :


### 2. **Utilisation d'onglets Markdown (GitHub Flavored Markdown - GFM)**


<details>
  <summary><strong>Linux/MacOS</strong></summary>

  ```bash
  sudo apt update && sudo apt install -y package-name
  ./run.sh
  ```
</details>
<details>
  <summary><strong>Windows</strong></summary>

  ```bash
  sudo apt update && sudo apt install -y package-name
  ./run.sh
  ```
</details>



```bash
python3 --version
pip --version
```

If not (an error is prompted) please install it with the following commands depending on your OS.
- <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Windows_logo_-_2012_%28dark_blue%29.svg/1920px-Windows_logo_-_2012_%28dark_blue%29.svg.png" alt="Logo Windows" width="15"> **Windows** : 

Go on [official python installation page](https://www.python.org/downloads/) and follow instruction to install it on Windows.

- <img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" alt="Logo Linux" width="15"> **Linux**:
```bash
sudo apt update
sudo apt install python3
sudo apt install python3-pip
```

-  **MacOS** (with homebrew) :

If you need you can find the installation documentation for homebrew  [here](https://brew.sh/)
```bash
brew update
brew install python
```

If you don't want to use homebrew you can install python directly on the [official python installation page](https://www.python.org/downloads/)


### Streamlit and other package

First you need to install streamlit (package to lunch web application in python). Open a terminal window and run this command :
```bash
pip install streamlit
```
If it doesn't work try with `pip3` instead of `pip`. Verify the installation with :
```bash
streamlit --version
```

Then install the other packages/dependancies with the following command (in the project repertory):
```bash
pip install -r requirements.txt
```

You can also do it manually with :
```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install plotly
pip install openpyxl
```


# Locale use

Open a terminal and navigate to where you put or cloned the repository. (i.e type `cd {directory_path}` where *directory_path* is the path). When you type `ls` in the terminal you should have the following list prompted :

```bash
$ ls
> B-LASSO_imports       README.md             ref_curves_&_exemples
> Data                  backends              requirements.txt
> IRMAR_report          exports               streamlit_app.py
```

To run the app you just have to write the following command :
```bash
streamlit run streamlit_app.py
```

If you are in Windows <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Windows_logo_-_2012_%28dark_blue%29.svg/1920px-Windows_logo_-_2012_%28dark_blue%29.svg.png" alt="Logo Windows" width="15"> it may not work. Try :
```bash
py -m streamlit run streamlit_app.py
```

After a few seconds you should see this message on your terminal :

```bash
> You can now view your Streamlit app in your browser.
> 
>   Local URL: http://localhost:8501
>   Network URL: http://192.168.1.18:8501
>   External URL: http://90.49.34.45:8501
```

It means the app has been successfully lunched. To access it please open the Local URL (in my case http://localhost:8501) in your browser. 

To close the application go back to your terminal. You can simply close your terminal or press : ctrl+C  

## Note

If you encounter any error/problem please contact me at : [pousset.jean1@gmail.com](mailto:pousset.jean1@gmail.com)

