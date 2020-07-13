# ML interface

This application is capable of building the model, save it for future purpose, without writing a single piece of code! 
<br>
A special library, `Streamlit` is used to develop the application's interface. The documentation can be found [here](https://www.streamlit.io/)

### 🦾 Usage

* The application can be found [here.](https://machine-learning-interface.herokuapp.com/)
* Upload the `.csv` file you wanted to build the model on.
* Select the *features/columns* form the drop-down menu.
* Handle the *missing data(NaN)* using different strategies. (A warning is displayed if it cannot be added. Try another strategy)
* *Enccode* the columns for training. (One-Hot encoder)
* Split the data into *training* and *dev/test sets.* (The max. split is set to 0.3 i.e., the dataset is split in the ratio, 70/30)
* *Normalise* the data. 
* Select the *algorithm* for predicting.
* *Modify the hyperparameters* on the sidebar for better results.
* Click `save` button to *save the model* for later use.

### ⛏️ Develop

* Clone the repository from above or in the commad line use:
```console
$ git clone https://github.com/Nitin1901/machine-learning-interface.git
```
* Change you current working directory.
```console
$ cd machine-learning-interface
```
* Create a virtual environment(recommended) and activate.
```console
$ python -m venv ml-intreface
$ ml-interface\Scripts\activate.bat
```
* Install the required packages from `requirements.txt`. You can manually install each package or use:
```console
$ pip install -r requirements.txt
```
* Open `app.py` in a text editor and start making changes.
* Run the app locally 
```console
$ streamlit run app.py
``` 

*If you wish to contribute, `fork` the repository, develop and create a `pull request`.*

