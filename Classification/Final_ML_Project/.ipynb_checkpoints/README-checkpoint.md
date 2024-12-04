# **Explanation of the Final Project**

This directory contains several notebooks, each with different purposes.

* The notebook titled `EDA_FinalProject_MLCourse.ipynb` contains the Exploratory Data Analysis of the database provided for the final project. This is quite detailed since the fact that the features did not have names that could provide any clues about their significance made it more challenging to make intuition-based decisions. Therefore, the EDA is complete in this notebook.

* The notebook titled `Modelo_Final_Elizabeth.ipynb` contains the Feature Engineering and the path followed to find the features and transformations that best suited the model. Various Feature Importance methods were applied to select the variables that made the model work most effectively. In this notebook, not only are three final models created, but it also details the reasons why these three were the final candidates. Additionally, it describes the decision criteria considered to decide which one to use for deployment.

* The notebook titled `FinalExam_Deploy.ipynb` contains the final model deployment. Using the `preprocessing.py` module, a Pipeline is created for the corresponding transformation of the variables to be used for the model's prediction. By applying the functions written in the module, the model is saved in a pickle file (`.pkl`), and using FastAPI and Docker, it is deployed.

      It is important to clarify that for the proper deployment and functioning of the model, the files `Dockerfile`, `model.py`, and `main.py` are created. In          the last file, the types of variables the model receives are specified, and with the help of `Pydantic`, it is defined which variables are mandatory and which have default values, etc. For a better understanding of how it works, we invite you to review the file.


