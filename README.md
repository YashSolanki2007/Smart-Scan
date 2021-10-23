# Smart-Health
## Inspiration
I was inspired to create this project on seeing all the mis beliefs that people have about the coronavirus that lead me into making this website to clear all the doubts that they have and also help doctors validate their predictions that they make with thr RT-PCR tests.

## What it does
It does a variety of covid assistance related things like med scans to detect COVID in lungs, track COVID cases around the world, check all the vaccine news around the world and also an AI chatbot that answers questions about the pandemic

## How we built it
I used flask for the backed and tensorflow for the ai model

## Challenges we ran into
The AI model was difficult to get perfect. Also the model was overfit in the beginning and it took a lot of fine tuning of the hyperparameters to get the model to a maximum accuracy in a short time frame with a limited computational power


# Please Read
## The dataset provided here does not have all the images because it was too large.
# LINK TO THE COMPLETE DATASET 
# https://www.kaggle.com/pranavraikokte/covid19-image-dataset

## Running the project 
To run this project first git clone it into your desktop. Then download rhe dataset that was linked abive and rep
ace the current project dataset with that one. Once that is done just install flask, tensorflow, openpyxl, requests and then run python3 main.py on mac or linux and python main.py on windows


Also delete the viral pneumonia directory from both training and testing and keep only the covid and normal directory
