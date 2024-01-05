## Streamlit Movie lens recommendation with explanations
![Recommendation web application](<Rec model Web App.png>)

Check out the  [Video Demo](<Streamlit movie recommendatioan.mov>)


## Deployment Instruction  

Step 1: Clone the repository
---
```
> cd $HOME
> git clone git@github.com:shett044/movie-lens-camile.git
```

Step 2: Copy the model file to Git repo
----

Since Git does not store files > 300mb, to deploy, I stored the file in the Gdrive. Please download them.
[Drive Link](https://drive.google.com/drive/folders/1yX6PPj-bRTEn_JAXN8gWfKpXT8oSJcVr?usp=sharing)

Copy the downloaded joblib file in following directory

```
> cp *joblib movie-lens-camile/results/best_model/class/rank_class_part1/
```

Step 3:
---
Start the docker daemon or docker desktop
Run the following command
```
> cd movie-lens-camile
> docker build -t movie-rec-camile:Final ./
> docker run  -ti -p 8501:8501 movie-rec-camile:Final

```
Open in browser with the following link

http://localhost:8501/



## Data Analysis report
Please use [Data analysis report](DataAnalysis-Report.ipynb) to explore data analysis and exploration done.


In this notebook we have covered important factors that determine user rating using Mutual information technique (similar to t-test but more effective)

## Modeling and Evaluation Report
Please refer to [Google Word document](https://docs.google.com/document/d/18DAr6bgHnC-IDhMT1xQ241MG0qfB0xE9rVSf15URvYM/edit?usp=sharing)

## Business Evaluation Report
Please refer to [Google Word document](https://docs.google.com/document/d/1QLItGTm5vErAvnov-fSO3sODGbk6V2Kz_cImVkanGFg/edit?usp=sharing)

## Additional recommendation Report
Please refer to [Google Word document](https://docs.google.com/document/d/1LIiQk8NJ1HM_exvxeJ7L4pYeSlBo85NV0E4ZDDVd2H4/edit?usp=sharing)



## Experimentaion
Please refer to [Google Word document](https://docs.google.com/document/d/1VzVU6xSEdCLOylmJabwTL-mnOWPk1JmnJl5Nh-J-h9A/edit?usp=sharing)




### Dataset
Latest Movie Lens Dataset. 

Dataset is Implict Feedback, If there is interaction between user and item, then target value will be 1.So if there is rating value between user and movie, then target value is 1, otherwise 0. 

For negative sampling, ratio between positive feedback and negative feedback is 1:4 in trainset, and 1:19 in testset. 

