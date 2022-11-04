# track irregularity forecasting
## abstract
To ensure the safety of railroad operations, it is important to monitor and forecast track geometry irregularities. A higher safety requires forecasting with a higher spatiotemporal frequency. For forecasting with a high spatiotemporal frequency, it is necessary to capture spatial correlations. Additionally, track geometry irregularities are influenced by multiple exogenous factors. In this study, we propose a method to forecast one type of track geometry irregularity, vertical alignment, by incorporating spatial and exogenous factor calculations. The proposed method embeds exogenous factors and captures spatiotemporal correlations using a convolutional long short-term memory (ConvLSTM). In the experiment, we compared the proposed method with other methods in terms of the forecasting performance. Additionally, we conducted an ablation study on exogenous factors to examine their contribution to the forecasting performance. The results reveal that spatial calculations and maintenance record data improve the forecasting of the vertical alignment.

## Contribution
The main contributions of the proposed method are summarized as follows.
- The proposed method can forecast vertical alignment with a high spatial or temporal frequency. This is the first attempt
to forecast at such high frequencies.
- The proposed method extends the basic ConvLSTM to capture the spatial correlation and the effect of exogenous factors.

## Notice
In the experimental results in the paper we use the track geometry irregularity data of Tokaido Shinkansen. 
However, in order to use the data, you must sign a confidentiality agreement with the Central Japan Railway Company.

You can check the operation of the code by using dummy data.

## Installation
We support `Anaconda` enviroment.
To create virtual enviroment, run:
```
conda create -n hoge python=3.8
```

To activate virtual enviroment, run:
```
conda activate hoge
```

To install the dependencies, run:
```
pip install -r requirements.txt
```

## Setup mlflow
To create conf/local/mlflow.yml, run:
```
kedro mlflow init
```

And modify
```
tracking:
    params:
        dict_params:
        flatten: False 
        recursive: True  
        sep: "."
        long_params_strategy: fail 
```
to
```
tracking:
    params:
        dict_params:
        flatten: False 
        recursive: True  
        sep: "."
        long_params_strategy: tag
```

## Prepare dataset
To train a model, you need to run the following four processes in order.

To preprocess track irregularity data, run:
```
kedro run --pipeline preprocess
```

To preprocess exogenous data, run:
```
kedro run --pipeline preprocess_exogenous
```

To make track irregularity dataset, run:
```
kedro run --pipeline data_engeering
```

To make exogenous dataset, run:
```
kedro run --pipeline exogenous_series
```

## Training
After setting the parameters, to train the model, run:
```
kedro run --pipeline train
```

After training, to test the model, run:
```
kedro run --pipeline test
```

After training, to inference, run:
```
kedro run --pipeline inference
```

## visualize
After learning and inference, you can visualize the figures in the paper with the following process.

To visualize figure 3, run `comparison_result_visualize.ipynb` in jupyter-notebook.

To visualize figure 4, run `ablation_study_visualize.ipynb` in jupyter-notebook.

To visualize figures 5 and 6, run `comparison_result_visualize.ipynb` in jupyter-notebook.