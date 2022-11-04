# track irregularity forecasting


## Installation
We support `Anaconda` enviroment.
To create virtual enviroment run:
```
conda create -n hoge python=3.8
```

To activate virtual enviroment run:
```
conda activate hoge
```

## setup mlflow
To create conf/local/mlflow.yml run:
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