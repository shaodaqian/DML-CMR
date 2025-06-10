

# DML-CMR

This repository includes the implementation of the method DML-CMR, a unbiased estimator for CMRs problems.

## Dependencies

Install all dependencies

```
pip install -r requirements.txt
```



## Basic Usage

Run experiments for IV regression using the aeroplane demand dataset with low-dim, high-dim and the two real world datasets.

```
python iv/main_low_d.py
python iv/main_mnist.py
python iv/main_realworld.py
```

Run experiments for PCL using the aeroplane demand dataset and the dSprites datasets.

```
python pcl/main.py config/demand/dml_pcl.json ate
python pcl/main.py config/dsprites/dml_pcl.json ate
```