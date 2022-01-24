# Group Invariant Dictionary Learning

This respository is implementation of the [paper](https://arxiv.org/abs/2007.07550) "Group Invariant Dictionary Learning" by [Yong Sheng Soh](https://yssoh.github.io/).

## Usage

This codebase is tested on `Python 3.6`. To install dependencies in the `requirements.txt` file, call `pip install -r requirements.txt`. 

If you would like to install and develop `gidl` after activating your [virtual environment](https://docs.python.org/3/library/venv.html) , you can call `pip install --editable .`.

### Running on your data

1. Process data as a `numpy.ndarray`.
2. Create a class object for the data using a dictionary learning framework of choice. The following frameworks are available:
    * `RegularDL`: Classic Dictionary Learning
    * `ConvDL`: For Integer Shift Invariance
    * `CtsShiftDL`: For Continuous Shift Invariance
    * `ConvInterpDL`: For Integer Shift Invariance with Interpolation
    * `SyncDL`: For Invariance to Orthogonal Transformations
    
    Optional knobs such as `learnedDict`, `reg_param` and `num_loops` can be set during initialization of data here.
3. Learn the dictionary with `.learn_dict_mul(num_iterations)` method where `num_iterations` is the number of iterations of alternating minimization.
4. Dictionary is accessible by calling `.learnedDict`.

#### Example
To learn a continuously shift invariant dictionary for `your_data` with `20` iterations of alternating minimization:
```python
>>> import gidl

>>> model = gidl.CtsShiftDL(your_data)  # initialization of data
>>> model.learn_dict_mul(20)            # learn the dictionary
>>> model.learnedDict                   # to access dictionary
```

### Running experiments in the paper

To replicate experiments in the paper, refer to `experiments/experiments.ipynb`.

#### Datasets
Datasets used in `experiments.ipynb` are *not* included in this repository, but scripts to preprocess/generate them can be found in `experiments/scripts/`:

* ECG data used for the ECG experiment can be downloaded from [MIT-BIH Arrhythmia Database](https://doi.org/10.13026/C2F305) and preprocessed using `preprocess_100ecg.py`.
* Synthetic data used for the synchronization experiment can be generated using `generate_data_syncdl.py`.

## Citation

```
@article{soh2021gidl,
   title={Group Invariant Dictionary Learning},
   volume={69},
   ISSN={1941-0476},
   url={http://dx.doi.org/10.1109/TSP.2021.3087900},
   DOI={10.1109/tsp.2021.3087900},
   journal={IEEE Transactions on Signal Processing},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Soh, Yong Sheng},
   year={2021},
   pages={3612–3626}
}
```

## Credits

This repository is based on an original codebase by Asst Prof [Yong Sheng Soh](https://yssoh.github.io/) whom is the author of the paper.