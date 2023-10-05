# Unsupervised Calibration through Prior Adaptation

This is the code to reproduce the results of the paper [Unsupervised Calibration through Prior Adaptation](https://arxiv.org/abs/2307.06713). 

## Installation

Install a conda environment to work with python 3.11.4:
```bash
conda create -n ucpa python=3.11.4
conda activate ucpa
pip install -r requirements.txt
```

Download the model weights:
```bash
python prepare_models.py --root_directory=. --model="gpt2-xl"
```

## Usage

To run the code, use the following command:
```bash
bash run.sh
```
Code support GPU and CPU usage. It will try to allocate part of the model in the GPU, up to the specified size in the config file.


## Citation

If you use this code, please cite the following paper:

```
@misc{estienne2023unsupervised,
      title={Unsupervised Calibration through Prior Adaptation for Text Classification using Large Language Models}, 
      author={Lautaro Estienne and Luciana Ferrer and Matías Vera and Pablo Piantanida},
      year={2023},
      eprint={2307.06713},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


