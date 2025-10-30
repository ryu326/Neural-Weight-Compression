# NWC: Neural Weight Compression

## Environment
```
pip install -r requirements.txt
```

## Dataset
Before generating dataset, we have to download the checkpoints of Llama from HuggingFace:
```
cd ./dataset
python down_hf_model.py
```

Then, generate the dataset .pt file from Llama model weights:
```
bash dataset_generation.sh
cd ..
```


## Training NWC models

Training NWC with lambda:
```
cd ./training

CUDA_VISIBLE_DEVICES=0 python -u train_nwc.py \
    --architecture nwc_ql \
    --dataset_path ../dataset/block_pt/meta-llama--Meta-Llama-3-8B/col_1024.pt \
    --dataset block_seq_ql_random \
    --iter 200000 \
    --input_size 16 \
    --M 16 \
    --Q 4 \
    --dim_encoder 512 \
    --batch_size 2048 \
    --loss rdloss_ql \
    --lmbda $lmbda
```

<details>
<summary>Details of each argument are as follows:</summary>

* dataset_path: Weight dataset path
* input_size: weight chunk size
* M: entropy bottleneck channel size
* Q: number of Quality levels
* dim_encoder: hidden dimension of encoder & decoder
* lmbda: Bit-rate distortion parameter
* dataset: use 'block_seq_ql_random' for random Quality level training
* loss: use 'rdloss_ql' for Quality level

</details>

We can use the following command for training NWC with multiple lambdas:
```
bash scripts/train.sh
```


## Compression & Evaluation
### Layerwise Hessian generation
We need to generate the Hessian before compressing LM models:
```
cd ../inference
bash scripts/input_hessian_llama.sh
```
This process might take some time.

For Llama2-7B, Llama2-13B, Llama2-70B, we can use [QuIP#](https://github.com/Cornell-RelaxML/quip-sharp)'s precomputed Hessians: 
```
bash scripts/down_quip_hess.sh
```

### Compression & Evaluation Llama model
Pseudo compress LLMs and evaluation perplexity & zeroshot accuracy: 
```
bash scripts/comp_eval_llama.sh
```

<details>
<summary>Details of each argument are as follows:</summary>

* base_model: LM model hf path
* comp_model_path: trained compression model ckpt path
* in_hess_path: hessians path
* ft_epochs: number of epochs for blockwise recovery finetuning, use 0 for skipping finetuning
* comp_batch_size: The number of weight columns that are processed by the compression model in a single forward pass.

</details>

