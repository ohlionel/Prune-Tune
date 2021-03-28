# Prune-Tune
Official code repository for AAAI2021 paper Finding Sparse Structures for Domain Specific Neural Machine Translation

This project is based on [Neurst](https://github.com/bytedance/neurst), an open source Neural Speech Translation Toolkit. 

Here is an example to train a general model for En-De Translation, then adapt to a target domain(novel) via Prune-Tune.


## Neurst Installation
Install from source:
```
git clone https://github.com/ohlionel/Prune-Tune.git
cd Prune-Tune/neurst/
pip3 install -e .
```
please see installation details in [Neurst](https://github.com/ohlionel/Prune-Tune/tree/main/neurst)

## Data Preprocess
We use two datasets:
|     |   |
|  ----  | ----  | 
| General Domain  | WMT14(En-De) | 
| Target Domain  | [Novel Dataset](https://opus.nlpl.eu/Books.php) from OPUS |


<!-- 
General Domain: WMT14(En-De)

Target Domain: [Novel Dataset](https://opus.nlpl.eu/Books.php) from OPUS -->

By runing with
```
cd neurst
bash ./scripts/prepare-wmt14en2de-wp.sh
cp data/wmt14_en_de/vocab data/novel/
bash ./scripts/prepare-novel-wp.sh
```
we will get the preprocessed training data and raw testsets under directory `data/wmt14_en_de/` and `data/novel`, i.e.
```bash
data/wmt14_en_de/
├── vocab  # wordpiece codes
├── newstest2013.de.txt   # newstest2013 as devset
├── newstest2013.en.txt
├── newstest2014.de.txt  # newstest2014 as testset
├── newstest2014.en.txt
├── prediction_args.yml   # the arguments for prediction
├── train.de.txt  # the raw training data
├── train.en.txt
├── training_args.yml  # the arguments for training
├── translation_wordpiece.yml  # the arguments for training data and data pre-processing logic
├── validation_args.yml  # the arguments for validation
├── training_records # directory of training TFRecords
    ├──train.tfrecords.00000
    ├──train.tfrecords.00001
    ├── ...
├── ...
```

## Train the General Domain Model
We can directly use the yaml-style configuration files generated above to train a general domain model on WMT14(En-De).
```bash
python3 -m neurst.cli.run_exp \
    --config_paths data/wmt14_en_de/training_args.yml,data/wmt14_en_de/translation_bpe.yml,data/wmt14_en_de/validation_args.yml \
    --hparams_set transformer_big \
    --model_dir models/benchmark_big
```

## Prune the General Domain Model 
We can simply prune a model with Neurst, see [Weight Pruning](https://github.com/ohlionel/Prune-Tune/tree/main/neurst/examples/weight_pruning) for details.
```bash
python3 -m neurst.cli.run_exp \
    --config_paths data/wmt14_en_de/training_args.yml,data/wmt14_en_de/translation_wordpiece.yml,data/wmt14_en_de/validation_args.yml \
    --hparams_set transformer_big \
    --pretrain_model models/benchmark_big/best/ \
    --model_dir models/sparsity_10 \
    --checkpoints_max_to_keep 3 \
    --initial_global_step 250000 \
    --train_steps 10000 \
    --summary_steps 200 \
    --save_checkpoints_steps 500 \
    --pruning_schedule polynomial_decay \
    --initial_sparsity 0 \
    --target_sparsity 0.1 \
    --begin_pruning_step 0 \
    --end_pruning_step 5000 \
    --pruning_frequency 100 \
    --nopruning_variable_pattern "(ln/gamma)|(ln/beta)|(modalit)" 
```
We will get the pruned model `models/sparsity_10` in which 10% of parameters is pruned. `sparsity_10/mask.pkl` save all binary pruning masks, where 0 indicates zero-value pruned weight.

## Partially Tune the Model with Taget Domian Dataset
According to the pruning mask file `sparsity_10/mask.pkl`, we can only update those pruned weight during tuning. 
```bash
python3 -m neurst.cli.run_exp \
    --config_paths data/novel/training_args.yml,data/novel/translation_wordpiece.yml,data/novel/validation_args.yml \
    --hparams_set transformer_big \
    --pretrain_model models/sparsity_10 \
    --model_dir models/sparsity_10_novel \
    --initial_global_step 0 \
    --train_steps 10000 \
    --summary_steps 200 \
    --save_checkpoints_steps 1000 \
    --partial_tuning \
    --mask_dir models/sparsity_10/mask.pkl 
```
## Evalution on General and Target Domain
Evaluate on target domain with full model:
```bash
python3 -m neurst.cli.run_exp \
    --entry mask_predict \
    --config_paths data/novel/prediction_args.yml \
    --model_dir models/sparsity_10_novel/best
```
Evaluate on general domain with the general sub-network:
```bash
python3 -m neurst.cli.run_exp \
    --entry mask_predict \
    --config_paths data/wmt14_en_de/prediction_args.yml \
    --model_dir models/sparsity_10_novel/best \
    --mask_dir models/sparsity_10/mask.pkl \
    --apply_mask
```

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{jianze2021prunetune,
  title={Finding Sparse Structures for Domain Specific Neural Machine Translation},
  author={Jianze Liang, Chengqi Zhao, Mingxuan Wang, Xipeng Qiu, Lei Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```





