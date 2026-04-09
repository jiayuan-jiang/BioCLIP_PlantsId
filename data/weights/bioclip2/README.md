---
license:
- mit
language:
- en
library_name: open_clip
model_name: BioCLIP 2
model_description: >-
  Foundation model for biology organismal images. It is trained on
  TreeOfLife-200M on the basis of a CLIP model (ViT-14/L) pre-trained on
  LAION-2B. BioCLIP 2 yields state-of-the-art performance in recognizing various
  species. More importantly, it demonstrates emergent properties beyond species
  classification after extensive hierarchical contrastive training.
tags:
- biology
- CV
- images
- imageomics
- clip
- species-classification
- biological visual task
- multimodal
- animals
- plants
- fungi
- species
- taxonomy
- rare species
- endangered species
- evolutionary biology
- knowledge-guided
- zero-shot-image-classification
datasets:
- imageomics/TreeOfLife-200M
- GBIF
- bioscan-ml/BIOSCAN-5M
- EOL
- FathomNet
new_version: imageomics/bioclip-2.5-vith14
---

<!--
Image with caption (jpg or png):
|![Figure #](https://huggingface.co/imageomics/<model-repo>/resolve/main/<filepath>)|
|:--|
|**Figure #.** [Image of <>](https://huggingface.co/imageomics/<model-repo>/raw/main/<filepath>) <caption description>.|
-->

<!--
Notes on styling:

To render LaTex in your README, wrap the code in `\\(` and `\\)`. Example: \\(\frac{1}{2}\\)

Escape underscores ("_") with a "\". Example: image\_RGB
-->

# Model Card for BioCLIP 2

BioCLIP 2 is a foundation model for biology organismal images. It is trained on [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) on the basis of a [CLIP](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K) model (ViT-14/L) pre-trained on LAION-2B.
BioCLIP 2 yields state-of-the-art performance in recognizing various species. More importantly, it demonstrates emergent properties beyond species classification after extensive hierarchical contrastive training.

## Model Details

### Model Description

Foundation models trained at scale exhibit emergent properties beyond their initial training objectives.
BioCLIP 2 demonstrates such emergence beyond species classification by scaling up the hierarchical contrastive training proposed by [BioCLIP](https://imageomics.github.io/bioclip/).
The model is trained on [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) (the largest and most diverse available dataset of biology images).
We evaluate BioCLIP 2 on a diverse set of biological tasks. Through training at scale, BioCLIP 2 improves species classification by 18.1% over BioCLIP. More importantly, we demonstrate that BioCLIP 2 generalizes to diverse biological questions beyond species classification solely through species-level supersvision. Further analysis reveals that BioCLIP 2 acquires two emergent properties through scaling up hierarchical contrastive learning: inter-species ecological alignment and intra-species variation separation.

- **Developed by:** Jianyang Gu, Samuel Stevens, Elizabeth G Campolongo, Matthew J Thompson, Net Zhang, Jiaman Wu, Andrei Kopanev, Zheda Mai, Alexander E. White, James Balhoff, Wasila M Dahdul, Daniel Rubenstein, Hilmar Lapp, Tanya Berger-Wolf, Wei-Lun Chao, and Yu Su
- **Model type:** The model uses a ViT-L/14 Transformer as an image encoder and uses a masked self-attention Transformer as a text encoder.
- **License:** MIT
- **Fine-tuned from model:** CLIP pre-trained on LAION-2B, ViT-L/14 ([Model weight](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K))

### Model Sources

- **Homepage:** [BioCLIP 2 Project Page](https://imageomics.github.io/bioclip-2/)
- **Repository:** [BioCLIP 2](https://github.com/Imageomics/bioclip-2)
- **Paper:** [BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](https://doi.org/10.48550/arXiv.2505.23883)
- **Demo:** [BioCLIP 2 Demo](https://huggingface.co/spaces/imageomics/bioclip-2-demo)

## Uses

### Direct Use

The model can be used for zero-shot classification provided the species names.
It can also be used for few-shot classification with some images serving as the support set.
Additionally, it is also recommended to use BioCLIP 2 as a visual encoder for other biological visual tasks.

## Bias, Risks, and Limitations

BioCLIP 2 is trained on an imbalanced dataset. Specifically, the TreeOfLife-200M dataset exhibits a long-tailed distribution across taxa.
Therefore, the predictions of BioCLIP 2 might be biased toward well-represented species. For more details, see the [discussion in the TreeOfLife-200M dataset card](https://huggingface.co/datasets/imageomics/TreeOfLife-200M#considerations-for-using-the-data).

BioCLIP 2 and TreeOfLife-200M provide great potential to improve and enhance existing conservation efforts, in particular by facilitating recognition of threatened species. 
Unfortunately, as with many open-source efforts to further conservation goals, there is also potential for bad actors to make use of these tools for malicious purposes. Though the improvement on threatened species could make it easier for poachers to identify protected species, these types of tools are a force-multiplier to monitor illicit trade and sales of these same species. The primary risk to endangered species comes from disclosure of precise location information rather than improved classification capability. Our data does not provide geo-tagged information of the organisms included, minimizing the vulnerabilities that could be used in poaching.

<!-- 
### Recommendations

This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. 

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.
-->

## How to Get Started with the Model

You can use the `open_clip` library to load BioCLIP 2.

```
import open_clip
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')
```

## Training Details

### Training Data

The model was trained with [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M) (Revision [a8f38b4](http://huggingface.co/datasets/imageomics/TreeOfLife-200M/tree/a8f38b4388579862c56ae57d6f094c2ac0e92e12)).
The dataset consists of nearly 214M images covering 952K taxa. 
The scale of TreeOfLife-200M fosters the emergent properties of BioCLIP 2.

In addition, we also used a subset of LAION-2B that consists of 26M samples for experience replay.
This part of data was downloaded from the first three parquet metadata files of LAION-2B, and the first 4,000 tar files were used.

### Training Procedure 

#### Preprocessing

Standard CLIP image preprocessing is adopted in the training.

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

We used an Adam optimizer with a maximum learning rate of 1e-4. 1,875 warming steps were adopted, followed by cosine decay.
The batch size of biological images was 2,816 per GPU, and that of replay data was 320 per GPU.
We trained the model on 32 GPUs for 30 epochs, with a weight decay of 0.2.
Each input image was resized to 224 x 224 resolution.

## Evaluation

We evaluated the model on both species classification and other biological visual tasks.

### Testing Data

For species classification tasks, we tested BioCLIP 2 on the following 10 tasks:
* [NABirds](https://dl.allaboutbirds.org/nabirds): We used 555 visual categories of 48,640 images for test.
* [Meta-Album](https://meta-album.github.io/): We used the Plankton, Insects, Insects2, PlantNet, Fungi, PlantVillage, and Medicinal Leaf datasets from Meta-Album. 
* [IDLE-OO Camera Traps](https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps): Species identification in camera trap images is a real-world scenario that BioCLIP 2 can be applied to.
We collected a class-balanced test set from five LILA-BC camera trap datasets. For more information on this test set, please visit the [dataset page](https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps).
* [Rare Species](https://huggingface.co/datasets/imageomics/rare-species): This dataset was introduced in the first BioCLIP paper.
It consists of 400 species labeled Near Threatened through Extinct in the Wild by the [IUCN Red List](https://www.iucnredlist.org/), with 30 images per species.
Top-1 accuracy is reported for both zero-shot and few-shot experiments.

For biological visual tasks beyond species classification, we used:
* [FishNet](https://fishnet-2023.github.io/): We used the original training set (75,631) images to train a two-layer linear classifier on top of the extracted features to predict the feeding path and habitat labels.
Then we tested the classifier with 18,901 images from the test set. Accuracy is reported as the metric, where only predicting all the 9 labels correctly counts as success.
* [NeWT](https://github.com/visipedia/newt): We used the 164 binary classification tasks proposed in the dataset. Micro-accuracy is reported across all the samples.
* [AwA2](https://cvml.ista.ac.at/AwA2/): We used the original train-test split for attribute classification. Macro-F1 score is reported across all the attributes.
* [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6): This is task to discover new species. We implement it as semi-supervised clustering. Clustering accuracy is calculated for the predictions on both seen and unseen classes.
* [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset): 2,598 images of 13 plant species and up to 17 classes of diseases are included in this dataset. We conducted the experiment in a multi-fold 1-shot learning fashion. Average accuracy over the test samples is reported.

More details regarding the evaluation implementation can be referred to in the [paper](https://doi.org/10.48550/arXiv.2505.23883).

### Results
We show the zero-shot classification and non-species classification task results here. For more detailed results, please check the [paper](https://doi.org/10.48550/arXiv.2505.23883).
<table cellpadding="0" cellspacing="0">
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="5">Animals</th>
          <th colspan="4">Plants & Fungi</th>
          <th rowspan="2">Rare Species</th>
          <th rowspan="2">Mean</th>
        </tr>
        <tr>
          <th>NABirds</th>
          <th>Plankton</th>
          <th>Insects</th>
          <th>Insects 2</th>
          <th>Camera Trap</th>
          <th>PlantNet</th>
          <th>Fungi</th>
          <th>PlantVillage</th>
          <th>Med. Leaf</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>CLIP (ViT-L/14)</td>
          <td>66.5</td>
          <td>1.3</td>
          <td>9.0</td>
          <td>11.7</td>
          <td>29.5</td>
          <td>61.7</td>
          <td>7.6</td>
          <td>6.5</td>
          <td>25.6</td>
          <td>35.2</td>
          <td>25.5</td>
        </tr>
        <tr>
          <td>SigLIP</td>
          <td>61.7</td>
          <td>2.4</td>
          <td>27.3</td>
          <td>20.7</td>
          <td>33.7</td>
          <td>81.8</td>
          <td>36.9</td>
          <td><b>28.5</b></td>
          <td>54.5</td>
          <td>47.6</td>
          <td>39.5</td>
        </tr>
        <tr>
          <td>BioTrove-CLIP</td>
          <td>39.4</td>
          <td>1.0</td>
          <td>20.5</td>
          <td>15.7</td>
          <td>10.7</td>
          <td>64.4</td>
          <td>38.2</td>
          <td>15.7</td>
          <td>31.6</td>
          <td>24.6</td>
          <td>26.2</td>
        </tr>
        <tr>
          <td>BioCLIP</td>
          <td>58.8</td>
          <td><b>6.1</b></td>
          <td>34.9</td>
          <td>20.5</td>
          <td>31.7</td>
          <td>88.2</td>
          <td>40.9</td>
          <td>19.0</td>
          <td>38.5</td>
          <td>37.1</td>
          <td>37.6</td>
        </tr>
        <tr>
          <td>BioCLIP 2</td>
          <td><b>74.9</b></td>
          <td>3.9</td>
          <td><b>55.3</b></td>
          <td><b>27.7</b></td>
          <td><b>53.9</b></td>
          <td><b>96.8</b></td>
          <td><b>83.8</b></td>
          <td>25.1</td>
          <td><b>57.8</b></td>
          <td><b>76.8</b></td>
          <td><b>55.6</b></td>
        </tr>
      </tbody>
    </table>

<table cellpadding="0" cellspacing="0">
      <thead>
        <tr>
          <th rowspan="2">Model</th>
          <th colspan="3">Animals</th>
          <th colspan="2">Plants</th>
          <th rowspan="2">Mean</th>
        </tr>
        <tr>
          <th>FishNet</th>
          <th>NeWT</th>
          <th>AwA2</th>
          <th>Herbarium19</th>
          <th>PlantDoc</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>CLIP (ViT-L/14)</td>
          <td>27.9</td>
          <td>83.4</td>
          <td>61.6</td>
          <td>18.2</td>
          <td>22.3</td>
          <td>42.7</td>
        </tr>
        <tr>
          <td>SigLIP</td>
          <td>31.9</td>
          <td>83.2</td>
          <td>67.3</td>
          <td>18.6</td>
          <td>28.2</td>
          <td>45.8</td>
        </tr>
        <tr>
          <td>Supervised-IN21K</td>
          <td>29.4</td>
          <td>75.8</td>
          <td>52.7</td>
          <td>14.9</td>
          <td>25.1</td>
          <td>39.6</td>
        </tr>
        <tr>
          <td>DINOv2</td>
          <td>37.4</td>
          <td>83.7</td>
          <td>48.6</td>
          <td>28.1</td>
          <td>38.6</td>
          <td>47.3</td>
        </tr>
        <tr>
          <td>BioTrove-CLIP</td>
          <td>22.1</td>
          <td>82.5</td>
          <td>45.7</td>
          <td>20.4</td>
          <td>37.7</td>
          <td>41.7</td>
        </tr>
        <tr>
          <td>BioCLIP</td>
          <td>30.1</td>
          <td>82.7</td>
          <td>65.9</td>
          <td>26.8</td>
          <td>39.5</td>
          <td>49.0</td>
        </tr>
        <tr>
          <td>BioCLIP 2</td>
          <td><b>39.8</b></td>
          <td><b>89.1</b></td>
          <td><b>69.5</b></td>
          <td><b>48.6</b></td>
          <td><b>40.4</b></td>
          <td><b>57.5</b></td>
        </tr>
      </tbody>
    </table>

#### Summary

BioCLIP 2 surpasses BioCLIP by 18.0% on zero-shot species classification benchmarks. 
More importantly, although the model is trained to discriminate different species, it also achieves the best performance on tasks beyond species classification.
Notably, BioCLIP 2 yields a 10.2% performance gap over DINOv2, which is broadly used for diverse visual tasks. 

## Model Examination

Please check Section 5.4 of our [paper](https://doi.org/10.48550/arXiv.2505.23883), where we provide formal analysis for the emergent properties of BioCLIP 2.

## Technical Specifications 

### Compute Infrastructure
The training was performed on 32 NVIDIA H100-80GB GPUs distributed over 4 nodes on [Pittsburgh Supercomputing Center](https://www.psc.edu/)'s Bridges-2 Cluster.
It took 10 days to complete the training of 30 epochs.


## Citation

**BibTeX:**
```​
@software{Gu_BioCLIP_2_model,
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  license = {MIT},
  title = {{BioCLIP 2}},
  url = {https://huggingface.co/imageomics/bioclip-2},
  version = {1.0.0},
  doi = {},
  publisher = {Hugging Face},
  year = {2025}
}
```
Please also cite our paper:
```
@article{gu2025bioclip,
  title = {{B}io{CLIP} 2: Emergent Properties from Scaling Hierarchical Contrastive Learning}, 
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G Campolongo and Matthew J Thompson and Net Zhang and Jiaman Wu and Andrei Kopanev and Zheda Mai and Alexander E. White and James Balhoff and Wasila M Dahdul and Daniel Rubenstein and Hilmar Lapp and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  year = {2025},
  eprint={2505.23883},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.23883}, 
}
```

Also consider citing OpenCLIP and BioCLIP:

```
@software{ilharco_gabriel_2021_5143773,
  author={Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  title={OpenCLIP},
  year={2021},
  doi={10.5281/zenodo.5143773},
}
```
Original BioCLIP Model:
```
@software{bioclip2023,
  author = {Samuel Stevens and Jiaman Wu and Matthew J. Thompson and Elizabeth G. Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M. Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  doi = {10.57967/hf/1511},
  month = nov,
  title = {BioCLIP},
  version = {v0.1},
  year = {2023}
}
```
Original BioCLIP Paper:
```
@inproceedings{stevens2024bioclip,
  title = {{B}io{CLIP}: A Vision Foundation Model for the Tree of Life}, 
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
  pages = {19412-19424}
}
```

## Acknowledgements

We would like to thank Zhiyuan Tao, Shuheng Wang, Ziheng Zhang, Zhongwei Wang, and Leanna House for their help with the TreeOfLife-200M dataset, Charles (Chuck) Stewart, Sara Beery, and other [Imageomics Team](https://imageomics.osu.edu/about/team) members for their constructive feedback and Sergiu Sanielevici, Tom Maiden, and TJ Olesky for their dedicated assistance with arranging the necessary computational resources.

We are grateful to Kakani Katija and Dirk Steinke for helpful conversations regarding use and integration of FathomNet and BIOSCAN-5M, respectively, as well as Stephen Formel and Markus Döring for GBIF. We thank Marie Grosjean for comparative methods for filtering citizen science images and Dylan Verheul for assistance with acquiring images from Observation.org from GBIF. We thank Suren Byna for a helpful conversation on early dataset design decisions. We thank Doug Johnson for his collaboration in hosting this large dataset on the Ohio Supercomputer Center research storage file system.

This work was supported by the [Imageomics Institute](https://imageomics.org), which is funded by the US National Science Foundation's Harnessing the Data Revolution (HDR) program under [Award #2118240](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2118240) (Imageomics: A New Frontier of Biological Information Powered by Knowledge-Guided Machine Learning).

Our research is also supported by resources from the [Ohio Supercomputer Center](https://ror.org/01apna436). This work used the [Bridges-2](https://doi.org/10.1145/3437359.3465593) system, which is supported by NSF award number OAC-1928147 at the Pittsburgh Supercomputing Center (PSC), under the auspices of the [NAIRR Pilot program](https://nairrpilot.org/).

Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

<!-- ## Glossary  -->

<!-- [optional] If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

<!-- ## More Information  -->

<!-- [optional] Any other relevant information that doesn't fit elsewhere. -->

## Model Card Authors

Jianyang Gu

## Model Card Contact
[gu.1220@osu.edu](mailto:gu.1220@osu.edu)