# E2TIMT


E2TIMT: Efficient and Effective Modal Adapter for Text Image Machine Translation

The official repository for ICDAR 2023 Conference paper: 

- **Cong Ma**, Yaping Zhang, Mei Tu, Yang Zhao, Yu Zhou, and Chengqing Zong. **E2TIMT: Efficient and Effective Modal Adapter for Text Image Machine Translation**. In The 17th Document Analysis and Recognition (ICDAR 2023), San José, California, USA. August 21-26, 2023. pp 70–88. Cham. Springer Nature Switzerland. [arXiv](https://arxiv.org/abs/2305.05166), [Springer_Link](https://link.springer.com/chapter/10.1007/978-3-031-41731-3_5)



## 1. Introduction

Text image machine translation (TIMT) aims to translate texts embedded in images from one source language to another target language. Existing methods, both two-stage cascade and one-stage end- to-end architectures, suffer from different issues. The cascade models can benefit from the large-scale optical character recognition (OCR) and MT datasets but the two-stage architecture is redundant. The end-to- end models are efficient but suffer from training data deficiency. To this end, in our paper, we propose an end-to-end TIMT model fully making use of the knowledge from existing OCR and MT datasets to pursue both an effective and efficient framework. More specifically, we build a novel modal adapter effectively bridging the OCR encoder and MT decoder. End-to-end TIMT loss and cross-modal contrastive loss are utilized jointly to align the feature distribution of the OCR and MT tasks. Extensive experiments show that the proposed method outperforms the existing two-stage cascade models and one-stage end-to-end models with a lighter and faster architecture. Furthermore, the ablation studies verify the generalization of our method, where the proposed modal adapter is effective to bridge various OCR and MT models.



<img src="./Figures/model.jpg" style="zoom:100%;" />



## 2. Usage

### 2.1 Requirements

- python==3.6.2
- pytorch == 1.3.1
- torchvision==0.4.2
- numpy==1.19.1
- lmdb==0.99
- PIL==7.2.0
- jieba==0.42.1
- nltk==3.5
- six==1.15.0
- natsort==7.0.1



### 2.2 Train the Model

```shell
bash ./train_model_guide.sh
```



### 2.3 Evaluate the Model

```shell
bash ./test_model_guide.sh
```



### 2.4 Datasets

We use the dataset released in [E2E_TIT_With_MT](https://github.com/EriCongMa/E2E_TIT_With_MT/tree/main).



## 3. Acknowledgement

The reference code of the provided methods are:

- [EriCongMa](https://github.com/EriCongMa)/[**E2E_TIT_With_MT**](https://github.com/EriCongMa/E2E_TIT_With_MT)
- [clovaai](https://github.com/clovaai)/**[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)**
- [OpenNMT](https://github.com/OpenNMT)/**[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)**
- [THUNLP-MT](https://github.com/THUNLP-MT)/**[THUMT](https://github.com/THUNLP-MT/THUMT)**


We thanks for all these researchers who have made their codes publicly available.



## 4. Citation

If you want to cite our paper, please use this bibtex version:

- Springer offered bib citation format

  - ```latex
    @InProceedings{10.1007/978-3-031-41731-3_5,
    author="Ma, Cong
    and Zhang, Yaping
    and Tu, Mei
    and Zhao, Yang
    and Zhou, Yu
    and Zong, Chengqing",
    editor="Fink, Gernot A.
    and Jain, Rajiv
    and Kise, Koichi
    and Zanibbi, Richard",
    title="E2TIMT: Efficient and Effective Modal Adapter for Text Image Machine Translation",
    booktitle="Document Analysis and Recognition - ICDAR 2023",
    year="2023",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="70--88",
    isbn="978-3-031-41731-3"
    }
    ```
  
- DBLP offered bib citation format

  - ```latex
    @inproceedings{DBLP:conf/icdar/MaZTZZZ23a,
      author       = {Cong Ma and
                      Yaping Zhang and
                      Mei Tu and
                      Yang Zhao and
                      Yu Zhou and
                      Chengqing Zong},
      editor       = {Gernot A. Fink and
                      Rajiv Jain and
                      Koichi Kise and
                      Richard Zanibbi},
      title        = {{E2TIMT:} Efficient and Effective Modal Adapter for Text Image Machine
                      Translation},
      booktitle    = {Document Analysis and Recognition - {ICDAR} 2023 - 17th International
                      Conference, San Jos{\'{e}}, CA, USA, August 21-26, 2023, Proceedings,
                      Part {VI}},
      series       = {Lecture Notes in Computer Science},
      volume       = {14192},
      pages        = {70--88},
      publisher    = {Springer},
      year         = {2023},
      url          = {https://doi.org/10.1007/978-3-031-41731-3\_5},
      doi          = {10.1007/978-3-031-41731-3\_5},
      timestamp    = {Fri, 16 Aug 2024 07:47:11 +0200},
      biburl       = {https://dblp.org/rec/conf/icdar/MaZTZZZ23a.bib},
      bibsource    = {dblp computer science bibliography, https://dblp.org}
    }
    ```



If you have any issues, please contact with [email](macong275262544@outlook.com).
