# Federated Adaptation for Foundation Model-based Recommendations
Code for ijcai-24 paper. [Federated Adaptation for Foundation Model-based Recommendations](https://arxiv.org/pdf/2405.04840).

Detailed data is on the way!

## Abatract
With the recent success of large language models, particularly foundation models with generalization abilities, applying foundation models for recommendations becomes a new paradigm to improve existing recommendation systems. It becomes a new open challenge to enable the foundation model to capture user preference changes in a timely manner with reasonable communication and computation costs while preserving privacy. This paper proposes a novel federated adaptation mechanism to enhance the foundation model-based recommendation system in a privacy-preserving manner. Specifically, each client will learn a lightweight personalized adapter using its private data. The adapter then collaborates with pre-trained foundation models to provide recommendation service efficiently with fine-grained manners. Importantly, users’ private behavioral data remains secure as it is not shared with the server. This data localization-based privacy preservation is embodied via the federated learning framework. The model can ensure that shared knowledge is incorporated into all adapters while simultaneously preserving each user’s personal preferences. Experimental results on four benchmark datasets demonstrate our method’s superior performance.


## Citation
If you find this project helpful, please consider to cite the following paper:

```
@article{zhang2024federated,
  title={Federated Adaptation for Foundation Model-based Recommendations},
  author={Zhang, Chunxu and Long, Guodong and Guo, Hongkuan and Fang, Xiao and Song, Yang and Liu, Zhaojie and Zhou, Guorui and Zhang, Zijian and Liu, Yang and Yang, Bo},
  journal={arXiv preprint arXiv:2405.04840},
  year={2024}
}
```
