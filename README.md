## A Unified Multiplicative Framework for Attribute Learning

Attributes are mid-level semantic properties of objects. Recent research has shown that visual attributes can benefit many traditional learning problems in computer vision community. However, attribute learning is still a challenging problem as the attributes may not always be predictable directly from input images and the variation of visual attributes is sometimes large across categories. In this paper, we propose a unified multiplicative framework for attribute learning, which tackles the key problems. Specifically, images and category information are jointly projected into a shared feature space, where the latent factors are disentangled and multiplied to fulfil attribute prediction. The resulting attribute classifier is category-specific instead of being shared by all  categories. Moreover, our model can leverage auxiliary data to enhance the predictive ability of attribute classifiers, which can reduce the effort of instance-level attribute annotation to some extent. By integrated into an existing deep learning framework, our model can both accurately predict attributes and learn efficient image representations. Experimental results show that our method achieves superior performance on both instance-level and category-level attribute prediction. For zero-shot learning based on attributes, our method significantly improves the state-of-the-art performance on AwA dataset and achieves comparable performance on CUB dataset.

## Reference

If you use this code as part of any published research, please acknowledge the
following papers:

**"A Unified Multiplicative Framework for Attribute Learning."**  
Kongming Liang, Hong Chang, Shiguang Shan, Xilin Chen. *[ICCV](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Liang_A_Unified_Multiplicative_ICCV_2015_paper.pdf)*

	@inproceedings{liang2015unified,
	  title={A Unified Multiplicative Framework for Attribute Learning},
	  author={Liang, Kongming and Chang, Hong and Shan, Shiguang and Chen, Xilin},
	  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
	  pages={2506--2514},
	  year={2015}
	}