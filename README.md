# Learning to Incorporate Structure Knowledge for Image Inpainting
Introductions and source code of AAAI 2020 paper *'Learning to Incorporate Structure Knowledge for Image Inpainting'*. You can get the paper in AAAI proceedings or [here](https://www.researchgate.net/publication/338984531_Learning_to_Incorporate_Structure_Knowledge_for_Image_Inpainting).

Code will be coming soon..

## Citation
> @inproceedings{jie2020inpainting,
  title={Learning to Incorporate Structure Knowledge for Image Inpainting},
  author={Jie Yang, Zhiquan Qi, Yong Shi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={3730--3738},
  year={2020}
}

# Introduction
This project develops a multi-task learning framework that attempts to incorporate the image structure knowledge to assist image inpainting, which is not well explored in previous works. The primary idea is to train a shared generator to simultaneously complete the corrupted image and corresponding structures --- edge and gradient, thus implicitly encouraging the generator to exploit relevant structure knowledge while inpainting. In the meantime, we also introduce a structure embedding scheme to explicitly embed the learned structure features into the inpainting process, thus to provide possible preconditions for image completion. Specifically, a novel pyramid structure loss is proposed to supervise structure learning and embedding. Moreover, an attention mechanism is developed to further exploit the recurrent structures and patterns in the image to refine the generated structures and contents. Through multi-task learning, structure embedding besides with attention, our framework takes advantage of the structure knowledge and outperforms several state-of-the-art methods on benchmark datasets quantitatively and qualitatively.

The overview of our multi-task framework is as in figure below. It leverages the structure knowledge with multi-tasking learning (simultaneous image and structure generation), structure embedding and attention mechanism.

![architecture](https://github.com/YoungGod/sturcture-inpainting/blob/master/project-images/architecture.jpg)

# Pyramid structure loss
We propose a pyramid structure loss to guide the structure generation and embedding, thus incorporating the structure information into the generation process.

# Attention Layer
![Attention](https://github.com/YoungGod/sturcture-inpainting/blob/master/project-images/attention.jpg)

Our attention operation is inspired by the non-local mean mechanism which has been used for deionizing and super-resolution. It calculates the response at a position of the output feature map as a weighted sum of the features in the whole input feature map. And the weight or attention score is measured by the feature similarity. Through attention, similar features from surroundings can be transferred to the missing regions to refine the generated contents and structures (e.g. smoothing the artifacts and enhancing the details).
