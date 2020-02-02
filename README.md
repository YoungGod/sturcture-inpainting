# structure-knowledge-image-inpainting
Source code of AAAI 2020 paper 'Learning to Incorporate Structure Knowledge for Image Inpainting'

Code Coming soon..

# Introduction
This project develops a multi-task learning framework that attempts to incorporate the image structure knowledge to assist image inpainting, which is not well explored in previous works. The primary idea is to train a shared generator to simultaneously complete the corrupted image and corresponding structures --- edge and gradient, thus implicitly encouraging the generator to exploit relevant structure knowledge while inpainting. In the meantime, we also introduce a structure embedding scheme to explicitly embed the learned structure features into the inpainting process, thus to provide possible preconditions for image completion. Specifically, a novel pyramid structure loss is proposed to supervise structure learning and embedding. Moreover, an attention mechanism is developed to further exploit the recurrent structures and patterns in the image to refine the generated structures and contents. Through multi-task learning, structure embedding besides with attention, our framework takes advantage of the structure knowledge and outperforms several state-of-the-art methods on benchmark datasets quantitatively and qualitatively.

The overview of our multi-task framework is as in figure below. It leverages the structure knowledge with multi-tasking learning (simultaneous image and structure generation), structure embedding and attention mechanism.
![architecture](https://github.com/YoungGod/sturcture-inpainting/blob/master/project-images/architecture.jpg)

