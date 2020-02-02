# Learning to Incorporate Structure Knowledge for Image Inpainting
Introductions and source code of AAAI 2020 paper *'Learning to Incorporate Structure Knowledge for Image Inpainting'*. You can get the paper in AAAI proceedings or [here](https://www.researchgate.net/publication/338984531_Learning_to_Incorporate_Structure_Knowledge_for_Image_Inpainting).

Code will be coming soon..

## Citation
```html
@inproceedings{jie2020inpainting,
  title={Learning to Incorporate Structure Knowledge for Image Inpainting},
  author={Jie Yang, Zhiquan Qi, Yong Shi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={3730--3738},
  year={2020}
}
```

# Introduction
This project develops a multi-task learning framework that attempts to incorporate the image structure knowledge to assist image inpainting, which is not well explored in previous works. The primary idea is to train a shared generator to simultaneously complete the corrupted image and corresponding structures --- edge and gradient, thus implicitly encouraging the generator to exploit relevant structure knowledge while inpainting. In the meantime, we also introduce a structure embedding scheme to explicitly embed the learned structure features into the inpainting process, thus to provide possible preconditions for image completion. Specifically, a novel pyramid structure loss is proposed to supervise structure learning and embedding. Moreover, an attention mechanism is developed to further exploit the recurrent structures and patterns in the image to refine the generated structures and contents. Through multi-task learning, structure embedding besides with attention, our framework takes advantage of the structure knowledge and outperforms several state-of-the-art methods on benchmark datasets quantitatively and qualitatively.

The overview of our multi-task framework is as in figure below. It leverages the structure knowledge with multi-tasking learning (simultaneous image and structure generation), structure embedding and attention mechanism.

![architecture](https://github.com/YoungGod/sturcture-inpainting/blob/master/project-images/architecture.jpg)

# Pyramid structure loss
We propose a pyramid structure loss to guide the structure generation and embedding, thus incorporating the structure information into the generation process.
```python
def pyramid_structure_loss(image, predicts, edge_alpha, grad_alpha):
    _, H, W, _ = image.get_shape().as_list()
    loss = 0.
    for predict in predicts:
        _, h, w, _ = predict.get_shape().as_list()
        if h != H:
            gt_img = tf.image.resize_nearest_neighbor(image, size=(h, w))
            
            # grad
            gt_grad = tf.image.sobel_edges(gt_img)
            gt_grad = tf.reshape(gt_grad, [-1, h, w, 6])    # 6 channel
            grad_error = tf.abs(predict - gt_grad)

            # edge
            gt_edge = tf.py_func(canny_edge, [gt_img], tf.float32, stateful=False)
            edge_priority = priority_loss_mask(gt_edge, ksize=5, sigma=1, iteration=2)
        else:
            gt_img = image

            # grad
            gt_grad = tf.image.sobel_edges(gt_img)
            gt_grad = tf.reshape(gt_grad, [-1, H, W, 6])  # 6 channel
            grad_error = tf.abs(predict - gt_grad)

            # edge
            gt_edge = tf.py_func(canny_edge, [gt_img], tf.float32, stateful=False)
            edge_priority = priority_loss_mask(gt_edge, ksize=5, sigma=1, iteration=2)

        grad_loss = tf.reduce_mean(grad_alpha * grad_error)
        edge_weight = edge_alpha * edge_priority
        # print("edge_weight", edge_weight.shape)
        # print("grad_error", grad_error.shape)
        edge_loss = tf.reduce_sum(edge_weight * grad_error) / tf.reduce_sum(edge_weight) / 6.    # 6 channel

        loss = loss + grad_loss + edge_loss

    return loss
```

# Attention Layer
![Attention](https://github.com/YoungGod/sturcture-inpainting/blob/master/project-images/attention.jpg)

Our attention operation is inspired by the non-local mean mechanism which has been used for deionizing and super-resolution. It calculates the response at a position of the output feature map as a weighted sum of the features in the whole input feature map. And the weight or attention score is measured by the feature similarity. Through attention, similar features from surroundings can be transferred to the missing regions to refine the generated contents and structures (e.g. smoothing the artifacts and enhancing the details).
