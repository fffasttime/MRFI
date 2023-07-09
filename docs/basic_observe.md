# Basic observation on LeNet

Let's observe network variable by two utility functions `get_activation_info` and `get_weight_info` in `mrfi.experiment`.

::: mrfi.experiment
    options:
      show_source: false
      members: [get_activation_info, get_weight_info]
      docstring_section_style: list
      show_root_toc_entry: false

## Observing by interactive Python program

In a jupyter environment, we obtain these information of each layer in LeNet:

1. Shape of weights and activaitons
1. Min/Max range of weights and activations
1. Standard deviation of weights and activations
1. Sampling value in tensor and visualize its distribution
1. Visualize its feature map

<iframe 
scrolling = "auto"
height=3000
width=100% 
src="lenet_basic_observe_ipynb.html"  
allowfullscreen>
</iframe>
