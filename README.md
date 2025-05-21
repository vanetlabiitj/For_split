# For_split
This repopsitory is for our paper titled "For-split: Forward-Optimized Federated Split Learning on Edge Devices". Here we provided our testbed setup code files. Simulation code of other model architectures (resnet, vgg), alongwith comparison between different forward only methods will be made available soon.

To execute this code files, First, you need four raspberry pi devices, where you can put all this files one on each, and then connect them with a network wi-fi or bluetooth. 

Second, establish serial pipeline connection between them using socket programming.

Third, In code files put the ip address of next raspberrypi in pipeline for transfering the output for next forward-forward pass.

All datasets used is publicly available online.
