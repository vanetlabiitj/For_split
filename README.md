# For_split
This repopsitory is for our paper titled "For-split: Forward-Optimized Federated Split Learning on Edge Devices". Here we provided our testbed setup code files.

To execute this code files, First, you need four raspberry pi devices, where you can put all this files one on each, and then connect them with a network wi-fi or bluetooth. 

Second, establish serial pipeline connection between them using socket programming.

Third, In code files put the ip address of next raspberry pi in pipeline for transfering the output for next forward-forward pass.

All datasets used is publicly available online, download them and put on the first (holding input layer) raspberry pi device. For peer-to-peer learning, swap the layers accordingly wherever data is available. 
