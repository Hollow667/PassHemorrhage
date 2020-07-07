[!alttag](https://o.remove.bg/downloads/372615ce-904d-4fa6-910c-29ad9b06c0f4/image-removebg-preview.png)

## What is It?
Basically its a model based on [_PassGAN_](https://github.com/brannondorsey/PassGAN) paper along with [_Common User Password Profiler_](https://github.com/Mebus/cupp) for users to create their own password lists with the help of AI.

## How it works
[!alt tag](https://miro.medium.com/max/1400/1*5rMmuXmAquGTT-odw-bOpw.jpeg)
Basically the passgen generates a set of password combo in the folder dataset from keywords given by the user and then the Generator trains on that dataset to generate its own words from that passwordlist and the Discriminator tells if the generated samples are provided by the generator or from the dataset and by repeating the process both the Generator and the Discriminator gets better in their job of generating and detecting passwords.

## Installation
If you are on AMD GPU and Linux (Debian Based Distros):
Run the following in your terminal :

```
sudo apt install libnuma-dev
wget -q -O - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt install rocm-dkms
sudo usermod -a -G video $LOGNAME
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh
sudo reboot
sudo apt update
sudo apt install rocm-libs miopen-hip rccl
pip3 install --user tensorflow-rocm
```
for a more detailed installation docs go to [_Install AMD ROCm_](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)

If you are on CUDA GPU:
Run the following in your terminal :
```
pip3 install tensorflow-gpu
```
## Training your own models
Run the following command in your terminal to start training and generating your passwords:
```
python3 main.py -pg
```

## Credits 
Thanks to both [_brannondorsey_](https://github.com/brannondorsey) and [_Mebus_](https://github.com/Mebus) who made it possible
