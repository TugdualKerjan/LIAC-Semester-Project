# Rephrasing the web 🎉 

### Masters semester project at EPFL 2021

---

## Week 2 :



---

## Week 3



---

## Week 4

Seems that there is a goal to work on text quality. BLEU, ROUGE and others can be useful for summarization but they lack being able to give a quality score to a text.

Probably going to use a mixture of multiple methods. Is this really that important ? Seems that using toddler text helps learning.



Installed Detectron2 in myenv environment, with PyTorch 1.7.1 and TorchVision 0.8.2, CPU version. Would like to connect to SCITAS and use that instead.

D2Go is interesting optimised version of Detectron2 but for mobile phones, gotta check it out.

* [Fantastic intro to detectron2](https://www.youtube.com/watch?v=EVtMT6Ve0sY)

* [Traffic sign detection](https://www.youtube.com/watch?v=SWaYRyi0TTs) Could be useful as similar to stickers

* [How to train detectron2 on a custom dataset](https://www.youtube.com/watch?v=CrEW8AfVlKQ)
    * [The blog](https://gilberttanner.com/blog/detectron-2-object-detection-with-pytorch) Where it is explained in great detail


* Datasets:
    * [FlikrLogos](https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/) Have to send email to get dataset ✔
    * [BelgaLogos](http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html#download)

* A [mobile first version](https://github.com/facebookresearch/d2go) of Detectron2 which is light weight

### How to run detecton2 demo:

<details close>
<summary></summary>

- Install packages from [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

- Run after pulling the git

```terminal
git clone https://github.com/facebookresearch/detectron2.git
cd demo
python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --input ../../input.jpg --opts MODEL.DEVICE cpu MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
</details>


### Issues I ran into:
<details close>
<summary></summary>
- Had to add MODEL.DEVICE cpu for it to run on CPU

- Had to point to a downloaded image
```
wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
```

- Had to install two libraries for OpenCV
```
pip install opencv
```
</details>

### What I learned

- How to do Markdown
- Why and how of Conda environments
- How to use detectron pretrained models

- Names of the pretrained

    - R50, R101 is [MSRA Residual Network](https://github.com/KaimingHe/deep-residual-networks)
    - X101 is ResNeXt
    - Use 3x as it is more trained than 1x

---
