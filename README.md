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

![](<CleanShot 2024-03-24 at 11.12.43@2x.png>)
Picture of Qwen 7B trying to predict the next words of James and the Giant Peach. 

![alt arst](<CleanShot 2024-03-24 at 11.14.06@2x.png>)
The hurst score of the text. probably not enough text (250 text), not right model (is the chat one), make it smaller to go faster otherwise for 27000 tokens takes 4h30 to compute, with the smaller qwen 1.8B model takes 1h30 and can batch without memory errors

* [Using taylorswift wiki to try things out ](https://en.m.wikipedia.org/wiki/Taylor_Swift)
Not enough text ? Window context has to be of min 2000 tokens in the paper. Have to remove the first few other wise it has a hard time at the start understanding the text. Move to james and the giant peach for an easier task.

![](<CleanShot 2024-03-24 at 11.22.12@2x.png>)

I didn't understand why the hurst exponents where so high, but it was due to a multitude of errors on my part in the calculation: Needed to normalize, and needed to predict one by one and not all at once.

![alt text](<CleanShot 2024-03-24 at 14.33.18@2x.png>)

Bizarrely it seems like the hurst parameter is under 0.5 for james and the giant peach which is a quite surprising find... Either there is something wrong with the code or this book is bonkers

Interestingly it seems that running on more complicated texts takes longer to infer. Same length, same amount of beep boops. on a paper 

```python
self.filename = "./papers/10.26434_chemrxiv-2022-djr5h.mmd"
```

vs 

```
self.filename = "./james.txt"
```

### Issues I ran into:
<details close>
<summary></summary>
- Didn't understand how the model outputs workd - what exactly is it outputting ?

- The logits weren't shifted correctly

- Hurst library I was using was adding one STEP too large (Adding window_size)
</details>

### What I learned

- How to exploit conda environments
- How to write Dataset classes
- Correctly formatting the batch sizes
- Using tensorboard correctly with SSH port forwarding
- How to use detectron pretrained models
- Using rcp to submit jobs correctly

---

## Week 7

I have managed to get ahold of the hurst pynb of the paper. Doesn't state how to train the omdels for downstream performance but whatever

![alt text](<CleanShot 2024-04-03 at 18.11.17@2x.png>)

Trying different things to get the hurst parameter to go higher, especially by giving more structure to the text. Unfortunately it seems like there is probably a big correlation between perplexity and the hurst parameter 

### Issues I ran into:
<details close>
<summary></summary>
- Understanding why my hurst library wasn't working, how to make it more efficient. In some ways teached me a lot about formatting and how to

- I want to generate my own dataset based on rephrased text of the scientific articles and observe a difference in tensorboard

- It seems like there isn't much difference between the non and rephrased text, apart from a big drop in perplexity.
</details>

### What I learned

- How to exploit conda environments
- How to write Dataset classes
- Correctly formatting the batch sizes
- How to use detectron pretrained models

---
