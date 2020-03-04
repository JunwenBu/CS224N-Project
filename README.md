# CS224N-Project (Under Construction ...)
## Abstractive Summarization on WikiHow dataset.
- **seq2seq** folder contains files related to Sequence-to-sequence (seq2seq) model.
- **pointer-generator** folder contains files related to Pointer-Generator model. Currently, for this method, we largely reused the code from *[abisee's repository](https://github.com/becxer/pointer-generator/)*, which contains implementation for the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*. 


## Dataset
**WikiHow Dataset** is a new large-scale dataset using the online *[WikiHow](http://www.wikihow.com/)* knowledge base. The dataset is introduced in *[WikiHow Dataset Paper](https://arxiv.org/abs/1810.09305)*. In this project, we used wikihowSep.csv can be downloaded from *[here](https://ucsb.box.com/s/7yq601ijl1lzvlfu4rjdbbxforzd2oag)*.

consisting of each paragraph and its summary:
|Part|Description|
|-------|-------------|
|Title|the title of the article as it appears on the WikiHow knowledge base|
|Overview|the introduction section of the WikiHow articles represented before the paragraphs corresponding to procedures|
|Headline|the bold line (the summary sentence) of the paragraph to serve as the reference summary|
|Text|the paragraph (except the bold line) to generate the article to be summarized|

## Fit data to our models
*[**ProcessRawData.ipynb**](https://github.com/JunwenBu/CS224N-Project/blob/master/ProcessRawData.ipynb)* shows how to clean up data for our seq2seq model.<br>
-Convert words to lowercase.<br>
-Expand language contractions.<br>
-Format words and remove unwanted characters.<br>
-Remove stop words.

*[**DataProcessing-PointerGenerator.ipynb**](https://github.com/JunwenBu/CS224N-Project/blob/master/DataProcessing-PointerGenerator.ipynb)* shows how to prepare the data for the pointer-generator model.<br>
-Prepare raw data and save to csv.<br>
-Tokenize the data.<br>
-Process into .bin and vocab files.

## Demos
- *[**Sum-base.ipynb**](https://github.com/JunwenBu/CS224N-Project/blob/master/Sum-base.ipynb)* shows a local test for our seq2seq model where we overfit a very tiny dataset. <br>
- *[**Test-base.ipynb**](https://github.com/JunwenBu/CS224N-Project/blob/master/Test-base.ipynb)* shows how to use trained model to generate results. <br>
- *[**ROUGE.ipynb**](https://github.com/JunwenBu/CS224N-Project/blob/master/ROUGE.ipynb)* shows how to compute ROUGE-1, ROUGE-2, ROUGE-L and ROUGE-BE. <br>
