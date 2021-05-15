# word-window-classifier
Trying Word Window Classification using PyTorch on Telugu data.

## Key Concepts
* Named Entity Recognition (NER) is an important task in NLP where the goal is to tag different words of the input sentence with labels such as LOCATION, ORGANIZATION,
PERSON etc. 
* The idea of word window classification is to use a context window of surrounding words in addition to the given word to determine the entity tags.
* PyTorch Datasets and Dataloader (and writing a custom collate function)
* Preprocessing input sentences (window-padding, converting the tokens (i.e. words) to indices, padding to make batch of samples to have equal length)
* Defining the model by inheriting from `nn.Module` class
* How does one training step look like
* Main training loop

This is the mini Telugu corpus
```
corpus = [
    'గౌహతి పట్టణం బ్రహ్మపుత్రా నదీ తీరంలో నున్నది',
    'హైద్రాబాదు భారతదేశం లో ఐదవ అతిపెద్ద మహానగరం',
    'అస్సాం రాష్ట్ర రాజధాని గౌహతి',
    'పదకవితా పితామహుడిగా పేరు గాంచిన తాళ్ళపాక అన్నమాచార్యులు కడప జిల్లాలోని తాళ్లపాక అనే గ్రామంలో జన్మించారు',
    'కాకతీయుల రాజధాని ఓరుగల్లు'
    'హైద్రాబాదు కి మరో పేరు భాగ్యనగరం',
    'తిరుమల దేవస్థానం చిత్తూరు జిల్లాలో ఉంది',
]
```

And here is the snapshot of the output
```
vocab size: 39
pad token: <pad> and pad index: 0
unk token: <unk> and unk index: 1
[['గౌహతి', 'పట్టణం', 'బ్రహ్మపుత్రా', 'నదీ', 'తీరంలో', 'నున్నది'], ['హైద్రాబాదు', 'భారతదేశం', 'లో', 'ఐదవ', 'అతిపెద్ద', 'మహానగరం'], ['అస్సాం', 'రాష్ట్ర', 'రాజధాని', 'గౌహతి'], ['పదకవితా', 'పితామహుడిగా', 'పేరు', 'గాంచిన', 'తాళ్ళపాక', 'అన్నమాచార్యులు', 'కడప', 'జిల్లాలోని', 'తాళ్లపాక', 'అనే', 'గ్రామంలో', 'జన్మించారు'], ['కాకతీయుల', 'రాజధాని', 'ఓరుగల్లుహైద్రాబాదు', 'కి', 'మరో', 'పేరు', 'భాగ్యనగరం'], ['తిరుమల', 'దేవస్థానం', 'చిత్తూరు', 'జిల్లాలో', 'ఉంది']]
[[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0]]
Epoch: 0, loss: 0.16540230065584183
Epoch: 10, loss: 0.10624329932034016
Epoch: 20, loss: 0.06460624933242798
Epoch: 30, loss: 0.03658249694854021
Epoch: 40, loss: 0.02548495354130864
Epoch: 50, loss: 0.016815258655697107
Epoch: 60, loss: 0.013529414078220725
Epoch: 70, loss: 0.01128937432076782
Epoch: 80, loss: 0.008093411102890968
Epoch: 90, loss: 0.005300927208736539
original test sentence: తెలంగాణ రాష్ట్ర రాజధాని హైద్రాబాదు
restored test sentence: ['<unk>', 'రాష్ట్ర', 'రాజధాని', 'హైద్రాబాదు']
padded test sentence: ['<pad>', '<pad>', '<unk>', 'రాష్ట్ర', 'రాజధాని', 'హైద్రాబాదు', '<pad>', '<pad>']
Predictions for test sentence: tensor([[0.9842, 0.2969, 0.0515, 0.9391]], grad_fn=<ViewBackward>)
```

There is an excellent tutorial on Stanford cs224n website which can be accessed [here](http://web.stanford.edu/class/cs224n/materials/CS224N_PyTorch_Tutorial.html)

