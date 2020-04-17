# ScopeIt

Implemenation of [ScopeIt: Scoping Task Relevant Sentences in Documents](https://arxiv.org/pdf/2003.04988v1.pdf)

__Note__: copy the data from emw: /home/asafaya19/sentence-classifier/data


```
=====================================================
Document evaluation on Pipeline review data
=====================================================

ScopeIt 512/2-layer fine-tune bert 
=====================================================
              precision    recall  f1-score   support
   macro avg     0.9048    0.9101    0.9068       185

Pipeline current model
===================================================== 
              precision    recall  f1-score   support
   macro avg     0.9179    0.9256    0.9185       185
```

```
=====================================================
Sentence level evaluation on Test and Pipeline data
=====================================================

ScopeIt 512/2-layer fine-tune bert
===================================================== 
              precision    recall  f1-score   support
Test set         0.8768    0.8733    0.8750      3804
Pipeline         0.8730    0.8686    0.8708       831

Pipeline current model
===================================================== 
              precision    recall  f1-score   support
Test set         0.8400    0.8908    0.8614      3804
Pipeline         0.8771    0.9057    0.8890       831
```