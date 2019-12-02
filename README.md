# HiggsToWWML

Setup:
```
cmsrel CMSSW_10_6_6
cd CMSSW_10_6_6/src
cmsenv
git clone https://github.com/jmduarte/HiggsToWWML
cd HiggsToWWML
```

Convert:
```
python convert-uproot.py
```

Train:
```
python train.py
```