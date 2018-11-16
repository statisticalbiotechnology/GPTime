
GPTime python package

-----------------------------------

1.0 - Dependencies

The GPTime package is now a python3 package and its dependencies can be installed
via

pip install -r requirements.txt

2.0 - Training

To train a model using GPTime, you need a file containing the peptides and
their recorded retention time. The content of the file should be organized
as following :

K.HLNICGTVGSIDNDMSTTDATIGAYSALDRICK.A   245.754
K.AANSVSQDSSYTDFSFTIAGTAHNAHSVTQSASK.V  184.938
K.FATVPTGGASSAAAGAAGAAAGGDAAEEEK.E      150.038
K.IGSGSFGDIYHGTNLISGEEVAIK.L    225.381
K.AASELRILYGGSANGSNAVTFK.D      191.693
K.DAGAISGLNVLRIINEPTAAAIAYGLGAGK.S      256.446
K.ATVDEFPLCVHLVSNELEQLSSEALEAARICANK.Y  256.898
K.GVLGYTEDAVVSSDFLGDSHSSIFDASAGIQLSPK.F 255.529
K.VNLQISDGQPTMCQLEQDYQASDFSVNVK.T       253.647
K.ISAVSTYFESFPYRVNPETGIIDYDTLEK.N       255.285
K.VTDCGDFSYTDLDGSVSDHQGLYVK.L   199.155
K.IPAVEYFGGESPVDVQSQVDSSSVSEDSAVFK.A    252.335

Different training files that were used in our paper can be found in ./Data .

Following command line is an example of how a model is trained. The output model
is saved to model.pk .

python train.py --peptides ./Data/20110922_EXQ4_NaNa_SA_YeastEasy_Labelfree_06.rtimes_q_0.001.tsv --model ./model.pkl --ntrain 100

This model is trained over the first 100 peptides of the data file ./Data/20110922_EXQ4_NaNa_SA_YeastEasy_Labelfree_06.rtimes_q_0.001.tsv
and is saved to ./model.pkl .

3.0 - Prediction

Similarly to predict the retention time for the content of a file :

python test.py --peptides ./Data/20110922_EXQ4_NaNa_SA_YeastEasy_Labelfree_06.rtimes_q_0.001.tsv --model ./model.pkl

This way, we calculate the RT time and Predictive Standard deviation of the
peptides in the file using the model ./model.pkl .

The output of this process for each row is as following :
peptide actual_rt predicted_rt predicted_variance predicted_std
