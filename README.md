# nnlp_asst
Train:
	python main.py train
	This produces the best model until now, based on the stopping criterion of validation set accuracy. 

Test:
	python main.py test
	This uses the best model produced in the earlier training step and produces the output files for validation and test sets.