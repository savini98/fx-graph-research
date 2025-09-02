clean:
	rm -f *.log *.csv *.bak *.nsys-rep *.sqlite

mod:
	chmod +x run_model_fixed-nsys.sh run_model-nsys.sh build-env.sh

env:
	./build-env.sh

run-fixed:
	./run_model_fixed-nsys.sh

run-original:
	./run_model-nsys.sh

.PHONY: clean run-fixed run-original env mod