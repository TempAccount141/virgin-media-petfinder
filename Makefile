build:
	docker build -t mo-petfinder:latest -f Dockerfile .

train:
	docker run -it --rm -v $$(pwd):/app --name mo-petfinder mo-petfinder:latest --train

infer:
	docker run -it --rm -v $$(pwd):/app --name mo-petfinder mo-petfinder:latest --infer

