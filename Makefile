# Formatting Python code using black and isort
format:
	black .
	isort .

# Run requirements.txt to install dependencies
install:
	pip install -r requirements.txt