start: install-deps
	python3 -m streamlit run app.py


install-deps:
	pip install -r requirements.txt