# GeoBoost2
GeoBoost2 (part of the ZoDo project) is a system for extracting the location of infected hosts (LOIH) for a given set of GenBank metadata accessions. This involves API implementations that utilize NCBI API's for accessing and parsing information and delivering it to the researcher interested in investigating a set of sequences. You can also use this tool mine Geographic Locations in scientific articles (available as abstracts on PubMed or PMC-OA) or text files. The online version of the ZoDo system is available [here](https://zodo.asu.edu/geoboost2)

The implementation accepts the following inputs through API's:
1. GenBank Accession identifiers (see option ```genbank```)
2. PubMed IDs/ PubMedCentral IDs (see option ```pubmed```)
3. Plain Texts (see option ```text```)

# Installation
GeoBoost2 relies on a few external packages for use:
1. The entire code is based on python. So, firstly install ```python 3``` (tested with version 3.7). The system also depends on a few python packages that can be installed using the command:
```
pip install --upgrade -r requirements.txt
```
2. Next, clone and install [zoophy-geonames](https://github.com/ZooPhy/zoophy-geonames) and start the service by following the instructions listed there. We use Geonames for disabmiguation and normalization. 
3. A file containing word embeddings i.e word vectors that can be loaded using the gensim model. You can download word embeddings trained on PubMed and Wikipedia articles(wikipedia-pubmed-and-PMC-w2v.bin) from http://bio.nlplab.org/ and place the bin file in the ```resources``` directory.

# Running GeoBoost2 
The GeoBoost2 tool can be run in a standalone mode on three types of inputs. To explore the options, use the ```-h``` option i.e.
```
python run.py -h
```

## Running GeoBoost2 for extracting location of infected hosts (LOIH) for GenBank accessions
GenBank accessions (in a list) : This option downloads GenBank accessions and linked PubMed/PMC articles and extracts the location of infected hosts. GeoBoost2 can process GenBank accessions in a list loaded from a file.
```
python run.py genbank examples/ebola_accessions.txt out/ebola/
```
It produces 2 files as output.
  * ```Locations.txt``` containing the best LOIHs for a given accession
  * ```Confidence.txt``` containing multiple possible LOIHs for a given accession along with their respective probabilities
To explore additional options, use the ```-h``` option i.e. ```python run.py genbank -h```


## Running GeoBoost2 for extracting geographic locations from scientific articles (PubMed-IDs / PMC-IDs)
```
python run.py pubmed examples/ebola_pubmedids.txt out/ebola/
```
It produces outputs in a single file.
  * ```Locations.txt``` containing geographic locations mentioned in the articles
To explore additional options, use the ```-h``` option i.e. ```python run.py pubmed -h```

# Additional options 
```
python run.py pubmed examples/ebola_pubmedids.txt out/ebola/
```
It produces outputs in a single file.
  * ```Locations.txt``` containing geographic locations mentioned in the article
To explore additional options, use the ```-h``` option i.e. ```python run.py text -h```

# GeoBoost2 REST API service
GeoBoost2 can run as a service and can be accessed through API. To start the server
```
python server.py
```
To explore additional options, use the ```-h``` option i.e. ```python server.py text -h```

# API Documentation
If you wish to use the API, follow the examples below:
## Process a list of GenBank records
* Type: GET
* Path: /accession?text=\<comma separated Genbank Accessions>
* Example: https://zodo.asu.edu/geoboost2/accession?text=GQ457496,KU497555,AB110657

## Annotate PubMed article or PMC article
* Type: GET
* Path: /pubmed?text=\<comma separated PubMed ids OR comma separated PMC ids with prefix PMC>
* Example: https://zodo.asu.edu/geoboost2/pubmed?text=10325350,10074191
* Example: https://zodo.asu.edu/geoboost2/pubmed?text=PMC84989,PMC104101

## Annotate text
* Type: POST (Data in JSON)
* Path: /resolve
* JSON: "The virus was found in Springfield, Illinois..."
* Example: https://zodo.asu.edu/geoboost2/resolve

# Unit Tests
Tests can be run using:
```
pytest tests/
```

# Server Maintenance
The following fuctions are for internal use only. Additional packages are required for performing these operations.
```
sudo apt-get install libpq-dev
pip install psycopg2
```
## Inserting records into the ZooPhy database
To insert records into ZooPhy database run the following:
```
python maintain.py insert
```
To explore additional options, use the ```-h``` option i.e. ```python maintain.py insert -h```

## Training the Named Entity Recognizer (NER)
To train the NER model for extracting geographic locations from texts run the following:
```
python maintain.py train_ner
```
To explore additional options, use the ```-h``` option i.e. ```python maintain.py insert -h```


# Citation
If you find this work useful you can cite it using:
```
@article{maggeISMB2018bidirectional,
  title={Bi-directional Recurrent Neural Network Models for Geographic Location Extraction in Biomedical Literature},
  author={Magge, Arjun and Weissenbacher, Davy and Sarker, Abeed and Scotch, Matthew and Gonzalez-Hernandez, Graciela},
  journal={Pacific Symposium on Biocomputing},
  publisher={World Scientific Publishing Company},
  year={2018}
}
```
