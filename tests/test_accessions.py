"""Contains test cases implemented using the pytest framework"""
import pytest
import pandas as pd
import re
import logging
from geopy.distance import geodesic
import requests
import argparse
import sys

logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s',
                    filemode='w', filename='logs/test_acc.log',
                    level=logging.INFO)

from zodo.GenBank import GenBankRequest
from zodo.settings import GEO_HOST, GEO_PORT
from zodo.utils import load_static_objects

GEONAMES_GIDLOOKUP_URL = "http://"+GEO_HOST+":"+GEO_PORT+"/search?query=GeonameId:GIDPH"

GEONAMES_LOOKUP_URL = "http://"+GEO_HOST+":"+GEO_PORT+"/location?location=LPH&count=1&mode=full"

cache = {}

def clean(n):
  return re.sub(r" \([^)]*\)", "", n)

def get_geoname(geonameid):
    if geonameid in cache:
        return cache[geonameid]
    else:
        url = GEONAMES_GIDLOOKUP_URL.replace("GIDPH", str(geonameid))
        response = requests.get(url)
        jsondata = response.json()
        if jsondata and "retrieved" in jsondata and int(jsondata["retrieved"]) > 0:
            loc = jsondata["records"][0]
            cache[geonameid] = loc
        else:
            logging.warning("Search returned no results: '"+ str(geonameid) +"'->'"+url+"'")
            cache[geonameid] = False
            return False
    return loc

def lookup_geoname(geoname):
    if geoname in cache:
        return cache[geoname]
    else:
        url = GEONAMES_LOOKUP_URL.replace("LPH", str(geoname))
        response = requests.get(url)
        jsondata = response.json()
        if jsondata and "retrieved" in jsondata and int(jsondata["retrieved"]) > 0:
            loc = jsondata["records"][0]
            cache[geoname] = loc
        else:
            logging.warning("Search returned no results: '"+ str(geoname) +"'->'"+url+"'")
            cache[geoname] = False
            return False
    return loc

def isSameLoc(gid1, gid2, metric):
    """Metrics calculating if given two geonames objects are same or similar.
    Supported metrics include {id, 50m(i.e.80.5km), 100m(i.e.161km), country, state}
    """
    if not gid1 or not gid2:
        return False
    loc1 = get_geoname(gid1)
    loc2 = get_geoname(gid2)
    if not loc1 or not loc2:
        return False
    if metric=="id":
        return loc1["GeonameId"] == loc2["GeonameId"]
    elif metric in ["50m", "100m"]:
        coords_1 = (loc1["Latitude"], loc1["Longitude"])
        coords_2 = (loc2["Latitude"], loc2["Longitude"])
        dist = geodesic(coords_1, coords_2).miles
        # print(dist)
        same = dist < 50 if metric=="50m" else dist < 100
        return same
    elif metric=="country":
        if "Country" in loc2:
            return "Country" in loc1 and loc1["Country"] == loc2["Country"]
        else:
            return "NA"
    elif metric=="state":
        if "State" in loc2:
            return "State" in loc1 and loc1["State"] == loc2["State"]
        else:
            return "NA"
    elif metric=="county":
        if "County" in loc2:
            return "County" in loc1 and loc1["County"] == loc2["County"]
        else:
            return "NA"
    else:
        return False

def checkLoc(gid, lat, lon, miles=50):
    """Check if gid matches with the lat, lon annotated"""
    if not gid or not str(lat) or not str(lon):
        logging.warning("Invalid inputs %s - %s , %s", gid, lat, lon)
        return False
    loc = get_geoname(gid)
    if not loc:
        logging.warning("Invalid GeonameID: %s", gid)
        return False
    coords_1 = (loc["Latitude"], loc["Longitude"])
    coords_2 = (lat, lon)
    try:
        dist = geodesic(coords_1, coords_2).miles
        same = dist < miles
    except Exception as e:
        logging.warning("Exception: '%s' in coordinates '%s' and '%s'", e, coords_1, coords_2)
        return False
    return same

def get_predictions(df, top=1):
    pred_geoids, pred_locs = {}, {}
    for _, row in df.iterrows():
        if row['Accession'] not in pred_geoids:
            pred_geoids[row['Accession']] = set()
            pred_locs[row['Accession']] = row['Location']
        if len(pred_geoids[row['Accession']]) < top:
            pred_geoids[row['Accession']].add(str(row['GeonameID']))
    return pred_geoids, pred_locs

def evaluate(gold, lookup, metric="id"):
    col = 'Correct_'+metric
    gold[col] = gold.apply(lambda row: isSameLoc(next(iter(lookup[row["Accession"]])), row["GeonameID"], metric=metric) if row["Accession"] in lookup and len(lookup[row["Accession"]])>0 else False, axis=1)
    correct, count = gold[gold[col] == True].Accession.count(), gold[gold[col] != "NA"].Accession.count()
    acc = round((correct*100)/count, 2)
    logging.info("Metric: %s Correct: %s, Count: %s, Accuracy :%s", metric, correct, count, acc)
    return gold, acc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='BIGRU')
    parser.add_argument('--work_dir', type=str, default="resources/")
    parser.add_argument('--save', type=str, default="model/")
    parser.add_argument('--emb_loc', type=str, default='resources/wikipedia-pubmed-and-PMC-w2v.bin',
                        help='word2vec embedding location')
    parser.add_argument('--embvocab', type=int, default=-1)
    parser.add_argument('--genbank_norm', type=str, default="resources/",
                        help='evaluation file for normalizing GenBank accession location in metadata (excluding pubmed)')
    parser.add_argument('--genbank_ete', type=str, default="resources/",
                        help='evaluation file for GenBank accession location in metadata and pubmed texts')
    args = parser.parse_args(args=[])
    return args

def format_genbank_pubmedoa():
    '''Convert xlsx to csv and format contents by adding fields'''
    # process the pubmed excel
    pm_dict, pmc_dict = {}, {}
    for line in open("/home/arjun/dump/oa_file_list.csv"):
        try:
            parts = line.split(",")
            pmcid, pmid = parts[2], parts[-2]
            pm_dict[pmid] = pmcid
            pmc_dict[pmcid] = pmid
        except Exception as e:
            logging.error("ERROR: %s on '%s'", e, line)        
    logging.info("Loaded PM: %s and PMC: %s", len(pm_dict), len(pmc_dict))
    
    gold = pd.read_excel("resources/SDA/v2/TableLocationAndSupplementFullAnnotationsWGeoID_6529.xlsx",
                         header=0, converters={'GeoName ID': int, 'pmid': int, 'pmcid': int})
    gold.rename({'id': 'Accession',
                 'pmcid': 'PMCID',
                 'pmid': 'PubMedID',
                 'country':	'country',
                 'organism': 'organism',
                 'strain': 'strain',
                 'isolate': 'isolate',
                 'Record Location': 'Location_Metadata',
                 'Table Location': 'Location_Article_Table',
                 'Supplemental ': 'Location_Article_Supplemental',
                 'Final Location ': 'Location_Final',
                 'GeoName ID': 'GeonameID',
                 'Lat': 'Latitude',
                 'Long': 'Longitude'}, 
                 axis=1, inplace=True)
    gold['PMCID'] = gold.apply(lambda row: "PMC"+str(row["PMCID"]) if row["PMCID"] else "", axis=1)

    pmc_oa_dict = {}
    for acc, g in gold.groupby("Accession"):
        has_oa = False
        for _, row in g.iterrows():
            if row["PMCID"] in pmc_dict:
                has_oa = True
            else:
                logging.warning("Accession-PMC %s:%s not in PMC-OA", acc, row["PMCID"])
            pmc_oa_dict[row["PMCID"]] = has_oa
        if not has_oa:
            logging.warning("\t\t %s has no associated publications in PMC - Open Access", acc)
    gold['PMID'] = gold.apply(lambda row: pmc_oa_dict[row["PMCID"]], axis=1)
    gold['Open_Access'] = gold.apply(lambda row: pmc_oa_dict[row["PMCID"]], axis=1)
    gold.to_csv("resources/SDA/v3/TableLocationAndSupplement_6529.csv", index=False,
                columns=["Accession","country","organism","strain","isolate","Location_Metadata",
                         "PMCID","Open_Access","Location_Article_Table","Location_Article_Supplemental",
                         "Location_Final","GeonameID","Latitude","Longitude"])
    gold.to_excel("resources/SDA/v3/TableLocationAndSupplement_6529.xlsx", index=False,
                  columns=["Accession","country","organism","strain","isolate","Location_Metadata",
                           "PMCID","Open_Access","Location_Article_Table","Location_Article_Supplemental",
                           "Location_Final","GeonameID","Latitude","Longitude"])

def format_genbank_metadata():
    '''Convert xlsx to csv and format contents by adding fields'''
    # process the metadata excel
    gold = pd.read_excel("resources/SDA/v2/RecordLocationGeoNameID_7719.xlsx",
                         header=0, converters={'GeoName ID': int, 'pmid': int, 'pmcid': int})
    logging.info(gold.columns)
    gold.rename({'Country':	'country',
                 'Organism': 'organism',
                 'Strain': 'strain',
                 'Record Location': 'Location_Metadata',
                 'Record Location (corrected)': 'Location_Final',
                 'Geoname ID_corrected 8/16/19': 'GeonameID'}, 
                 axis=1, inplace=True)
    gold.to_csv("resources/SDA/v3/RecordLocationGeoNameID_7719.csv", index=False,
                columns=["Accession","country","organism","strain","isolate","Location_Metadata",
                         "Location_Final","GeonameID"])
    gold.to_excel("resources/SDA/v3/RecordLocationGeoNameID_7719.xlsx", index=False,
                  columns=["Accession","country","organism","strain","isolate","Location_Metadata",
                           "Location_Final","GeonameID"])

def test_genbank_pubmedoa_performance():
    '''Test the rows in the dataframe'''
    # format_genbank_pubmedoa()
    logging.info("Evaluating performance with insufficient locations in metadata")
    
    gold = pd.read_csv("resources/SDA/v3/TableLocationAndSupplement_6529.csv", converters={'GeonameID': str})
    # validate entries
    gold['GID_Valid'] = gold.apply(lambda row: False if not row['GeonameID'] or row['GeonameID'] != row['GeonameID'] or not row['GeonameID'].strip().isdigit() else True, axis=1)
    gold['LatLon_Valid'] = gold.apply(lambda row: False if not row['Latitude'] or not row['Longitude'] or row['Latitude'] != row['Latitude'] or row['Longitude'] != row['Longitude'] or not str(row['Latitude']).strip().isdigit() or not str(row['Longitude']).strip().isdigit() else True, axis=1)
    gold['Valid'] = gold.apply(lambda row: True if row['GID_Valid'] or row['LatLon_Valid'] else False, axis=1)
    valid_accessions = gold[gold.Valid == True]
    logging.info("Found %s valid entries out of %s rows", valid_accessions.Accession.count(), gold.Accession.count())
    
    logging.info("Testing between GeoBoost v1 and GeoBoost v2 files")

    v1 = pd.read_csv("resources/GeoBoost_v1/articles_6529/Confidence.txt", sep="\t")
    v1_pred, _ = get_predictions(v1, top=1)
    logging.info("Evaluating dataset v1 - %s", len(v1_pred))
    v1_result, v1_acc = evaluate(valid_accessions, v1_pred, metric="50m")
    v1_result.to_csv("resources/GeoBoost_v1/articles_6529/Results.csv", sep="\t")

    v2 = pd.read_csv("resources/GeoBoost_v2/articles_6529/Confidence.txt", sep="\t")
    v2_pred, _ = get_predictions(v2, top=1)
    logging.info("Evaluating dataset v2 - %s", len(v2_pred))
    v2_result, v2_acc = evaluate(valid_accessions, v2_pred, metric="50m")
    v2_result.to_csv("resources/GeoBoost_v2/articles_6529/Results.csv", sep="\t")

    assert v2_acc > v1_acc

def test_genbank_metadata_performance():
    '''Test the rows in the dataframe'''
    logging.info("Evaluating performance with locations in metadata")
    
    gold = pd.read_csv("resources/SDA/v3/RecordLocationGeoNameID_7719.csv", converters={'GeonameID': str})
    # validate entries
    gold['GID_Valid'] = gold.apply(lambda row: False if not row['GeonameID'] or row['GeonameID'] != row['GeonameID'] or not row['GeonameID'].strip().isdigit() else True, axis=1)
    gold['Valid'] = gold.apply(lambda row: True if row['GID_Valid'] else False, axis=1)
    valid_accessions = gold[gold.Valid == True]
    logging.info("Found %s valid entries out of %s rows", valid_accessions.Accession.count(), gold.Accession.count())
    
    logging.info("Testing between GeoBoost v1 and GeoBoost v2 files")

    v1 = pd.read_csv("resources/GeoBoost_v1/metadata_7719/Confidence.txt", sep="\t")
    v1_pred, _ = get_predictions(v1, top=1)
    logging.info("Evaluating dataset v1 - %s", len(v1_pred))
    v1_result, v1_acc = evaluate(valid_accessions, v1_pred, metric="50m")
    v1_result.to_csv("resources/GeoBoost_v1/metadata_7719/Results.csv", sep="\t")

    v2 = pd.read_csv("resources/GeoBoost_v2/metadata_7719/Confidence.txt", sep="\t")
    v2_pred, _ = get_predictions(v2, top=1)
    logging.info("Evaluating dataset v2 - %s", len(v2_pred))
    v2_result, v2_acc = evaluate(valid_accessions, v2_pred, metric="50m")
    v2_result.to_csv("resources/GeoBoost_v2/metadata_7719/Results.csv", sep="\t")

    assert v2_acc > v1_acc

def test_genbank_influenza_study():
    '''Test the rows in the dataframe'''
    logging.info("Evaluating performance with locations in metadata")
    gold = pd.read_csv("resources/SDA/v3/Influenza_CaseStudy_5728_expanded.csv", sep="\t", converters={'GeonameID': str})
    # validate entries
    gold['GID_Valid'] = gold.apply(lambda row: False if not row['GeonameID'] or row['GeonameID'] != row['GeonameID'] or not row['GeonameID'].strip().isdigit() else True, axis=1)
    gold['Valid'] = gold.apply(lambda row: True if row['GID_Valid'] else False, axis=1)
    valid_accessions = gold[gold.Valid == True]
    logging.info("Found %s valid entries out of %s rows", valid_accessions.Accession.count(), gold.Accession.count())
    
    logging.info("Testing between GeoBoost v1 and GeoBoost v2 files")

    adm1 = pd.read_csv("resources/GeoBoost_v2/influenza_5728/ADM1/Confidence.txt", sep="\t")
    adm1_pred, _ = get_predictions(adm1, top=1)
    logging.info("Evaluating dataset at ADM1 - %s", len(adm1_pred))
    results, acc_id = evaluate(valid_accessions, adm1_pred, metric="id")
    results, acc_50m = evaluate(results, adm1_pred, metric="50m")
    results, acc_country = evaluate(results, adm1_pred, metric="country")
    results, acc_state = evaluate(results, adm1_pred, metric="state")
    results, acc_county = evaluate(results, adm1_pred, metric="county")

    logging.info("Accuracy - ID: %s, 50m: %s, Country: %s, State: %s, County: %s",
                 acc_id, acc_50m, acc_country, acc_state, acc_county)
    # adm2 = pd.read_csv("resources/GeoBoost_v2/influenza_5728/ADM2/Confidence.txt", sep="\t")
    # adm2_pred, _ = get_predictions(adm2, top=1)
    # logging.info("Evaluating dataset at ADM2 - %s", len(adm2_pred))
    # adm2_result, adm2_acc = evaluate(valid_accessions, adm2_pred)

    # adm2_result['Correct'] = adm2_result.apply(lambda row: True if row["ADM1_Correct_50m"] or row["ADM2_Correct_50m"] else False, axis=1)
    logging.info("Writing results to file")
    results.to_csv("resources/GeoBoost_v2/influenza_5728/Results.csv", sep="\t")
    logging.info("Done")
    assert 2 > 1
    