"""Methods used to populate ZooPhy database"""
import argparse
import logging
import re
from configparser import ConfigParser
from random import randint
from typing import List, Tuple, Dict
import psycopg2

from zodo.GenBank import GenBankRequest

SQL_INSERT_POSS = "INSERT INTO \"Possible_Location\" (\"Accession\", \"Geoname_ID\", \"Location\", \"Latitude\", \"Longitude\", \"probability\") VALUES(%s,%s,%s,%s,%s,%s)"
SQL_DELETE_POSS = "DELETE FROM \"Possible_Location\" WHERE \"Accession\" = %s"

# clears the table entirely
SQL_CLEAR_POSS = "DELETE FROM \"Possible_Location\""

SQL_QUERY_ALL_ACCESSIONS = "SELECT  DISTINCT \"Accession\" FROM \"Sequence\" ORDER BY \"Accession\" DESC;"
SQL_QUERY_PROCESSED_ACCESSIONS = "SELECT DISTINCT \"Accession\" FROM \"Possible_Location\" ORDER BY \"Accession\" DESC;"

def config(filename='zodo/db/database.ini', section='postgresql') -> Dict[str, str]:
    """Gets the configuration to be used for the database from the database.ini file.
    
    Keyword Arguments:
        filename {str} -- Path to the database.ini file (default: {'zodo/db/database.ini'})
        section {str} -- Section in the .ini file to be loaded (default: {'postgresql'})
    
    Raises:
        Exception: Section not found
    
    Returns:
        Dict[str, str] -- Dictionary object containing the database configuration
    """
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)
    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db

def get_rows_from_db(query: str):
    """Execute a select query and get rows from db in tuple format
    
    Arguments:
        query {str} -- SQL query to be executed
    
    Returns:
        Tuple[str] -- Returns tuples of results
    """
    conn = None
    try:
        params = config()
        logging.debug("Connecting to DB - %s", params['database'])
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        logging.debug("Executing '%s'", query)
        cur.execute(query)
        rows = cur.fetchall()
        logging.debug("Row count: %s", cur.rowcount)
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(error)
    finally:
        if conn is not None:
            conn.close() 
    return rows

def execute(query: str) -> None:
    """ Execute a single query with no returns
        i.e. INSERT, UPDATE, DELETE
    Arguments:
        query {str} -- SQL query to the executed
    """
    conn = None
    try:
        params = config()
        logging.debug("\nConnecting to DB - %s", params['database'])
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        logging.debug("Executing '%s'", query)
        cur.execute(query)
        conn.commit()
        logging.debug("Done running query")
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error("ERROR: %s", error)
    finally:
        if conn is not None:
            conn.close()

def execute_on_many_rows(query: str, rows):
    """ Execute insertion queries for adding multiple rows into the table
    
    Arguments:
        query {str} -- [description]
        rows {Tuple} -- [description]
    """
    conn = None
    try:
        params = config()
        logging.debug("\nConnecting to DB - %s", params['database'])
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        logging.debug("Executing '%s'", query)
        cur.executemany(query, rows)
        conn.commit()
        logging.debug("Done processing %s rows", len(rows))
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error("ERROR: %s", error)
    finally:
        if conn is not None:
            conn.close()

def get_rows(accids: List[str]):
    """ Get possible locations for accession ids
    
    Arguments:
        accids {List[str]} -- List of accession identifiers
    
    Returns:
        List[Tuple] -- List of tuples to be used as rows
    """
    rows = []
    logging.info("Extracting LOIH for %s accessions", len(accids))
    gb_req = GenBankRequest(accids, "ADM1", 10)
    gb_req.process_genbank_ids()
    no_gb_locations = 0
    for gbr in gb_req.genbank_records:
        if gbr.possible_locs:
            # if there were possible locations extracted, then extract rows
            for poss_loc in gbr.possible_locs:
                row = tuple([gbr.accid, int(poss_loc['GeonameId']),
                            re.sub(r" \([^)]*\)", "", poss_loc['FullHierarchy']),
                            float(poss_loc['Latitude']), float(poss_loc['Longitude']),
                            poss_loc['Probability']])
                rows.append(row)
        else:
            # if no locations were extracted, then just use empty values
            # this is so that they are not processed again unnecessarily
            no_gb_locations += 1
            row = tuple([gbr.accid, -1, "", 0, 0, 0])
            # rows.append(row)
    if no_gb_locations > 0:
        logging.info("%s / %s accessions with no locations", no_gb_locations, len(accids))
    return rows

def get_accns_from_file(filepath):
    """ Process accessions in a given file
    
    Arguments:
        filepath {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    accids = [x.strip() for x in open(filepath) if x.strip()]
    logging.debug("Processing accessions: %s", ",".join(accids))
    return accids

def get_accns_from_db() -> List[str]:
    """ Fetches pending accessions to be analyzed from ZooPhy database.
        1) Retrieves all accessions in the database
        2) Retriece list of accessions already processed
        Determine the ac
    Returns:
        List[str] -- List of accession identifiers to be processed
    """
    # get all accessions
    all_accns = [x[0] for x in get_rows_from_db(SQL_QUERY_ALL_ACCESSIONS)]
    # get processed accessions
    proc_accns = set([x[0] for x in get_rows_from_db(SQL_QUERY_PROCESSED_ACCESSIONS)])
    # get unprocessed accessions
    accids = [x for x in all_accns if x not in proc_accns]
    return accids

def insert_into_db(args):
    ''' load accession ids and insert them'''
    # first retrieve accessions either from file or db
    accnids = []
    if args.filepath:
        accnids = get_accns_from_file(args.filepath)
    else:
        accnids = get_accns_from_db()
    # if ids were found, then process them
    if accnids:
        batch_size = args.batch_size
        for index in range(0, len(accnids), batch_size):
            logging.info("\n\nProcessing ZooPhy DB %s-%s/%s accessions\n", index, index+batch_size, len(accnids))
            batch_accessions = accnids[index:index + batch_size]
            logging.info("Accession range: %s-%s\n", batch_accessions[0], batch_accessions[-1])
            rows = get_rows(batch_accessions)
            if rows:
                logging.info("Inserting into DB %s rows from %s accessions \n", len(rows), len(batch_accessions))
                execute_on_many_rows(SQL_INSERT_POSS, rows)

def clear_table() -> None:
    """ Clear the possible location table in ZooPhy database for insertion 

    Returns:
        None -- No return
    """
    logging.warning("\n\n!!!! Warning: This will clear all entries from the table !!!!")
    pin = str(randint(1000, 9999))
    inp_pin = input("Please enter '" + pin + "' to confirm deletion: ")
    print("You entered: " + inp_pin)
    if pin == inp_pin:
        logging.info("Clearing table from database")
        execute(SQL_CLEAR_POSS)
        logging.info("Cleared table")
