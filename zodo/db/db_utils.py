"""File containing datbase utilities"""
import psycopg2
from configparser import ConfigParser
from os.path import join, exists
from os import makedirs
import sys

DATABASES = ['GenBankViruses_Oct2018', 'GenBankViruses_Dec2018',
             'GenBankViruses_Feb2019', 'GenBankViruses_Apr2019', 'GenBankViruses_Jul2019', 'GenBankViruses_Aug2019']

SQL_QUERIES = {
            #    "# Accessions in DB":"SELECT COUNT ( \"Accession\" ) FROM \"Sequence_Details\"",
               "# Accessions with lat_lon or country field":"SELECT  COUNT( DISTINCT \"Accession\") FROM \"Features\" WHERE \"Key\" = 'lat_lon' OR \"Key\" = 'country';",
            #    "# Accessions with lat_lon field":"SELECT  COUNT( DISTINCT \"Accession\") FROM \"Features\" WHERE \"Key\" = 'lat_lon';",
            #    "# Accessions with country field":"SELECT  COUNT( DISTINCT \"Accession\") FROM \"Features\" WHERE \"Key\" = 'country'",
               "# Accessions with info more than country field":"SELECT  COUNT( DISTINCT \"Accession\") FROM \"Features\" WHERE (\"Key\" = 'lat_lon') OR (\"Key\" = 'country' AND \"Value\" LIKE '%:%');",
            #    "# Accessions with directly linked PubMed Ids":"SELECT COUNT ( DISTINCT \"Accession\" ) FROM \"Sequence_Publication\"",
               }

EXPORT_ACCNS_QUERY = "SELECT \"Accession\" FROM \"Sequence_Details\""
EXPORT_ACCNS_COUNTRY_LATLON_QUERY = "SELECT \"Accession\", \"Key\", \"Value\" FROM \"Features\" WHERE \"Key\" = 'lat_lon' OR \"Key\" = 'country';"
EXPORT_ACCNS_PUBMED = "SELECT \"Accession\", \"Pub_ID\" FROM \"Sequence_Publication\""


def config(filename='zodo/db/database.ini', section='postgresql'):
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

def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()
 
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)
      
        # create a cursor
        cur = conn.cursor()
        
   # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')
 
        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)
       
       # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def run_queries(dbname):
    """ query count data from the table """
    conn = None
    try:
        params = config()
        params['database'] = dbname
        print("\nConnecting to DB", dbname)
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        for desc, query in SQL_QUERIES.items():
            cur.execute(query)
            print(desc, "Count:", cur.fetchone())
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close() 

def write_data(cur, filepath):
    """ write data to file """
    print("Writing Count:", cur.rowcount)
    with open(filepath, "w") as outfile:
        row = cur.fetchone()
        while row is not None:
            print('\t'.join(str(i) for i in row), file=outfile)
            row = cur.fetchone()

def export_data(dbname):
    """ export data from tables """
    # Create output directory if it doesn't exist
    outdir = join("out", dbname)
    if not exists(outdir):
        makedirs(outdir)
    conn = None
    try:
        params = config()
        params['database'] = dbname
        print("\nConnecting to DB", dbname)
        conn = psycopg2.connect(**params)

        # # write all accessions to file
        # cur = conn.cursor()
        # cur.execute(EXPORT_ACCNS_QUERY)
        # write_data(cur, join(outdir, "accnids.txt"))

        # # write accessions with geoinfo to file
        # cur = conn.cursor()
        # cur.execute(EXPORT_ACCNS_COUNTRY_LATLON_QUERY)
        # write_data(cur, join(outdir, "accn_country_latlon.txt"))

        # write accessions with geoinfo to file
        cur = conn.cursor()
        cur.execute(EXPORT_ACCNS_PUBMED)
        write_data(cur, join(outdir, "accn_pubmed.txt"))

        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close() 

if __name__ == '__main__':
    # test connection
    connect()
    # get stats
    for dbname in DATABASES[-1:]:
        run_queries(dbname)
        # export_data(dbname)