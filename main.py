import argparse
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import pymongo
import random
import sys

from datetime import datetime as dt
from pydicom.dataelem import isMultiValue
from pydicom.valuerep import PersonName3
from urllib import request
from zipfile import ZipFile

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

DEFAULT_MONGO_URI = 'mongodb://localhost:27017/dicom'
""" default mongodb connection string """

DEFAULT_DATA_DIR = './data'
""" default data directory to use if not is specified """

LOAD_MEASURE = 'load'
READ_MEASURE = 'read'
IMG_MEASURE = 'img'
MIX_MEASURE = 'mix'
LOAD_RUNS = 1000   # equates to 1000 * 497 = 497000 documents
READ_RUNS = 10000  # number of random reads
STUDIES = 1000

measures = []

DICOM_SAMPLES = {
    'DICOM_CT_SAMPLE': 'https://www.dicomlibrary.com/?' 
                       'requestType=WADO&' 
                       'studyUID=1.2.826.0.1.3680043.8.1055.1.20111102150758591.92402465.76095170&' 
                       'manage=1b9baeb16d2aeba13bed71045df1bc65',
    'DICOM_MR_SAMPLE': 'https://www.dicomlibrary.com/?' 
                       'requestType=WADO&' 
                       'studyUID=1.2.826.0.1.3680043.8.1055.1.20111103111148288.98361414.79379639&' 
                       'manage=02ef8f31ea86a45cfce6eb297c274598',
    'DICOM_OT_SAMPLE': 'https://www.dicomlibrary.com/?' 
                       'requestType=WADO&' 
                       'studyUID=1.2.826.0.1.3680043.8.1055.1.20111103112244831.40200514.30965937&' 
                       'manage=feb6447a72c9a0a31e1bb4459e547964'
}
""" sample files to download from the dicom library website """


def get_millis(start, end):
    """
    get the milliseconds between two datetime objects

    :param start:
    :param end:
    :return:
    """
    d = (end - start)
    return d.total_seconds() * 1000


def attempt_measures_insert(db, sz=1000):
    """
    :return:
    """
    logging.info('attempting to insert measures')
    if len(measures) > sz:
        try:
            result = db['metrics'].insert_many(measures)
            measures.clear()
            logging.info('%s: inserted %d metrics' % (result.acknowledged, len(result.inserted_ids)))
        except Exception as e:
            logging.error('failed to insert metrics: %s' % e)
            raise e


def setup_parser():
    """
    parse command lines

    :return:
    """
    parser = argparse.ArgumentParser(description='dcm processor demo')
    parser.add_argument(
        '--uri',
        type=str,
        action='store',
        help='mongodb url',
        required=False,
        default=DEFAULT_MONGO_URI
    )
    parser.add_argument(
        '--datadir',
        type=str,
        action='store',
        help='data directory for sample zip files',
        required=False,
        default=DEFAULT_DATA_DIR
    )
    parser.add_argument(
        '--drop',
        action='store_true',
        required=False,
        default=False
    )
    parser.add_argument(
        '--download',
        action='store_true',
        required=False,
        default=False
    )

    parser.add_argument(
        '--measure',
        type=str,
        action='store',
        help='type of use case to measure for',
        default=MIX_MEASURE
    )

    parser.add_argument(
        '--noimg',
        action='store_true',
        required=False,
        default=False
    )
    return parser


def generate_random_images(db, count=10, cmap=plt.cm.bone):
    """
    generate random images

    :param db:
    :param count:
    :param cmap:
    :return:
    """
    logger.debug('attempting plot random image from datastore')
    items = db['dcm'].distinct('meta.study')
    for i in range(count):
        start = dt.now()
        for doc in db['dcm'].find({'meta.study': random.choice(items)}).limit(1):
            logger.info('attempting to build image for: %s' % doc['meta']['fn'])
            rows = doc['meta']['r']
            columns = doc['meta']['c']
            n = np.frombuffer(doc['meta']['pixels'], dtype=doc['meta']['dtype'])
            n.shape = (rows, columns)
            plt.imshow(n, cmap=cmap)
            end = dt.now()
            measures.append({
                'measure': 'img_process',
                'items_processed': 1,
                'process_time': get_millis(start, end),
                'date': dt.now()
            })
            plt.show()


def download_samples(data_dir=DEFAULT_DATA_DIR):
    """
    Download zip file, only needs to be run once and reuse the data

    :param data_dir:
    :return:
    """
    logger.debug('attepting to download samples to %s' % data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for key in DICOM_SAMPLES.keys():
        sample_dir = '{}/{}'.format(data_dir, key)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        zip_f = '{}/{}.zip'.format(sample_dir, key)
        logger.info('downloading data to %s' % zip_f)
        request.urlretrieve(DICOM_SAMPLES[key], zip_f)
        zip_r = ZipFile(zip_f, 'r')
        zip_r.extractall(path=sample_dir)
        zip_r.close()
        os.remove(zip_f)
        logger.info('zip file extracted and deleted')


def process_data(dataset):
    """
    process the the pydicom.Dataset, iterating through each of the data elements to create
    a dictionary

    Note: this is a recursive call, to process all sequences as well

    :param dataset:
    :return:
    """
    logger.debug('attempting to process dataset')
    data = {}
    for element in dataset:
        key = element.tag.to_bytes(4, 'big').hex()
        # skip the array as we will translate the numpy array later
        if key != '7fe00010':
            data[key] = {'vr': element.VR}
            if element.VR == 'SQ':
                data[key]['Value'] = []
                [data[key]['Value'].append(process_data(item)) for item in element]
            else:
                if isMultiValue(element.value):
                    data[key]['Value'] = list(element.value)
                else:
                    if isinstance(element.value, PersonName3):
                        data[key]['Value'] = [element.value.original_string]
                    else:
                        data[key]['Value'] = [element.value]
    return data


def process_dcm_file(dcm_file, version=1, study=1, noimg=False):
    """
    process each file, adding some meta data as needed
    Capture some metrics and store in metrics collection

    :param dcm_file:
    :param version:
    :param study:
    :param noimg:
    :return:
    """
    logger.debug('attempting to read dicom data for %s' % dcm_file)
    start = dt.now()
    ds = pydicom.dcmread(dcm_file, stop_before_pixels=noimg)
    data = process_data(ds)

    # add the file_meta data which has to be explicitly asked for
    logger.debug('attempting to add file_meta data for %s' % dcm_file)
    data['file_meta'] = process_data(ds.get('file_meta'))

    # add meta data
    logger.debug('attempting to add meta data for %s' % dcm_file)
    data['meta'] = {
        'fn': dcm_file,
        'version': version,
        'date': dt.now(),
        'study': study,
        'studyInstanceUID': ds.StudyInstanceUID,
        'seriesInstanceUID': ds.SeriesInstanceUID
    }
    if not noimg:
        data['meta']['dtype'] = str(ds.pixel_array.dtype)
        data['meta']['r'] = ds.pixel_array.shape[0]
        data['meta']['c'] = ds.pixel_array.shape[1]
        data['meta']['pixels'] = ds.pixel_array.tobytes()
    end = dt.now()
    measures.append({
        'measure': 'dcm_process',
        'items_processed': 1,
        'process_time': get_millis(start, end),
        'date': dt.now()
    })
    return data


def insert_data(db, datadir, noimg=False):
    """
    Attempt to insert data

    :param db:
    :param datadir:
    :param noimg:
    :return:
    """
    logger.info('attempting to insert data')
    to_insert = []
    for i in range(1, LOAD_RUNS+1):
        start = dt.now()
        w = 0
        study = random.randint(1, STUDIES)
        for root, directory, files in os.walk(datadir):
            for file in files:
                w = w + 1
                logger.debug('processing: %s/%s' % (root, file))
                to_insert.append(process_dcm_file('{}/{}'.format(root, file), version=i, study=study, noimg=noimg))
        try:
            s_l = dt.now()
            result = db['dcm'].insert_many(to_insert)
            w = w + len(result.inserted_ids)
            e_l = dt.now()
            measures.append({
                'measure': 'insert_time',
                'items_processed': len(result.inserted_ids),
                'process_time': get_millis(s_l, e_l),
                'date': dt.now()
            })
            attempt_measures_insert(db)
            logger.info('%s(%d), successfully inserted %d records' % (result.acknowledged, i, len(result.inserted_ids)))
        except Exception as e:
            logger.error('insert many failure: %s' % e)
            pass
        to_insert.clear()
        end = dt.now()
        # includes all processing include (file processing & inserts!
        measures.append({
            'measure': 'series_time',
            'items_processed': w,
            'process_time': get_millis(start, end),
            'date': dt.now()
        })


def read_data(db):
    """

    :return:
    """
    logger.info('attempting reads')
    start = dt.now()
    total_count = 0
    studies = db['dcm'].distinct('meta.study')
    for i in range(1, READ_RUNS + 1):
        s_r = dt.now()
        count = 0
        for doc in db['dcm'].find({'meta.study': random.choice(studies)}):
            count = count + 1
            logger.debug('doc: %s' % doc['_id'])
        total_count = total_count + count
        e_r = dt.now()
        measures.append({
            'measure': 'single_read',
            'items_processed': count,
            'process_time': get_millis(s_r, e_r),
            'date': dt.now()
        })
        attempt_measures_insert(db)

    end = dt.now()
    measures.append({
        'measure': 'total_read',
        'items_processed': total_count,
        'process_time': get_millis(start, end),
        'date': dt.now()
    })
    attempt_measures_insert(db)
    logger.info('finished in reading records')


def main():
    """

    :return:
    """
    logger.info('starting program')
    parser = setup_parser()
    args = parser.parse_args()
    client = pymongo.MongoClient(args.uri)

    try:
        db = client.get_database()
        if args.drop:
            logger.info('dropping database %s before loading new data' % db.name)
            client.drop_database(db.name)
            db['dcm'].create_index([('meta.study', 1), ('meta.version', 1)])
            db['dcm'].create_index(
                [('meta.studyInstanceUID', 1), ('meta.seriesInstanceUID', 1), ('meta.version', 1)]
            )
            db['dcm'].create_index(
                [('meta.study', 1), ('meta.studyInstanceUID', 1), ('meta.version', 1)]
            )
    except Exception as e:
        logger.error('database not set in mongodb uri, cannot continue')
        logger.error('ensure standard mongodb uri has database: mongodb://localhost:27017/dbname')
        logger.error('exception: %s' % e)
        sys.exit(1)

    if args.download:
        download_samples(args.datadir)

    print(args.measure)
    if args.measure == LOAD_MEASURE or args.measure == MIX_MEASURE:
        insert_data(db, args.datadir, noimg=args.noimg)

    if args.measure == READ_MEASURE or args.measure == MIX_MEASURE:
        read_data(db)

    if not args.noimg and (args.measure == IMG_MEASURE or args.measure == MIX_MEASURE):
        generate_random_images(db)
    attempt_measures_insert(db, 0)
    client.close()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    r_logger = logging.getLogger('m_read')  # capture read metrics
    main()
