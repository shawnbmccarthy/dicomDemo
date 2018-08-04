import argparse
import logging.config
import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import pymongo
import random
import sys

from pydicom.dataelem import isMultiValue
from pydicom.valuerep import PersonName3
from urllib import request
from zipfile import ZipFile

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

DEFAULT_MONGO_URI = 'mongodb://localhost:27017/dicom'
""" default mongodb connection string """

DEFAULT_DATA_DIR = './data'
""" default data directory to use if not is specified """

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


def setup_parser():
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
    return parser


def generate_random_images(db, count=10, cmap=plt.cm.bone):
    logger.debug('attempting plot random image from datastore')
    sz = db['dcm'].count_documents({})

    for i in range(count):
        skip = random.randint(0, sz-1)
        doc = db['dcm'].find(skip=skip, limit=1).next()
        logger.info('attempting to build image for: %s' % doc['meta']['fn'])
        rows = doc['meta']['r']
        columns = doc['meta']['c']
        n = np.frombuffer(doc['meta']['pixels'], dtype=doc['meta']['dtype'])
        n.shape = (rows, columns)
        plt.imshow(n, cmap=cmap)
        plt.show()


def download_samples(data_dir=DEFAULT_DATA_DIR):
    """
    Download zip file

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


def process_dcm_file(dcm_file):
    """
    process each file, adding some meta data as needed

    :param dcm_file:
    :return:
    """
    logger.info('attempting to read dicom data for %s' % dcm_file)
    ds = pydicom.dcmread(dcm_file)
    data = process_data(ds)

    # add the file_meta data which has to be explicitly asked for
    logger.debug('attempting to add file_meta data for %s' % dcm_file)
    data['file_meta'] = process_data(ds.get('file_meta'))

    # add meta data
    logger.debug('attempting to add meta data for %s' % dcm_file)
    data['meta'] = {
        'fn': dcm_file,
        'dtype': str(ds.pixel_array.dtype),
        'r': ds.pixel_array.shape[0],
        'c': ds.pixel_array.shape[1],
        'pixels': ds.pixel_array.tobytes()
    }
    return data


def main():
    logger.info('starting program')
    parser = setup_parser()
    args = parser.parse_args()
    client = pymongo.MongoClient(args.uri)
    to_insert = []

    try:
        db = client.get_database()
        if args.drop:
            logger.info('dropping database %s before loading new data' % db.name)
            client.drop_database(db.name)
    except Exception as e:
        logger.error('database not set in mongodb uri, cannot continue')
        logger.error('ensure standard mongodb uri has database: mongodb://localhost:27017/dbname')
        logger.error('exception: %s' % e)
        sys.exit(1)

    if args.download:
        download_samples(args.datadir)

    for root, directory, files in os.walk(args.datadir):
        for file in files:
            logger.debug('processing: %s/%s' % (root, file))
            to_insert.append(process_dcm_file('{}/{}'.format(root, file)))
    try:
        result = db['dcm'].insert_many(to_insert)
        logger.info('%s, successfully inserted %d records' % (result.acknowledged, len(result.inserted_ids)))
    except Exception as e:
        logger.error('insert many failure: %s' % e)
        sys.exit(1)

    generate_random_images(db)
    client.close()


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    main()
