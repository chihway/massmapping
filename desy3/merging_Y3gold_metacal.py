from glob import glob
import pandas as pd
import pyfits as pf
import numpy as np


def select_files_from_folder(folder_name):
    return glob(folder_name + '/*.csv')


def open_file(file_to_open, columns, index_ref):
    print('Reading file: ', file_to_open)

    try:
        generic_frame = pd.read_csv(file, index_col=index_ref)
        return generic_frame[columns]
    except ValueError:
        print('ValueError: you probably did not pass the correct columns')


if __name__ == "__main__":

    '''
        It loads columns from Y3_gold_catalog and matches them with
        MOF testbed catalogs from Erin. 
        
        easyaccess query:
        SELECT coadd_object_id,ra, dec, BPZ_Z_MEAN, BPZ_Z_MC, flag_gold,flag_foreground,flag_footprint FROM y3_gold_1_0 ; >y3a1_gold.csv
    
        files from Erin (testbeds) can be found here : https://cdcvs.fnal.gov/redmine/projects/deswlwg/wiki/MetacalibrationY3Testbed
    
        MOF appears to have been run on all y3_gold_1_0. Apply footprint_flag afterwards.
    '''

    # input *************************************************************

    folder_Y3_gold = './Y3_GOLD/'
    Erin_file = 'y3v02-mcal-t003b-combined-blind-v1.fits'
    output_file = './MOF_matched_Y3_t003b-combined-blind-v1.csv'
    index = 'COADD_OBJECT_ID'
    index_erin = 'coadd_objects_id'

    Erin_columns = ['e1', 'e2', 'e1_1p', 'e1_2p', 'e2_1p', 'e2_2p',
                    'e1_1m', 'e1_2m', 'e2_1m', 'e2_2m', 'R11', 'R12', 'R21', 'R22',
                    'mask_frac', 'flags', 'mcal_psf_size', 'size', 'snr']

    columns_gold = ['RA', 'DEC', 'BPZ_Z_MEAN', 'BPZ_Z_MC', 'FLAG_GOLD', 'FLAG_FOREGROUND', 'FLAG_FOOTPRINT']

    # load Erin's file ***************************************************

    data = pf.open(Erin_file)

    data = data[1].data
    data_frame_Erin = pd.DataFrame(data[index_erin].byteswap().newbyteorder(), columns=[index])

    for col in Erin_columns:
        data_frame_Erin[col] = data[col].byteswap().newbyteorder()

    data_frame_Erin = data_frame_Erin.set_index(index)

    # match to Y3 *********************************************************

    files = select_files_from_folder(folder_Y3_gold)

    with open(output_file, 'w') as f_out:

        for chunk, file in enumerate(files):
            fram1 = open_file(file, columns_gold, index)
            fram1 = fram1.join(data_frame_Erin)
            mask = ~np.isnan(fram1['R11'])
            fram1 = fram1[mask]

            # *** star galaxy separation *******
            # from Erin..

            mask_sg = (fram1['size'] / fram1['mcal_psf_size'] > 0.5)
            fram1.loc[mask_sg, 'star_gal_sep'] = 1
            fram1.loc[~mask_sg, 'star_gal_sep'] = 0

            # Save to csv ********************
            fram1.to_csv(f_out,
                         columns=fram1.keys(),
                         index=True,
                         header=chunk == 0,
                         mode='a'
                         )
