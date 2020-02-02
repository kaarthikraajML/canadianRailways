# Dependencies
from flask import Flask, request, jsonify
from sklearn.externals import joblib
from sklearn import preprocessing
from flask_basicauth import BasicAuth
import csv


import traceback
import pandas as pd
import numpy as np
import json

# Your API definition
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = 'password'
basic_auth = BasicAuth(app)

def csvtodict(filename):
    data={}
    with open(filename) as fin:
        reader=csv.reader(fin, skipinitialspace=True)
        for row in reader:
            if not (row):
                continue
            data[row[0]] = int(row[1])
    return data
#input_file = csv.DictReader(open(file))
#print(input_file)
#print(csvtodict(file))

file='../../../Dictionary/RandomForest/Version1/TrackID_dictionary.csv'
#{'S70P0R01': 0, 'S70P0R05': 1, 'W40P0G01': 2, 'W40P0R01': 3, 'W40P0R02': 4, 'W40P0R03': 5, 'W40P0W09': 6, 'W50P0R02': 7, 'W50P0R03': 8, 'W50P0R05': 9, 'W50P0R06': 10, 'W50P0W15': 11, 'RS81-S80P0R01': 12, 'RS92-S90P0R02': 13, 'RS95-S90P0R05': 14, 'S70P0T01': 15, 'S80P0T03': 16, 'TS71-S70P0T01': 17, 'TS72-S70P0T02': 18, 'TS73-S70P0T03': 19, 'TS74-S70P0T04': 20, 'TS75-S70P0T05': 21, 'TW41-W40P0T01': 22, 'TW42-W40P0T02': 23, 'TW43-W40P0T03': 24, 'TW51-W50P0T01': 25, 'W51T01TA': 26, 'W51T04TA': 27, 'undefined': 28}

TrackID_dictionary = csvtodict(file)
#{'1': 0, '2': 1, '34': 2, 'AO 11': 3, 'AO 13': 4, 'AO 15': 5, 'BG01': 6, 'Y022': 7, 'Y023': 8, 'undefined': 9}

print(TrackID_dictionary)
file='../../../Dictionary/RandomForest/Version1/AnchorPattern_dictionary.csv'
AnchorPattern_dictionary = csvtodict(file)

file = '../../../Dictionary/RandomForest/Version1/MegaWorkBlock_dictionary.csv'
MegaWorkBlock_dictionary = csvtodict(file)
#{'No': 0, 'Yes': 1}
file ='../../../Dictionary/RandomForest/Version1/ShadowWorkBlock_dictionary.csv'
ShadowWorkBlock_dictionary = csvtodict(file)
#{'No': 0, 'Yes': 1}
file ='../../../Dictionary/RandomForest/Version1/SplitWorkBlock_dictionary.csv'
SplitWorkBlock_dictionary = csvtodict(file)
#{'No': 0, 'Yes': 1}
#file ='../../../Dictionary/RandomForest/Version1/TieType_dictionary.csv'
#TieType_dictionary = csvtodict(file)
#{'CONCRETE': 0, 'WOOD': 1}
#{'Every 2 nd': 0, 'Every 3 rd': 1, 'Every 4 th': 2, 'Every Tie': 3, 'Pattern 10': 4, 'Pattern 6': 5, 'Pattern 8': 6}
file ='../../../Dictionary/RandomForest/Version1/SpikePattern_dictionary.csv'
SpikePattern_dictionary = csvtodict(file)
#{'3 Spiked': 0, '4 Spiked': 1, '5 Spiked': 2, '6 Spiked': 3, 'B': 4, 'C': 5, 'D': 6, 'NONE': 7}
#TieNumberType_dictionary = {'None': 0, 'Tie Type 1': 1, 'Tie Type 2': 2}
#PadInsulatorType_dictionary  = {'None': 0, 'Pads': 1}
file ='../../../Dictionary/RandomForest/Version1/FastenerTypeTie_dictionary.csv'
FastenerTypeTie_dictionary = csvtodict(file)
#{'CLIP_RAIL': 0, 'None': 1}
file ='../../../Dictionary/RandomForest/Version1/PlateChangeOutRequired_dictionary.csv'
PlateChangeOutRequired_dictionary = csvtodict(file)
#{'No': 0, 'None': 1}
file = '../../../Dictionary/RandomForest/Version1/SubDivision_dictionary.csv'
SubDivision_dictionary = csvtodict(file)
#{'ALBREDA': 0, 'ASHCROFT': 1, 'BRAZEAU': 2, 'CAMROSE': 3, 'CLEARWATER': 4, 'FORT FRANCES': 5, 'NEENAH': 6, 'RAINY': 7, 'RIVERS': 8, 'SPRAGUE': 9, 'SUPERIOR': 10, 'THREE HILLS': 11, 'VEGREVILLE': 12, 'YALE': 13}

file='../../../Dictionary/RandomForest/Version1/RailType_dictionary.csv'
RailType_dictionary= csvtodict(file)
#{'New': 0, 'Old': 1}
file ='../../../Dictionary/RandomForest/Version1/CwrTerritory_dictionary.csv'
CwrTerritory_dictionary = csvtodict(file)
#{'No': 0, 'Yes': 1}
file ='../../../Dictionary/RandomForest/Version1/DestressingMethod_dictionary.csv'
DestressingMethod_dictionary = csvtodict(file)
#{'HEATERS': 0, 'PULLERS': 1}
#file ='../../../Dictionary/RandomForest/Version1/workBlockPlannedMinutes_dictionary.csv'
#workBlockPlannedMinutes_dictionary= csvtodict(file)
#{'60': 0, '180': 1, '200': 2, '240': 3, '300': 4, '360': 5, '420': 6, '440': 7, '480': 8, '600': 9}
file = '../../../Dictionary/RandomForest/Version1/ClosureType_dictionary.csv'
ClosureType_dictionary= csvtodict(file)
#{'No': 0, 'Yes': 1}
file = '../../../Dictionary/RandomForest/Version1/GangId_dictionary.csv'
GangId_dictionary = csvtodict(file)



def data_preprocessing(Rail_df):
     Rail_df[Rail_df.select_dtypes(['object']).columns] = Rail_df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
     #col_list = ['ActivityParentId', 'ActivityId' , 'PlanId', 'Date', 'ConfidenceLevel', 'ProductionType', 'Version','CwrTerritory','RailType','ClosureType','DestressingMethod','OrderNumber']
     #col_list = ['ProductionType', 'Version','CwrTerritory','RailType','ClosureType','DestressingMethod','OrderNumber']
     #tier_df=tier_df.drop(columns=col_list)
     Rail_df['Miles'] = abs(Rail_df['WBMileFrom'] - Rail_df['WBMileTo'])

     encode = preprocessing.LabelEncoder()
     Rail_df['GangId'] = GangId_dictionary[Rail_df['GangId'].values[0]]
     Rail_df['TrackId'] = TrackID_dictionary[Rail_df['TrackId'].values[0]]
     try:
        Rail_df['MegaWorkBlock']=Rail_df['MegaWorkBlock'].cat.add_categories('No')
     except:
        print("")
     Rail_df['MegaWorkBlock'].fillna('No', inplace=True)
     Rail_df['MegaWorkBlock']=MegaWorkBlock_dictionary[Rail_df['MegaWorkBlock'].values[0]]
     try:
        Rail_df['ShadowWorkBlock']=Rail_df['ShadowWorkBlock'].cat.add_categories('No')
     except:
        print("")
     Rail_df['ShadowWorkBlock'].fillna('No', inplace=True)
     Rail_df['ShadowWorkBlock']=ShadowWorkBlock_dictionary[Rail_df['ShadowWorkBlock'].values[0]]
     #Rail_df['SplitWorkBlock'].fillna(0, inplace=True)
     Rail_df['SplitWorkBlock']=SplitWorkBlock_dictionary[Rail_df['SplitWorkBlock'].values[0]]
     #Rail_df['TieType']=TieType_dictionary[Rail_df['TieType'].values[0]]
     Rail_df['RailType']=RailType_dictionary[Rail_df['RailType'].values[0]]
     Rail_df['AnchorPattern']=AnchorPattern_dictionary[Rail_df['AnchorPattern'].values[0]]
     Rail_df['SpikePattern']=SpikePattern_dictionary[Rail_df['SpikePattern'].values[0]]
     Rail_df['CwrTerritory']=CwrTerritory_dictionary[Rail_df['CwrTerritory'].values[0]]
     Rail_df['DestressingMethod']=DestressingMethod_dictionary[Rail_df['DestressingMethod'].values[0]]
     Rail_df['workBlockPlannedMinutes']=Rail_df['workBlockPlannedMinutes']
     Rail_df['ClosureType']=ClosureType_dictionary[Rail_df['ClosureType'].values[0]]
     #tier_df['NumberofInsulatedJoints'].fillna(0, inplace=True)
     #tier_df['NumberofCompromiseRails'].fillna(0, inplace=True)
     #tier_df['TravelTimeDuringBlocks'].fillna(0, inplace=True)
     #tier_df['TieNumberType'].fillna('None', inplace=True)
     #Rail_df['TieNumberType']=TieNumberType_dictionary[Rail_df['TieNumberType'].values[0]]
     #tier_df['NumberofTampers'].fillna(0, inplace=True)
     #tier_df['PadInsulatorType']=tier_df['PadInsulatorType'].cat.add_categories('None')
     #tier_df['PadInsulatorType'].fillna('None', inplace=True)
     #Rail_df['PadInsulatorType']=PadInsulatorType_dictionary[Rail_df['PadInsulatorType'].values[0]]
     #tier_df['FastenerTypeTie']=tier_df['FastenerTypeTie'].cat.add_categories('None')
     #tier_df['FastenerTypeTie'].fillna('None', inplace=True)
     Rail_df['FastenerTypeTie']=FastenerTypeTie_dictionary[Rail_df['FastenerTypeTie'].values[0]]
     #tier_df['PlateChangeOutRequired']=tier_df['PlateChangeOutRequired'].cat.add_categories('None')
     #tier_df['PlateChangeOutRequired'].fillna('None', inplace=True)
     Rail_df['PlateChangeOutRequired']=PlateChangeOutRequired_dictionary[Rail_df['PlateChangeOutRequired'].values[0]]
     Rail_df['SubDivision']=SubDivision_dictionary[Rail_df['SubDivision'].values[0]]
     #tier_df['NumberofTransitionRails'].fillna(0, inplace=True)
     Rail_df['OperationNumber']=Rail_df.OperationNumber.str.extract('(^\d*)')
     Rail_df = Rail_df[Rail_df['OperationNumber'] != ""]
     Rail_df['OperationNumber'] = Rail_df['OperationNumber'].astype('int32')
     return Rail_df

@app.route('/rail/predict', methods=['POST'])
@basic_auth.required
def predictRail():
    if railmodel:
        try:
            json_ = request.json
            print(json_)
            #print(df = df.reindex(columns=list(json_[0].keys())))
            query = pd.read_json(json.dumps(json_),orient='index')
            query.reset_index(level=0,inplace=True)
            columns = query['index']
            values = query.iloc[:,1:2]
            values = values.transpose()
            values.columns = columns
            rail_df_in = values
            columns = ["SubDivision","GangId","OperationNumber","TrackId","MegaWorkBlock","ShadowWorkBlock","SplitWorkBlock","WBMileFrom","WBMileTo","AnchorPattern","SpikePattern","CwrTerritory","DestressingMethod","ClosureType","NumberofInsulatedJoints","NumberofTransitionRails","NumberofCompromiseRails","FastenerTypeTie","PlateChangeOutRequired","RailType","TravelTimeDuringBlocks","workBlockPlannedMinutes"]
            rail_df=pd.DataFrame(columns = columns)
            for i in range(len(rail_df.columns)):
                rail_df[columns[i]] = rail_df_in[columns[i]]
            rail_df["WBMileFrom"] = rail_df["WBMileFrom"].astype("float")
            rail_df["WBMileTo"] = rail_df["WBMileTo"].astype("float")
            rail_df_X = data_preprocessing(rail_df)
            prediction = railmodel.predict(rail_df_X)
            return jsonify({'prediction': str(prediction[0])})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12346 # If you don't provide any port the port will be set to 12345

    railmodel = joblib.load("finalized_model.sav") # Load "model.pkl"

    print ('Model loaded')

    app.run(port=port, debug=True)